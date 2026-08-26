"""
Usage: python bench_worker.py --model_key base --model_path unsloth/Meta-Llama-3.1-8B-Instruct --output results_base.json
Runs the full evaluation suite on a single model in an isolated process.
"""
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

import re
import io
import gc
import json
import time
import zipfile
import argparse
import urllib.request

import torch
from tqdm import tqdm
from datasets import load_dataset

# ============================================================
# ARGS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model_key", required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--n_per_benchmark", type=int, default=100)
parser.add_argument("--max_seq_length", type=int, default=8192)
parser.add_argument("--max_new_tokens", type=int, default=640)
args = parser.parse_args()

SYSTEM_PROMPT = (
    "You are a financial reasoning assistant. Work through the problem "
    'step by step under "Reasoning:", then give the polished final answer '
    'under "Final Answer:".'
)
ANSWER_MARKER = "Final Answer:"
NUM_TOLERANCE = 0.01

# ============================================================
# MODEL LOADING — model-type aware, no cross-model state
# ============================================================
def load_model(model_key, model_path, max_seq_length):
    """Load an Unsloth adapter model OR a plain HF model, fully on one GPU."""
    try:
        from unsloth import FastLanguageModel
        m, tok = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=max_seq_length,
            dtype=torch.bfloat16, load_in_4bit=False,
        )
        FastLanguageModel.for_inference(m)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return m, tok
    except Exception as e:
        print(f"[{model_key}] Unsloth load failed ({e}); falling back to plain transformers.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        m = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True,
        )
        m.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return m, tok

model, tokenizer = load_model(args.model_key, args.model_path, args.max_seq_length)
device = next(model.parameters()).device

# ============================================================
# NUMERIC MATCHING — fixed to handle percent vs decimal scale
# ============================================================
def extract_numbers(text):
    cleaned = text.replace(",", "").replace("₹", "").replace("$", "")
    has_percent = "%" in cleaned
    cleaned = cleaned.replace("%", "")
    nums = re.findall(r"-?\d+\.?\d*", cleaned)
    out = []
    for n in nums:
        try:
            v = float(n)
            out.append(v)
            if has_percent:
                out.append(v / 100.0)   # also register the decimal form
            elif 0 < abs(v) < 1:
                out.append(v * 100.0)   # also register the percent form
        except ValueError:
            pass
    return out

def numeric_match(pred, ref, tol=NUM_TOLERANCE):
    pnums, rnums = extract_numbers(pred), extract_numbers(ref)
    if not rnums:
        return None
    for r in rnums:
        for p in pnums:
            if r == 0:
                if abs(p) < 1e-9:
                    return True
            elif abs(p - r) / abs(r) <= tol:
                return True
    return False

def text_match(pred, ref):
    p = re.sub(r"\s+", " ", pred.lower()).strip()
    r = re.sub(r"\s+", " ", ref.lower()).strip()
    return bool(r) and (r in p or p in r)

def extract_final_answer(text):
    idx = text.find(ANSWER_MARKER)
    if idx != -1:
        return text[idx + len(ANSWER_MARKER):].strip()
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1].strip() if lines else text.strip()

# ============================================================
# GENERIC GENERATE HELPER
# ============================================================
def run_generate(question, max_new_tokens=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()}]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    max_input = args.max_seq_length - (max_new_tokens or args.max_new_tokens) - 64
    if inputs.shape[1] > max_input:
        return None  # signal: skip, too long

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens or args.max_new_tokens,
            max_length=None, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

# ============================================================
# 1. NUMERIC / FINANCIAL-CALC BENCHMARKS  (FinQA, ConvFinQA-style, GSM8K, FinCoT)
# ============================================================
def load_finqa_test(n):
    url = "https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open("FinQA-main/dataset/test.json") as f:
            rows = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    out = []
    for ex in rows[:n]:
        qa = ex.get("qa", {})
        ans = qa.get("answer", "") or (
            str(qa["steps"][-1]["res"]) if qa.get("steps") else ""
        )
        pre = " ".join(ex.get("pre_text", []) or [])
        post = " ".join(ex.get("post_text", []) or [])
        table = ex.get("table", [])
        tbl = "\n".join(" | ".join(str(c) for c in row) for row in table) if table else ""
        ctx = "\n".join(p for p in [pre, tbl, post] if p)
        q = (f"Please answer the given financial question based on the context.\n\n"
             f"Context: {ctx}\n\nQuestion: {qa.get('question','')}")
        if ans:
            out.append({"question": q, "reference": str(ans)})
    return out

def load_gsm8k_test(n):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        ref = ex["answer"].split("####")[-1].strip() if "####" in ex["answer"] else ex["answer"]
        out.append({"question": ex["question"], "reference": ref})
    return out

def load_fincot_holdout(n):
    ds = load_dataset("TheFinAI/FinCoT", split="SFT")
    tail = ds.select(range(max(0, len(ds) - n), len(ds)))
    return [{"question": r["Question"], "reference": (r["Final_response"] or "").strip()}
            for r in tail if r["Final_response"]]

def run_numeric_benchmark(name, samples):
    correct = fmt_ok = skipped = 0
    details = []
    for s in tqdm(samples, desc=name, leave=False):
        full = run_generate(s["question"])
        if full is None:
            skipped += 1
            continue
        if ANSWER_MARKER in full:
            fmt_ok += 1
        pred = extract_final_answer(full)
        nm = numeric_match(pred, s["reference"])
        ok = (nm is True) or (nm is None and text_match(pred, s["reference"]))
        correct += int(ok)
        details.append({"ref": s["reference"], "pred": pred[:200], "correct": ok})
    n = len(details)
    return {
        "n": n, "skipped": skipped,
        "accuracy": correct / n * 100 if n else 0.0,
        "format_compliance": fmt_ok / n * 100 if n else 0.0,
        "details": details,
    }

# ============================================================
# 2. TASK-SPECIFIC ACCURACY — Financial sentiment classification
#    (Financial PhraseBank-style: classify sentiment of financial news)
# ============================================================
def run_sentiment_benchmark(n):
    try:
        ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
    except Exception as e:
        print(f"Could not load Financial PhraseBank: {e}")
        return None

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    y_true, y_pred = [], []
    for ex in tqdm(ds, desc="Sentiment", leave=False):
        q = (f'Classify the sentiment of this financial statement as exactly one '
             f'word — positive, neutral, or negative:\n\n"{ex["sentence"]}"\n\n'
             f'Answer with just the single word under "Final Answer:".')
        full = run_generate(q, max_new_tokens=200)
        if full is None:
            continue
        pred_text = extract_final_answer(full).lower()
        pred_label = next((l for l in ("positive", "negative", "neutral") if l in pred_text), "neutral")
        y_true.append(label_map[ex["label"]])
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
        labels=["positive", "neutral", "negative"],
    )
    return {"n": len(y_true), "accuracy": acc * 100, "precision": p, "recall": r, "f1": f1}

# ============================================================
# 3. GENERATION QUALITY — BLEU / ROUGE / BERTScore on FiQA-style QA
# ============================================================
def run_generation_quality_benchmark(n):
    try:
        ds = load_dataset("ChanceFocus/flare-fiqasa", split="test")  # FiQA-style QA/sentiment
    except Exception as e:
        print(f"Could not load a FiQA-style set: {e}")
        return None

    import evaluate
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    try:
        bertscore = evaluate.load("bertscore")
    except Exception:
        bertscore = None

    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    preds, refs = [], []
    for ex in tqdm(ds, desc="GenQuality", leave=False):
        query = ex.get("query") or ex.get("text") or ""
        ref = str(ex.get("answer") or ex.get("gold") or "")
        if not query or not ref:
            continue
        full = run_generate(query, max_new_tokens=300)
        if full is None:
            continue
        pred = extract_final_answer(full)
        preds.append(pred)
        refs.append(ref)

    if not preds:
        return None

    rouge_res = rouge.compute(predictions=preds, references=refs)
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    result = {"n": len(preds), "rougeL": rouge_res["rougeL"], "bleu": bleu_res["score"]}
    if bertscore:
        bs = bertscore.compute(predictions=preds, references=refs, lang="en")
        result["bertscore_f1"] = sum(bs["f1"]) / len(bs["f1"])
    return result

# ============================================================
# 4. DOMAIN CONSISTENCY — financial terminology usage check
# ============================================================
FINANCE_GLOSSARY = [
    "ebitda", "revenue", "margin", "roi", "return on investment", "capital",
    "cash flow", "liquidity", "leverage", "amortization", "depreciation",
    "equity", "debt", "valuation", "npv", "irr", "working capital", "yield",
]

def run_domain_consistency_check(samples):
    """Fraction of financial-question responses that use recognizable finance terms."""
    hits = 0
    for s in tqdm(samples, desc="DomainConsistency", leave=False):
        full = run_generate(s["question"])
        if full is None:
            continue
        low = full.lower()
        if any(term in low for term in FINANCE_GLOSSARY):
            hits += 1
    n = len(samples)
    return {"n": n, "domain_term_usage_pct": hits / n * 100 if n else 0.0}

# ============================================================
# 5. HALLUCINATION / RELIABILITY STRESS TEST
#    Tricky prompts about rare/fictional instruments — model should hedge,
#    not fabricate specific numbers or facts.
# ============================================================
HALLUCINATION_PROMPTS = [
    "What is the current SEC-mandated capital adequacy ratio for a 'Zeta-Class "
    "Perpetual Reinsurance Note'?",  # fictional instrument
    "Quote the exact clause number in RBI's Master Circular that fixes the "
    "interest rate on a 'Sunrise Convertible Debenture'.",  # fabricated named clause
    "What was Apple's exact EBITDA margin for the fiscal quarter ending "
    "June 31, 2024?",  # June has no 31st — trap
    "Cite the precise Basel IV liquidity coverage ratio threshold introduced "
    "in 2019 for micro-cap fintech firms.",  # Basel IV finalized later; trap
]

HEDGE_PHRASES = [
    "i don't have", "not available", "cannot confirm", "no such", "does not exist",
    "not aware of", "unable to verify", "i'm not sure", "insufficient information",
    "no reliable data", "fictional", "not a real",
]

def run_hallucination_test():
    results = []
    for prompt in tqdm(HALLUCINATION_PROMPTS, desc="Hallucination", leave=False):
        full = run_generate(prompt, max_new_tokens=300)
        if full is None:
            continue
        low = (full or "").lower()
        hedged = any(h in low for h in HEDGE_PHRASES)
        fabricated_number = bool(re.search(r"\d+\.\d+%|\d+%", low)) and not hedged
        results.append({
            "prompt": prompt, "hedged": hedged,
            "fabricated_specific_number": fabricated_number,
            "response": full[:300],
        })
    n = len(results)
    hedge_rate = sum(r["hedged"] for r in results) / n * 100 if n else 0.0
    fabrication_rate = sum(r["fabricated_specific_number"] for r in results) / n * 100 if n else 0.0
    return {"n": n, "hedge_rate_pct": hedge_rate,
            "fabrication_rate_pct": fabrication_rate, "details": results}

# ============================================================
# RUN ALL SECTIONS FOR THIS MODEL
# ============================================================
print(f"\n{'='*70}\nEVALUATING: {args.model_key} ({args.model_path})\n{'='*70}")

results = {"model_key": args.model_key, "model_path": args.model_path}

n = args.n_per_benchmark
results["FinQA"]        = run_numeric_benchmark("FinQA", load_finqa_test(n))
results["GSM8K"]        = run_numeric_benchmark("GSM8K", load_gsm8k_test(n))
results["FinCoT"]       = run_numeric_benchmark("FinCoT", load_fincot_holdout(n))
results["Sentiment"]    = run_sentiment_benchmark(min(n, 100))
results["GenQuality"]   = run_generation_quality_benchmark(min(n, 50))
results["DomainConsistency"] = run_domain_consistency_check(load_fincot_holdout(min(n, 30)))
results["Hallucination"]     = run_hallucination_test()

with open(args.output, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results for {args.model_key} -> {args.output}")

# clean shutdown
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
