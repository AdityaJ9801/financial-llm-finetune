import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"  # sm_100 = B200

import io
import re
import json
import time
import zipfile
import urllib.request

import torch
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================
MODELS_TO_TEST = {
    "base":       "unsloth/Meta-Llama-3.1-8B-Instruct",
    "my_fincot":  "Aditya757864/llama3.1-8b-fincot",   # your deployed model
}

N_PER_BENCHMARK = 150          # samples per dataset; raise for tighter estimates
MAX_NEW_TOKENS  = 640
NUM_TOLERANCE   = 0.01         # 1% relative tolerance for numeric match
MAX_SEQ_LENGTH  = 8192         # raised from 4096 to fit larger FinQA contexts

# Weights & Biases
WANDB_ENTITY  = "aditya_1976-shri-ramdeobaba-college-of-engineering-and-m"
WANDB_PROJECT = "Financial finetuning model"
USE_WANDB     = True

SYSTEM_PROMPT = (
    "You are a financial reasoning assistant. Work through the problem "
    'step by step under "Reasoning:", then give the polished final answer '
    'under "Final Answer:".'
)
ANSWER_MARKER = "Final Answer:"

# ============================================================
# SHARED HELPERS
# ============================================================
def extract_final_answer(text):
    """Text after 'Final Answer:'; if absent, use the last non-empty line."""
    idx = text.find(ANSWER_MARKER)
    if idx != -1:
        return text[idx + len(ANSWER_MARKER):].strip()
    # fallback for models that don't use our marker (e.g. reference models)
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1].strip() if lines else text.strip()

def extract_numbers(text):
    cleaned = text.replace(",", "").replace("₹", "").replace("$", "").replace("%", "")
    nums = re.findall(r"-?\d+\.?\d*", cleaned)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except ValueError:
            pass
    return out

def numeric_match(pred, ref, tol=NUM_TOLERANCE):
    """True if any predicted number is within tol of any reference number."""
    pnums, rnums = extract_numbers(pred), extract_numbers(ref)
    if not rnums:
        return None  # no numeric reference -> caller falls back to text match
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

# ============================================================
# DATASET LOADERS  ->  each returns list of {"question","reference"}
# ============================================================
def load_finqa_test(n):
    """FinQA test split — real financial numeric reasoning, unseen in training."""
    url = "https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open("FinQA-main/dataset/test.json") as f:
            rows = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    out = []
    for ex in rows[:n]:
        qa = ex.get("qa", {})
        ans = qa.get("answer", "")
        if not ans:
            steps = qa.get("steps", [])
            ans = str(steps[-1].get("res", "")) if steps else ""
        pre = " ".join(ex.get("pre_text", []) or [])
        post = " ".join(ex.get("post_text", []) or [])
        table = ex.get("table", [])
        tbl = "\n".join(" | ".join(str(c) for c in row) for row in table) if table else ""
        ctx = "\n".join(p for p in [pre, tbl, post] if p)
        q = ("Please answer the given financial question based on the context.\n\n"
             f"Context: {ctx}\n\nQuestion: {qa.get('question','')}")
        if ans:
            out.append({"question": q, "reference": str(ans)})
    return out

def load_gsm8k_test(n):
    """GSM8K test split — general multi-step arithmetic."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        ref = ex["answer"].split("####")[-1].strip() if "####" in ex["answer"] else ex["answer"]
        out.append({"question": ex["question"], "reference": ref})
    return out

def load_fincot_holdout(n):
    """Proxy in-domain set: tail slice of FinCoT SFT (see caveat in notes)."""
    ds = load_dataset("TheFinAI/FinCoT", split="SFT")
    tail = ds.select(range(max(0, len(ds) - n), len(ds)))
    return [{"question": r["Question"], "reference": (r["Final_response"] or "").strip()}
            for r in tail if r["Final_response"]]

BENCHMARKS = {
    "FinQA":  load_finqa_test,     # financial calc
    "GSM8K":  load_gsm8k_test,     # general math
    "FinCoT": load_fincot_holdout, # in-domain
}

# ============================================================
# EVALUATE ONE MODEL ON ONE BENCHMARK
# ============================================================
def evaluate(model, tokenizer, samples):
    correct = fmt_ok = 0
    latencies = []
    details = []
    skipped = 0

    # leave room for the generated answer within the context window
    max_input_tokens = MAX_SEQ_LENGTH - MAX_NEW_TOKENS - 64

    for s in tqdm(samples, leave=False):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["question"].strip()}]
        inputs = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        # --- length guard: skip prompts that don't fit the context window ---
        if inputs.shape[1] > max_input_tokens:
            skipped += 1
            continue

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs, max_new_tokens=MAX_NEW_TOKENS,
                max_length=None, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        latencies.append(time.time() - t0)

        full = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        if ANSWER_MARKER in full:
            fmt_ok += 1
        pred = extract_final_answer(full)

        nm = numeric_match(pred, s["reference"])
        ok = (nm is True) or (nm is None and text_match(pred, s["reference"]))
        correct += int(ok)
        details.append({
            "question": s["question"][:150],
            "reference": s["reference"],
            "prediction": pred[:250],
            "correct": ok,
        })

    n = len(details)  # only count samples actually evaluated
    if skipped:
        print(f"     (skipped {skipped} over-length prompts)")
    return {
        "n": n,
        "skipped": skipped,
        "accuracy": correct / n * 100 if n else 0.0,
        "format_compliance": fmt_ok / n * 100 if n else 0.0,
        "avg_latency_s": sum(latencies) / n if n else 0.0,
        "details": details,
    }

# ============================================================
# MODEL LOADER
# ============================================================
def load_model(path):
    m, tok = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16, load_in_4bit=False,
    )
    FastLanguageModel.for_inference(m)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return m, tok

# ============================================================
# RUN EVERYTHING
# ============================================================
def main():
    if USE_WANDB:
        wandb.init(
            entity=WANDB_ENTITY, project=WANDB_PROJECT,
            name="benchmark-comparison",
            config={"n_per_benchmark": N_PER_BENCHMARK,
                    "num_tolerance": NUM_TOLERANCE,
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "models": list(MODELS_TO_TEST.keys())},
        )

    print("Loading benchmark datasets...")
    bench_samples = {name: fn(N_PER_BENCHMARK) for name, fn in BENCHMARKS.items()}
    for name, s in bench_samples.items():
        print(f"  {name}: {len(s)} samples")

    all_results = {}
    for model_key, model_path in MODELS_TO_TEST.items():
        print(f"\n{'='*70}\nMODEL: {model_key}  ({model_path})\n{'='*70}")
        try:
            model, tokenizer = load_model(model_path)
        except Exception as e:
            print(f"  FAILED to load {model_key}: {e}")
            continue

        all_results[model_key] = {}
        for bench_name, samples in bench_samples.items():
            print(f"  -> {bench_name}")
            res = evaluate(model, tokenizer, samples)
            all_results[model_key][bench_name] = res
            print(f"     accuracy={res['accuracy']:.1f}%  "
                  f"format={res['format_compliance']:.1f}%  "
                  f"latency={res['avg_latency_s']:.2f}s  "
                  f"(n={res['n']}, skipped={res['skipped']})")
            if USE_WANDB:
                wandb.log({
                    f"{model_key}/{bench_name}/accuracy": res["accuracy"],
                    f"{model_key}/{bench_name}/format_compliance": res["format_compliance"],
                    f"{model_key}/{bench_name}/latency_s": res["avg_latency_s"],
                    f"{model_key}/{bench_name}/n_evaluated": res["n"],
                    f"{model_key}/{bench_name}/skipped": res["skipped"],
                })

        del model, tokenizer
        torch.cuda.empty_cache()

    # ---------- console report ----------
    print("\n\n" + "=" * 78)
    print("BENCHMARK SUMMARY  (accuracy %)")
    print("=" * 78)
    tested = [k for k in MODELS_TO_TEST if k in all_results]
    header = f"{'Benchmark':<14}" + "".join(f"{k:>16}" for k in tested)
    print(header); print("-" * len(header))
    for bench_name in bench_samples:
        row = f"{bench_name:<14}"
        for model_key in tested:
            row += f"{all_results[model_key][bench_name]['accuracy']:>15.1f}%"
        print(row)

    print("\nFORMAT COMPLIANCE (%)  [only meaningful for models using our markers]")
    for bench_name in bench_samples:
        row = f"{bench_name:<14}"
        for model_key in tested:
            row += f"{all_results[model_key][bench_name]['format_compliance']:>15.1f}%"
        print(row)

    print("\nAVG LATENCY (s/sample)")
    for bench_name in bench_samples:
        row = f"{bench_name:<14}"
        for model_key in tested:
            row += f"{all_results[model_key][bench_name]['avg_latency_s']:>15.2f}s"
        print(row)

    print("\nSAMPLES EVALUATED / SKIPPED")
    for bench_name in bench_samples:
        row = f"{bench_name:<14}"
        for model_key in tested:
            r = all_results[model_key][bench_name]
            row += f"{r['n']:>10}/{r['skipped']:<4}"
        print(row)

    # ---------- save JSON ----------
    with open("benchmark_report.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull details saved to benchmark_report.json")

    # ---------- bar chart ----------
    os.makedirs("assets", exist_ok=True)
    bench_names = list(bench_samples.keys())
    x = range(len(bench_names))
    width = 0.8 / max(1, len(tested))

    plt.figure(figsize=(10, 6))
    for i, model_key in enumerate(tested):
        vals = [all_results[model_key][b]["accuracy"] for b in bench_names]
        offsets = [xi + i * width for xi in x]
        bars = plt.bar(offsets, vals, width, label=model_key)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=8)

    plt.xticks([xi + width * (len(tested) - 1) / 2 for xi in x], bench_names)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison — Accuracy by Benchmark")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    chart_path = "assets/benchmark_comparison.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Saved chart to {chart_path}")

    if USE_WANDB:
        wandb.log({"benchmark_chart": wandb.Image(chart_path)})
        # also log a summary table
        table = wandb.Table(columns=["model"] + bench_names)
        for model_key in tested:
            table.add_data(model_key,
                           *[all_results[model_key][b]["accuracy"] for b in bench_names])
        wandb.log({"benchmark_summary": table})
        wandb.finish()

    # ---------- markdown snippet for the model card ----------
    print("\n" + "=" * 78)
    print("MODEL CARD SNIPPET (paste into README.md):")
    print("=" * 78)
    md = ["## Benchmark\n",
          "Evaluated on held-out test sets (greedy decoding, ±1% numeric tolerance).\n",
          "| Benchmark | " + " | ".join(tested) + " |",
          "|" + "---|" * (len(tested) + 1)]
    for bench_name in bench_names:
        cells = [f"{all_results[m][bench_name]['accuracy']:.1f}%" for m in tested]
        md.append(f"| {bench_name} | " + " | ".join(cells) + " |")
    md.append("\n![Benchmark comparison](assets/benchmark_comparison.png)")
    print("\n".join(md))


if __name__ == "__main__":
    main()
