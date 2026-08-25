import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"  # sm_100 = B200

import io
import re
import json
import time
import zipfile
import urllib.request
import numpy as np

import torch
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_dataset

# ============================================================
# CONFIGURATION
# ============================================================
MODELS_TO_TEST = {
    "base":       "unsloth/Meta-Llama-3.1-8B-Instruct",
    "my_fincot":  "Aditya757864/llama3.1-8b-fincot",   # your deployed adapter model
}

N_PER_BENCHMARK = 150          # samples per dataset
MAX_NEW_TOKENS  = 640
NUM_TOLERANCE   = 0.01         # 1% relative tolerance for numeric match
MAX_SEQ_LENGTH  = 8192         
PRINT_FAILURES  = False        # Set to True to debug exact mismatches in the console

# Weights & Biases
WANDB_ENTITY  = "aditya_1976-shri-ramdeobaba-college-of-engineering-and-m"
WANDB_PROJECT = "Financial finetuning model"
USE_WANDB     = False          # Toggle True/False depending on if you want WandB tracking

SYSTEM_PROMPT = (
    "You are a financial reasoning assistant. Work through the problem "
    'step by step under "Reasoning:", then give the polished final answer '
    'under "Final Answer:".'
)
ANSWER_MARKER = "Final Answer:"

# ============================================================
# SHARED HELPERS & METRICS
# ============================================================
def extract_final_answer(text):
    """Extracts text exactly after 'Final Answer:'."""
    idx = text.find(ANSWER_MARKER)
    if idx != -1:
        return text[idx + len(ANSWER_MARKER):].strip()
    
    # Fallback if marker is missing
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1].strip() if lines else text.strip()

def extract_numbers(text):
    # Strip common financial formatting symbols before regex
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
    """Returns True if any predicted number is within 1% of any reference number."""
    pnums = extract_numbers(pred)
    rnums = extract_numbers(ref)
    
    if not rnums:
        return None  # No numbers in reference, fallback to text match
        
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
# DATASET LOADERS
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

BENCHMARKS = {
    "FinQA":  load_finqa_test,
    "GSM8K":  load_gsm8k_test,
    "FinCoT": load_fincot_holdout,
}

# ============================================================
# EVALUATION LOOP
# ============================================================
def evaluate(model, tokenizer, samples, bench_name):
    correct = fmt_ok = 0
    latencies = []
    details = []
    skipped = 0

    max_input_tokens = MAX_SEQ_LENGTH - MAX_NEW_TOKENS - 64

    for s in tqdm(samples, desc=f"Evaluating {bench_name}", leave=False):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["question"].strip()}]
        
        inputs = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

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
        
        # --- DEBUG PRINTS ---
        if not ok and PRINT_FAILURES:
            print("\n" + "="*50)
            print(f"FAILED ON: {bench_name}")
            print(f"Extracted Answer Text : {pred}")
            print(f"Extracted Numbers     : {extract_numbers(pred)}")
            print(f"Reference Numbers     : {extract_numbers(s['reference'])}")
            print("="*50)

        details.append({
            "question": s["question"][:150],
            "reference": s["reference"],
            "prediction": pred[:250],
            "correct": ok,
        })

    n = len(details)
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
    """
    Unsloth handles loading both base models and PEFT adapters cleanly.
    device_map={"": 0} forces all weights off 'meta' device onto GPU.
    """
    m, tok = FastLanguageModel.from_pretrained(
        model_name=path, 
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16, 
        load_in_4bit=False,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(m)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return m, tok

# ============================================================
# CHART GENERATOR MODULE
# ============================================================
def generate_visual_charts(all_results, bench_names, tested_models, output_dir="assets", use_wandb=False):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(tested_models)}
    
    x = np.arange(len(bench_names))
    width = 0.8 / max(1, len(tested_models))
    
    # 1. ACCURACY CHART
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for i, model_key in enumerate(tested_models):
        accs = [all_results[model_key][b]["accuracy"] for b in bench_names]
        bars = ax.bar(x + i * width, accs, width, label=model_key, color=model_colors[model_key], edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width * (len(tested_models) - 1) / 2)
    ax.set_xticklabels(bench_names, fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_title("Benchmark Accuracy Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_accuracy.png")
    plt.close()

    # 2. LATENCY CHART
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for i, model_key in enumerate(tested_models):
        lats = [all_results[model_key][b]["avg_latency_s"] for b in bench_names]
        bars = ax.bar(x + i * width, lats, width, label=model_key, color=model_colors[model_key], edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, lats):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(tested_models) - 1) / 2)
    ax.set_xticklabels(bench_names, fontsize=10, fontweight="bold")
    ax.set_ylabel("Avg Latency (seconds / sample)", fontsize=10, fontweight="bold")
    ax.set_title("Inference Speed by Benchmark", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_latency.png")
    plt.close()

    # 3. FORMAT COMPLIANCE CHART
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for i, model_key in enumerate(tested_models):
        fmts = [all_results[model_key][b]["format_compliance"] for b in bench_names]
        bars = ax.bar(x + i * width, fmts, width, label=model_key, color=model_colors[model_key], edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, fmts):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(tested_models) - 1) / 2)
    ax.set_xticklabels(bench_names, fontsize=10, fontweight="bold")
    ax.set_ylabel("Format Compliance (%)", fontsize=10, fontweight="bold")
    ax.set_title("CoT Formatting Adherence", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_format_compliance.png")
    plt.close()

    # 4. RADAR CHART
    if len(bench_names) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(bench_names), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=150)
        for model_key in tested_models:
            values = [all_results[model_key][b]["accuracy"] for b in bench_names]
            values += values[:1]
            ax.plot(angles, values, label=model_key, linewidth=2, color=model_colors[model_key])
            ax.fill(angles, values, color=model_colors[model_key], alpha=0.15)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(bench_names, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_title("Model Capability Profile", fontsize=12, fontweight="bold", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/benchmark_radar.png")
        plt.close()

    # 5. MASTER DASHBOARD (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=180)
    for i, model_key in enumerate(tested_models):
        accs = [all_results[model_key][b]["accuracy"] for b in bench_names]
        axs[0, 0].bar(x + i * width, accs, width, label=model_key, color=model_colors[model_key])
    axs[0, 0].set_xticks(x + width * (len(tested_models) - 1) / 2)
    axs[0, 0].set_xticklabels(bench_names, fontweight="bold")
    axs[0, 0].set_ylabel("Accuracy (%)")
    axs[0, 0].set_title("Accuracy by Benchmark", fontweight="bold")
    axs[0, 0].set_ylim(0, 105)
    axs[0, 0].legend()

    for i, model_key in enumerate(tested_models):
        fmts = [all_results[model_key][b]["format_compliance"] for b in bench_names]
        axs[0, 1].bar(x + i * width, fmts, width, label=model_key, color=model_colors[model_key])
    axs[0, 1].set_xticks(x + width * (len(tested_models) - 1) / 2)
    axs[0, 1].set_xticklabels(bench_names, fontweight="bold")
    axs[0, 1].set_ylabel("Compliance (%)")
    axs[0, 1].set_title("Format Compliance Rate", fontweight="bold")
    axs[0, 1].set_ylim(0, 105)

    for i, model_key in enumerate(tested_models):
        lats = [all_results[model_key][b]["avg_latency_s"] for b in bench_names]
        axs[1, 0].bar(x + i * width, lats, width, label=model_key, color=model_colors[model_key])
    axs[1, 0].set_xticks(x + width * (len(tested_models) - 1) / 2)
    axs[1, 0].set_xticklabels(bench_names, fontweight="bold")
    axs[1, 0].set_ylabel("Seconds / Sample")
    axs[1, 0].set_title("Inference Latency", fontweight="bold")

    for model_key in tested_models:
        avg_acc = np.mean([all_results[model_key][b]["accuracy"] for b in bench_names])
        avg_lat = np.mean([all_results[model_key][b]["avg_latency_s"] for b in bench_names])
        axs[1, 1].scatter(avg_lat, avg_acc, s=180, label=model_key, color=model_colors[model_key], edgecolors="black", zorder=4)
        axs[1, 1].annotate(model_key, (avg_lat, avg_acc), textcoords="offset points", xytext=(8, 5), fontweight="bold")
    axs[1, 1].set_xlabel("Mean Latency (s/sample)")
    axs[1, 1].set_ylabel("Mean Accuracy (%)")
    axs[1, 1].set_title("Overall Efficiency Trade-Off", fontweight="bold")
    axs[1, 1].set_ylim(0, 105)

    plt.suptitle("Financial Reasoning LLM Benchmark Dashboard", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    dashboard_path = f"{output_dir}/benchmark_dashboard.png"
    plt.savefig(dashboard_path)
    plt.close()
    
    print(f"\nAll visual charts & dashboard saved to ./{output_dir}/")

    if use_wandb:
        wandb.log({
            "charts/accuracy": wandb.Image(f"{output_dir}/benchmark_accuracy.png"),
            "charts/latency": wandb.Image(f"{output_dir}/benchmark_latency.png"),
            "charts/format_compliance": wandb.Image(f"{output_dir}/benchmark_format_compliance.png"),
            "charts/dashboard": wandb.Image(dashboard_path),
        })
        if len(bench_names) >= 3:
            wandb.log({"charts/radar": wandb.Image(f"{output_dir}/benchmark_radar.png")})

# ============================================================
# MAIN PIPELINE
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
            res = evaluate(model, tokenizer, samples, bench_name)
            all_results[model_key][bench_name] = res
            
            print(f"     -> {bench_name}: accuracy={res['accuracy']:.1f}%  "
                  f"format={res['format_compliance']:.1f}%  "
                  f"latency={res['avg_latency_s']:.2f}s")

            if USE_WANDB:
                wandb.log({
                    f"{model_key}/{bench_name}/accuracy": res["accuracy"],
                    f"{model_key}/{bench_name}/format_compliance": res["format_compliance"],
                    f"{model_key}/{bench_name}/latency_s": res["avg_latency_s"],
                })

        del model, tokenizer
        torch.cuda.empty_cache()

    # ---------- FINAL REPORT & EXPORTS ----------
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

    with open("benchmark_report.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate multi-panel charts
    generate_visual_charts(
        all_results=all_results,
        bench_names=list(bench_samples.keys()),
        tested_models=tested,
        output_dir="assets",
        use_wandb=USE_WANDB,
    )

    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()
