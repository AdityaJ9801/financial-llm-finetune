"""
Runs bench_worker.py once per model in its OWN subprocess (avoids the
meta-tensor crash from sharing CUDA/accelerate state across models),
then aggregates all results into one report + chart + W&B log.
"""
import os
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import wandb

MODELS_TO_TEST = {
    "base":       "unsloth/Meta-Llama-3.1-8B-Instruct",
    "my_fincot":  "Aditya757864/llama3.1-8b-fincot",
    "fino1_ref":  "TheFinAI/Fino1-8B",
}
N_PER_BENCHMARK = 100

WANDB_ENTITY  = "aditya_1976-shri-ramdeobaba-college-of-engineering-and-m"
WANDB_PROJECT = "Financial finetuning model"

os.makedirs("bench_results", exist_ok=True)
result_files = {}

for model_key, model_path in MODELS_TO_TEST.items():
    out_path = f"bench_results/{model_key}.json"
    print(f"\n### Running {model_key} in an isolated subprocess ###")
    cmd = [
        sys.executable, "bench_worker.py",
        "--model_key", model_key,
        "--model_path", model_path,
        "--output", out_path,
        "--n_per_benchmark", str(N_PER_BENCHMARK),
    ]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"  !! {model_key} failed (exit {ret.returncode}) — skipping in report.")
        continue
    result_files[model_key] = out_path

# ---------- aggregate ----------
all_results = {}
for model_key, path in result_files.items():
    with open(path) as f:
        all_results[model_key] = json.load(f)

print("\n\n" + "=" * 78)
print("NUMERIC ACCURACY SUMMARY (%)")
print("=" * 78)
tested = list(all_results.keys())
bench_names = ["FinQA", "GSM8K", "FinCoT"]
header = f"{'Benchmark':<12}" + "".join(f"{k:>16}" for k in tested)
print(header)
for b in bench_names:
    row = f"{b:<12}"
    for k in tested:
        row += f"{all_results[k][b]['accuracy']:>15.1f}%"
    print(row)

print("\nSENTIMENT CLASSIFICATION (F1)")
for k in tested:
    s = all_results[k].get("Sentiment")
    print(f"  {k:<14}: {s['f1']:.3f}" if s else f"  {k:<14}: N/A")

print("\nGENERATION QUALITY (ROUGE-L / BLEU / BERTScore-F1)")
for k in tested:
    g = all_results[k].get("GenQuality")
    if g:
        print(f"  {k:<14}: ROUGE-L={g['rougeL']:.3f}  BLEU={g['bleu']:.1f}  "
              f"BERTScore={g.get('bertscore_f1', float('nan')):.3f}")
    else:
        print(f"  {k:<14}: N/A")

print("\nDOMAIN TERMINOLOGY USAGE (%)")
for k in tested:
    d = all_results[k].get("DomainConsistency")
    print(f"  {k:<14}: {d['domain_term_usage_pct']:.1f}%" if d else f"  {k:<14}: N/A")

print("\nHALLUCINATION STRESS TEST (hedge rate % / fabrication rate %)")
for k in tested:
    h = all_results[k].get("Hallucination")
    if h:
        print(f"  {k:<14}: hedge={h['hedge_rate_pct']:.0f}%  fabrication={h['fabrication_rate_pct']:.0f}%")
    else:
        print(f"  {k:<14}: N/A")

with open("bench_results/combined_report.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ---------- chart ----------
os.makedirs("assets", exist_ok=True)
x = range(len(bench_names))
width = 0.8 / max(1, len(tested))
plt.figure(figsize=(10, 6))
for i, k in enumerate(tested):
    vals = [all_results[k][b]["accuracy"] for b in bench_names]
    offs = [xi + i * width for xi in x]
    bars = plt.bar(offs, vals, width, label=k)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, v+1, f"{v:.0f}", ha="center", fontsize=8)
plt.xticks([xi + width*(len(tested)-1)/2 for xi in x], bench_names)
plt.ylabel("Accuracy (%)"); plt.ylim(0, 100); plt.legend(); plt.grid(axis="y", alpha=0.3)
plt.title("Numeric Accuracy by Benchmark")
plt.tight_layout()
plt.savefig("assets/benchmark_comparison.png", dpi=150)
print("\nSaved chart to assets/benchmark_comparison.png")

# ---------- W&B ----------
run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name="full-benchmark-suite")
for k in tested:
    for b in bench_names:
        wandb.log({f"{k}/{b}/accuracy": all_results[k][b]["accuracy"]})
    if all_results[k].get("Sentiment"):
        wandb.log({f"{k}/sentiment_f1": all_results[k]["Sentiment"]["f1"]})
    if all_results[k].get("GenQuality"):
        wandb.log({f"{k}/rougeL": all_results[k]["GenQuality"]["rougeL"]})
wandb.log({"benchmark_chart": wandb.Image("assets/benchmark_comparison.png")})
wandb.finish()
