import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"  # sm_100 = B200

import io
import re
import json
import zipfile
import urllib.request

import torch
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- 0. Weights & Biases ---
WANDB_ENTITY  = "aditya_1976-shri-ramdeobaba-college-of-engineering-and-m"
WANDB_PROJECT = "Financial finetuning model"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# --- Blend configuration (tune these) ---
N_GSM8K   = 4000     # general math samples to add
N_FINQA   = None     # None = use all FinQA train rows
BLEND_SEED = 42

run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name="llama3.1-8b-fincot-math-sft",
    config={
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "datasets": ["TheFinAI/FinCoT", "FinQA (github)", "openai/gsm8k"],
        "n_gsm8k": N_GSM8K, "n_finqa": N_FINQA,
        "lora_r": 16, "lora_alpha": 16,
        "learning_rate": 2e-4, "epochs": 1,
        "max_seq_length": 4096, "gpu": "B200",
    },
)

print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
assert torch.cuda.is_bf16_supported(), "bf16 should be supported on B200"

# 1. Model
max_seq_length = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# 2. LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407, use_rslora=False, loftq_config=None,
)

# ============================================================
# 3. LOAD + NORMALIZE ALL SOURCES INTO A COMMON SCHEMA
#    Every source becomes: Question / Reasoning_process / Final_response
# ============================================================

# --- 3a. FinCoT (already in the right schema — keep as is) ---
fincot = load_dataset("TheFinAI/FinCoT", split="SFT")
fincot = fincot.select_columns(["Question", "Reasoning_process", "Final_response"])
print(f"FinCoT rows: {len(fincot)}")

# --- 3b. FinQA (financial numeric reasoning) ---
# The HF repo uses a loading script, which newer `datasets` refuses to run.
# So we fetch the raw JSON straight from the FinQA GitHub repo instead.
FINQA_ZIP_URL = "https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip"

def load_finqa_json(split_filename):
    """Download the FinQA repo zip once and read a split's JSON file."""
    print(f"Downloading FinQA archive for {split_filename} ...")
    with urllib.request.urlopen(FINQA_ZIP_URL) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        inner = f"FinQA-main/dataset/{split_filename}"
        with z.open(inner) as f:
            return json.load(io.TextIOWrapper(f, encoding="utf-8"))

finqa_train = load_finqa_json("train.json")
print(f"FinQA raw examples: {len(finqa_train)}")

def convert_finqa_example(example):
    qa = example.get("qa", {})
    question = qa.get("question", "")

    # answer: 'answer' can be empty in some rows; fall back to last step result
    answer = qa.get("answer", "")
    if not answer:
        steps = qa.get("steps", [])
        if steps:
            answer = str(steps[-1].get("res", ""))

    # reasoning: prefer the gold supporting facts, then the program
    gold_inds = qa.get("gold_inds", {})
    if isinstance(gold_inds, dict):
        reasoning_facts = " ".join(str(v) for v in gold_inds.values())
    else:
        reasoning_facts = " ".join(str(v) for v in gold_inds)
    program = str(qa.get("program", ""))
    reasoning = (reasoning_facts + ("\nProgram: " + program if program else "")).strip()

    # context: pre_text + table + post_text
    pre_text  = " ".join(example.get("pre_text", []) or [])
    post_text = " ".join(example.get("post_text", []) or [])
    table = example.get("table", [])
    table_str = "\n".join(" | ".join(str(c) for c in row) for row in table) if table else ""

    context = "\n".join(p for p in [pre_text, table_str, post_text] if p).strip()

    full_q = (
        "Please answer the given financial question based on the context.\n\n"
        + (f"Context: {context}\n\n" if context else "")
        + f"Question: {question}"
    )
    return {
        "Question": full_q,
        "Reasoning_process": reasoning,
        "Final_response": str(answer).strip(),
    }

finqa_rows = [convert_finqa_example(ex) for ex in finqa_train]
finqa = Dataset.from_list(finqa_rows)
finqa = finqa.filter(lambda x: x["Final_response"] and x["Question"])
if N_FINQA:
    finqa = finqa.shuffle(seed=BLEND_SEED).select(range(min(N_FINQA, len(finqa))))
print(f"FinQA rows after convert/filter: {len(finqa)}")

# --- 3c. GSM8K (general math) ---
# GSM8K schema: 'question', 'answer' where answer = reasoning + "#### <number>"
gsm8k_raw = load_dataset("openai/gsm8k", "main", split="train")

def convert_gsm8k(ex):
    sol = ex["answer"]
    if "####" in sol:
        reasoning, final = sol.split("####", 1)
        reasoning, final = reasoning.strip(), final.strip()
    else:
        reasoning, final = sol.strip(), sol.strip()
    # strip GSM8K's <<...>> calculator annotations from the reasoning text
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
    return {
        "Question": ex["question"].strip(),
        "Reasoning_process": reasoning,
        "Final_response": final,
    }

gsm8k = gsm8k_raw.map(convert_gsm8k, remove_columns=gsm8k_raw.column_names)
if N_GSM8K:
    gsm8k = gsm8k.shuffle(seed=BLEND_SEED).select(range(min(N_GSM8K, len(gsm8k))))
print(f"GSM8K rows: {len(gsm8k)}")

# --- 3d. Blend (FinCoT dominant, math as minority) ---
dataset = concatenate_datasets([fincot, finqa, gsm8k]).shuffle(seed=BLEND_SEED)
print("=" * 70)
print(f"BLENDED DATASET SIZE: {len(dataset)}")
print(f"  FinCoT : {len(fincot)}")
print(f"  FinQA  : {len(finqa)}")
print(f"  GSM8K  : {len(gsm8k)}")
print("=" * 70)

# ============================================================
# 4. FORMATTING — unchanged template, now fed the blended data
# ============================================================
EOS_TOKEN = tokenizer.eos_token
REASONING_HEADER = "Reasoning:"
ANSWER_MARKER    = "Final Answer:"

SYSTEM_PROMPT = (
    "You are a financial reasoning assistant. Work through the problem "
    "step by step under \"Reasoning:\", then give the polished final answer "
    "under \"Final Answer:\"."
)

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{reasoning_header}
{reasoning}

{answer_marker}
{response}<|eot_id|>"""

def formatting_func(examples):
    texts = []
    for q, r, resp in zip(examples["Question"],
                          examples["Reasoning_process"],
                          examples["Final_response"]):
        q = (q or "").strip()
        r = (r or "").strip() or "(no reasoning provided)"
        resp = (resp or "").strip() or "(no answer provided)"
        text = prompt_template.format(
            system=SYSTEM_PROMPT,
            question=q,
            reasoning_header=REASONING_HEADER,
            reasoning=r,
            answer_marker=ANSWER_MARKER,
            response=resp,
        ) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

print("SAMPLE FORMATTED EXAMPLE:")
print(dataset[0]["text"][:2000])
print("=" * 70)

# 5. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True, fp16=False, tf32=True,
        logging_steps=1,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",
        run_name="llama3.1-8b-fincot-math-sft",
    ),
)

trainer_stats = trainer.train()

# 6. Save + log
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
artifact = wandb.Artifact("fincot-math-lora-adapter", type="model")
artifact.add_dir("lora_model")
run.log_artifact(artifact)
wandb.finish()

# 7. Inference smoke test — verify both markers appear
FastLanguageModel.for_inference(model)
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Please answer the given financial question based on the context.\n\n"
                                "Context: estimated amortization expense — 2017: $10,509 (in thousands); "
                                "total: $58,370 (in thousands).\n\n"
                                "Question: What portion of the estimated amortization expense will be "
                                "recognized in 2017?"},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True)
print(tokenizer.batch_decode(outputs))
