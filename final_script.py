# ==========================================
# 0. DEPLOYED TOP-LEVEL IMPORTS (MANDATORY FOR UNSLOTH)
# ==========================================
from unsloth import FastLanguageModel

import os
import torch
import wandb
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# ==========================================
# 0.5 B200 / MIG PREFLIGHT CHECK
# ==========================================
# Your GPU is a B200 running under a MIG partition (per your nvidia-smi output),
# so torch only sees the MIG slice's memory, not the full card. This check makes
# that visible up front instead of failing with an opaque OOM mid-training.
def preflight_check():
    assert torch.cuda.is_available(), "No CUDA device visible to torch."
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {name}")
    print(f"Compute capability: {cap} (Blackwell B200 should report (10, 0))")
    print(f"Visible memory to this process: {total_mem_gb:.1f} GB")
    if cap[0] < 10:
        print(
            "WARNING: compute capability below (10, 0) — your torch/CUDA build may "
            "predate Blackwell support. bitsandbytes 4-bit kernels may fail or fall "
            "back to slow paths."
        )
    if total_mem_gb < 30:
        print(
            "WARNING: visible memory is well under a full B200's 180GB — this looks "
            "like a MIG slice with another workload already resident. Consider "
            "reducing per_device_train_batch_size or max_seq_length if you hit OOM."
        )


preflight_check()

# ==========================================
# 1. WEIGHTS & BIASES INITIALIZATION
# ==========================================
run = wandb.init(
    entity="aditya_1976-shri-ramdeobaba-college-of-engineering-and-m",
    project="Financial finetuning model",
    config={
        "learning_rate": 2e-4,
        "architecture": "Gemma-2-9b-it (LoRA)",
        "dataset": "TheFinAI/FinCoT",
        "max_steps": 60,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
    },
)

# ==========================================
# 2. CONFIGURATION & B200 CUDA 12.8 RUNTIME ADAPTATIONS
# ==========================================
max_seq_length = 4096
dtype = torch.bfloat16  # Native BF16 on Blackwell (sm_100) — no change needed here
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ==========================================
# 3. PEFT / LORA TARGET PARAMETERS
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ==========================================
# 4. DATASET UTILITIES
# ==========================================
dataset = load_dataset("TheFinAI/FinCoT", split="SFT")

def format_fin_cot(examples):
    texts = []
    for question, computation_trace in zip(
        examples["question"], examples["answer"]
    ):
        messages = [
            {
                "role": "user",
                "content": f"Solve the following financial problem with step-by-step reasoning:\n{question}",
            },
            {"role": "assistant", "content": computation_trace},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


dataset = dataset.map(format_fin_cot, batched=True)

# ==========================================
# 5. SFT TRAINER WITH PROGRESS RETENTION
# ==========================================
output_directory = "outputs"

# NOTE: per_device_train_batch_size kept at 2 since your MIG slice showed ~26GB
# free after the existing ollama process. If you hit OOM, drop this to 1 and
# raise gradient_accumulation_steps to 8 to keep the same effective batch size.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=60,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_directory,
        report_to="wandb",
        logging_dir=f"{output_directory}/runs",
        # --- RESILIENCE CHECKPOINTS ---
        save_strategy="steps",
        save_steps=10,  # Snapshot progress every 10 updates
        save_total_limit=2,
        load_best_model_at_end=False,
    ),
)

# Recover from sudden terminal closures if checkpoint objects are found
checkpoint_exists = False
if os.path.exists(output_directory):
    checkpoints = [
        d for d in os.listdir(output_directory) if d.startswith("checkpoint-")
    ]
    if checkpoints:
        checkpoint_exists = True

if checkpoint_exists:
    print(f"Resuming background pipeline execution from: {output_directory}")
    trainer_stats = trainer.train(resume_from_checkpoint=True)
else:
    print("Starting a clean execution sequence...")
    trainer_stats = trainer.train()

run.finish()

# ==========================================
# 6. SAVE COMPLETED ADAPTERS
# ==========================================
model.save_pretrained("gemma2_9b_fincot_adapters")
tokenizer.save_pretrained("gemma2_9b_fincot_adapters")
print("Process completed successfully.")
