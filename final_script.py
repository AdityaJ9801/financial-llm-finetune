# ==========================================
# 0. DEPLOYED TOP-LEVEL IMPORTS (MANDATORY FOR UNSLOTH)
# ==========================================
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import os
import torch
import wandb
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# ==========================================
# 0.5 B200 / MIG PREFLIGHT CHECK
# ==========================================
def preflight_check():
    assert torch.cuda.is_available(), "No CUDA device visible to torch."
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {name} (Compute Capability: {cap})")
    print(f"Visible memory: {total_mem_gb:.1f} GB")

preflight_check()

# ==========================================
# 1. INITIALIZE W&B
# ==========================================
run = wandb.init(
    entity="aditya_1976-shri-ramdeobaba-college-of-engineering-and-m",
    project="Financial finetuning model",
    config={
        "learning_rate": 1e-4,
        "architecture": "Gemma-2-9b-it (LoRA)",
        "dataset": "TheFinAI/FinCoT",
        "max_steps": 250,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
    },
)

# ==========================================
# 2. MODEL & TOKENIZER CONFIGURATION
# ==========================================
max_seq_length = 2048  # 2048 is optimal for FinCoT and speeds up training
dtype = torch.bfloat16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply correct Gemma chat template formatting
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma",
)

# ==========================================
# 3. PEFT / LORA SETUP
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
)

# ==========================================
# 4. DATASET UTILITIES
# ==========================================
dataset = load_dataset("TheFinAI/FinCoT", split="SFT")

def format_fin_cot(examples):
    texts = []
    questions = examples["Question"]
    reasonings = examples["Reasoning_process"]
    finals = examples["Final_response"]

    for q, r, f in zip(questions, reasonings, finals):
        # Format the model's full Chain of Thought
        assistant_content = f"{r}\n\nFinal Answer: {f}"

        messages = [
            {"role": "user", "content": q.strip()},
            {"role": "assistant", "content": assistant_content.strip()},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return {"text": texts}

dataset = dataset.map(
    format_fin_cot,
    batched=True,
    remove_columns=dataset.column_names,
)

# ==========================================
# 5. SFT TRAINER WITH RESPONSE-ONLY MASKING
# ==========================================
output_directory = "outputs"

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
        warmup_steps=20,
        max_steps=250,          # Increased from 60 to 250 for actual convergence
        learning_rate=1e-4,     # Lowered from 2e-4 to prevent mode collapse on 9B
        bf16=True,
        fp16=False,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=output_directory,
        report_to="wandb",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
    ),
)

# Mask loss on user prompts so the model only learns assistant completions
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

# Run training
print("Starting fine-tuning...")
trainer_stats = trainer.train()

run.finish()

# ==========================================
# 6. SAVE COMPLETED ADAPTERS
# ==========================================
model.save_pretrained("gemma2_9b_fincot_adapters")
tokenizer.save_pretrained("gemma2_9b_fincot_adapters")
print("Model fine-tuned and saved successfully!")
