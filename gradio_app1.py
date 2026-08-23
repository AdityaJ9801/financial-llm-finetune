import os
# Blackwell: make arch visible to JIT-compiled kernels
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"  # sm_100 = B200

import torch
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- 0. Weights & Biases setup ---
# Run `wandb login` once in your shell, or set WANDB_API_KEY as an env var.
WANDB_ENTITY  = "aditya_1976-shri-ramdeobaba-college-of-engineering-and-m"
WANDB_PROJECT = "Financial finetuning model"

os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log model checkpoints as artifacts

run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name="llama3.1-8b-fincot-sft",
    config={
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "dataset": "TheFinAI/FinCoT",
        "split": "SFT",
        "lora_r": 16,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "epochs": 1,
        "max_seq_length": 4096,
        "gpu": "B200",
    },
)

# Sanity check the GPU
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))  # expect (10, 0)
assert torch.cuda.is_bf16_supported(), "bf16 should be supported on B200"

# 1. Load model
max_seq_length = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# 2. LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 3. Dataset
dataset = load_dataset("TheFinAI/FinCoT", split="SFT")

# 4. Formatting
EOS_TOKEN = tokenizer.eos_token

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a financial reasoning assistant. Think step by step, then give the final answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{reasoning}

{response}<|eot_id|>"""

def formatting_func(examples):
    texts = []
    for q, r, resp in zip(examples["Question"],
                          examples["Reasoning_process"],
                          examples["Final_response"]):
        texts.append(prompt_template.format(
            question=q.strip(), reasoning=r.strip(), response=resp.strip()
        ) + EOS_TOKEN)
    return {"text": texts}

dataset = dataset.map(formatting_func, batched=True)

# 5. Trainer — report_to="wandb" wires the logging in
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
        bf16=True,
        fp16=False,
        tf32=True,
        logging_steps=1,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",            # <-- send metrics to W&B
        run_name="llama3.1-8b-fincot-sft",
    ),
)

trainer_stats = trainer.train()

# 6. Save + log the final adapter as a W&B artifact
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

artifact = wandb.Artifact("fincot-lora-adapter", type="model")
artifact.add_dir("lora_model")
run.log_artifact(artifact)

wandb.finish()
