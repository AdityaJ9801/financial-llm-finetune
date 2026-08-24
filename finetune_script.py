import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"  # sm_100 = B200

import torch
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- 0. Weights & Biases ---
WANDB_ENTITY  = "aditya_1976-shri-ramdeobaba-college-of-engineering-and-m"
WANDB_PROJECT = "Financial finetuning model"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name="llama3.1-8b-fincot-sft",
    config={
        "base_model": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "dataset": "TheFinAI/FinCoT",
        "split": "SFT",
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

# 3. Dataset
dataset = load_dataset("TheFinAI/FinCoT", split="SFT")

# 4. Formatting — the two dataset fields ARE the two panels.
# Reasoning_process  -> chain-of-thought (reasoning box)
# Final_response     -> polished answer  (final answer box)
# Fixed markers make the split deterministic at inference time.
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

dataset = dataset.map(formatting_func, batched=True)

print("=" * 70)
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
        run_name="llama3.1-8b-fincot-sft",
    ),
)

trainer_stats = trainer.train()

# 6. Save + log
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
artifact = wandb.Artifact("fincot-lora-adapter", type="model")
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
