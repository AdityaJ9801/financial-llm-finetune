import os
# Blackwell: make sure the arch is visible to any JIT-compiled kernels
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"  # sm_100 = B200

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Sanity check the GPU is recognized as Blackwell
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))  # expect (10, 0)
assert torch.cuda.is_bf16_supported(), "bf16 should be supported on B200"

# 1. Load model
max_seq_length = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",  # non-4bit; B200 has plenty of VRAM
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,       # native bf16 on Blackwell
    load_in_4bit=False,         # 180GB HBM3e — no need to quantize an 8B
    # load_in_4bit=True,        # still fine if you want it; needs bitsandbytes>=0.45
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

# 5. Trainer — larger batch to exploit B200 HBM3e
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    packing=True,  # good throughput win on Blackwell
    args=TrainingArguments(
        per_device_train_batch_size=8,      # bump; B200 has 180GB
        gradient_accumulation_steps=2,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,                          # force bf16 on Blackwell
        fp16=False,
        tf32=True,                          # enable TF32 matmuls
        logging_steps=1,
        optim="adamw_torch_fused",          # fused optimizer, fast on Blackwell
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer_stats = trainer.train()

# 6. Save
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# 7. Inference
FastLanguageModel.for_inference(model)
messages = [
    {"role": "system", "content": "You are a financial reasoning assistant. Think step by step, then give the final answer."},
    {"role": "user", "content": "What portion of the estimated amortization expense will be recognized in 2017?"},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True)
print(tokenizer.batch_decode(outputs))
