import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

import torch
from unsloth import FastLanguageModel

# Your deployed adapter model
MODEL_ID = "Aditya757864/llama3.1-8b-fincot"

print("Loading model...")
# NOTE: Unsloth automatically detects this is an adapter, 
# pulls the base Llama-3.1 model, and applies your adapter to it natively!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    device_map={"": 0}, 
)
FastLanguageModel.for_inference(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# A simple financial question
question = "Company A has $500,000 in revenue and $350,000 in operating expenses. What is their operating profit?"
system_prompt = (
    "You are a financial reasoning assistant. Work through the problem "
    'step by step under "Reasoning:", then give the polished final answer '
    'under "Final Answer:".'
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question}
]

print("\nTokenizing input...")
inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to("cuda")

print("Generating response (this might take a few seconds)...\n")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs, 
        max_new_tokens=256, 
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

# Extract only the generated text (ignoring the prompt)
generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

print("="*60)
print("RAW MODEL OUTPUT:")
print("="*60)
print(generated_text)
print("="*60)
