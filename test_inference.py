import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

# 1. Load the fine-tuned adapter on top of base Gemma-2-9B
max_seq_length = 4096
dtype = torch.bfloat16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="gemma2_9b_fincot_adapters",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable native fast inference
FastLanguageModel.for_inference(model)

# Ensure pad token is cleanly assigned
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Define sample financial query
sample_prompt = (
    "Please answer the given financial question based on the context.\n\n"
    "Context: In FY2023, Company Alpha reported total revenue of $850 million, "
    "cost of goods sold (COGS) of $510 million, and operating expenses of $170 million. "
    "In FY2022, total revenue was $700 million with an operating margin of 18%.\n\n"
    "Question: What is Company Alpha's operating margin for FY2023, and by how many "
    "basis points did it expand or contract compared to FY2022?"
)

# 3. Format using chat template and return dict with attention_mask
messages = [
    {"role": "user", "content": sample_prompt},
]

# return_dict=True returns {'input_ids': ..., 'attention_mask': ...}
model_inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

# 4. Generate with live token streaming (passing **model_inputs unpacks input_ids and attention_mask)
streamer = TextStreamer(tokenizer, skip_prompt=True)

print("\n" + "=" * 50)
print("FINANCIAL COT MODEL RESPONSE:")
print("=" * 50 + "\n")

_ = model.generate(
    **model_inputs,
    streamer=streamer,
    max_new_tokens=1024,
    temperature=0.1,
    top_p=0.9,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
)
