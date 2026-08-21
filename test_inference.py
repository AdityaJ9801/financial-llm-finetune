import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

# 1. Load the fine-tuned adapter on top of base Gemma-2-9B
max_seq_length = 4096
dtype = torch.bfloat16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="gemma2_9b_fincot_adapters",  # Path where your adapters were saved
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable native fast inference (2x speedup)
FastLanguageModel.for_inference(model)

# 2. Define a sample financial query (matching FinCoT format)
sample_prompt = (
    "Please answer the given financial question based on the context.\n\n"
    "Context: In FY2023, Company Alpha reported total revenue of $850 million, "
    "cost of goods sold (COGS) of $510 million, and operating expenses of $170 million. "
    "In FY2022, total revenue was $700 million with an operating margin of 18%.\n\n"
    "Question: What is Company Alpha's operating margin for FY2023, and by how many "
    "basis points did it expand or contract compared to FY2022?"
)

# 3. Format using the Gemma chat template
messages = [
    {"role": "user", "content": sample_prompt},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

# 4. Generate with live token streaming
streamer = TextStreamer(tokenizer, skip_prompt=True)

print("\n" + "=" * 50)
print("FINANCIAL COT MODEL RESPONSE:")
print("=" * 50 + "\n")

_ = model.generate(
    input_ids=inputs,
    streamer=streamer,
    max_new_tokens=1024,
    temperature=0.1,  # Low temperature for precise numerical calculations
    top_p=0.9,
    use_cache=True,
)
