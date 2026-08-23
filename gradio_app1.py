from threading import Thread
import time
import gradio as gr
import torch
import wandb
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel


# ============================================================
# CONFIGURATION & W&B INITIALIZATION
# ============================================================

MODEL_NAME = "gemma2_9b_fincot_adapters"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True
WANDB_PROJECT = "financial-llm-gradio"

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# Initialize Weights & Biases
wandb.init(
    project=WANDB_PROJECT,
    name="gemma2-9b-inference-session",
    config={
        "model_name": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "load_in_4bit": LOAD_IN_4BIT,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    },
)

# Table to log all prompt/response interactions
interaction_table = wandb.Table(
    columns=["timestamp", "prompt", "response", "temperature", "max_tokens", "latency_seconds"]
)


# ============================================================
# STARTUP INFORMATION
# ============================================================

print("=" * 70)
print("FINANCIAL LLM")
print("=" * 70)
print(f"Model      : {MODEL_NAME}")
print(f"Device     : {DEVICE}")
print(f"Dtype      : {DTYPE}")
print(f"4-bit      : {LOAD_IN_4BIT}")
print(f"Max Length : {MAX_SEQ_LENGTH}")

if torch.cuda.is_available():
    print(f"GPU        : {torch.cuda.get_device_name(0)}")

print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading fine-tuned financial model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

print("Model loaded successfully.")

FastLanguageModel.for_inference(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer configured.")
print("Model ready.\n")


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_response(prompt, temperature, max_tokens):
    """Generate a streaming response and log inference metrics to W&B."""
    if prompt is None or not prompt.strip():
        yield "Please enter a financial problem or context to analyze."
        return

    prompt = prompt.strip()
    start_time = time.time()

    try:
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        if isinstance(model_inputs, dict):
            model_inputs = {key: value.to(DEVICE) for key, value in model_inputs.items()}
        else:
            model_inputs = {"input_ids": model_inputs.to(DEVICE)}

        streamer = TextIteratorStreamer(
            tokenizer,
            timeout=120.0,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        temperature = float(temperature)
        max_tokens = int(max_tokens)

        generation_kwargs = {
            **model_inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0.0 else None,
            "top_p": 0.9 if temperature > 0.0 else None,
            "do_sample": temperature > 0.0,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        generation_thread = Thread(
            target=model.generate,
            kwargs=generation_kwargs,
            daemon=True,
        )
        generation_thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

        generation_thread.join(timeout=5)

        # ----------------------------------------------------
        # Log generation metrics to Weights & Biases
        # ----------------------------------------------------
        latency = round(time.time() - start_time, 3)
        input_token_count = len(model_inputs["input_ids"][0])
        output_token_count = len(tokenizer.encode(generated_text))

        wandb.log({
            "inference/latency_seconds": latency,
            "inference/input_tokens": input_token_count,
            "inference/output_tokens": output_token_count,
            "inference/total_tokens": input_token_count + output_token_count,
            "inference/tokens_per_second": round(output_token_count / latency, 2) if latency > 0 else 0,
            "inference/temperature": temperature,
            "inference/max_tokens": max_tokens,
        })

        interaction_table.add_data(
            time.strftime("%Y-%m-%d %H:%M:%S"),
            prompt,
            generated_text,
            temperature,
            max_tokens,
            latency,
        )
        wandb.log({"inference/interactions": interaction_table})

    except Exception as error:
        print("\n" + "=" * 70)
        print("GENERATION ERROR")
        print("=" * 70)
        print(error)
        print("=" * 70)

        wandb.log({"inference/errors": 1})
        yield f"Generation failed.\n\nError: {error}"


# ============================================================
# EXAMPLES
# ============================================================

default_examples = [
    [
        (
            "Please answer the given financial question based on the context.\n\n"
            "Context: amortization expense, which is included in selling, "
            "general and administrative expenses, was $13.0 million, "
            "$13.9 million and $8.5 million for the years ended December 31, "
            "2016, 2015 and 2014, respectively.\n\n"
            "Estimated amortization expense:\n"
            "2017: $12.5M\n"
            "2018: $11.0M\n"
            "2019: $9.2M\n"
            "2020: $8.0M\n\n"
            "Question: What is the cumulative amortization expense estimated "
            "for the two-year period from 2017 to 2018?"
        ),
        0.1,
        1024,
    ],
    [
        (
            "Please answer the given financial question based on the context.\n\n"
            "Context: In FY2023, Company Alpha reported total revenue of "
            "$850 million, cost of goods sold (COGS) of $510 million, "
            "and operating expenses of $170 million.\n\n"
            "In FY2022, total revenue was $700 million with an operating "
            "margin of 18%.\n\n"
            "Question: What is Company Alpha's operating margin for FY2023, "
            "and by how many basis points did it expand or contract "
            "compared to FY2022?"
        ),
        0.1,
        1024,
    ],
    [
        (
            "Please answer the given financial question based on the context.\n\n"
            "Context: As of Q4, ABC Corp holds $450 million in Total Debt "
            "and $50 million in Cash & Cash Equivalents. "
            "Its trailing twelve months (TTM) Adjusted EBITDA is $100 million.\n\n"
            "Question: Calculate the company's Net Debt and its "
            "Net Debt-to-EBITDA leverage ratio."
        ),
        0.1,
        1024,
    ],
]


# ============================================================
# CSS
# ============================================================

custom_css = """
#output_box textarea {
    font-family: monospace !important;
    font-size: 14px !important;
}

.gradio-container {
    max-width: 1400px !important;
}

textarea {
    font-family: monospace !important;
}
"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks() as demo:

    gr.Markdown(
        """
# 📊 Gemma-2-9B Financial Model
### Fine-Tuned Financial Question Answering
Enter financial context and a question to generate a solution.
"""
    )

    gr.Markdown("You can select an example below or enter your own question.")

    with gr.Row():
        with gr.Column(scale=5):
            prompt_input = gr.Textbox(
                label="Financial Question / Context",
                lines=12,
                placeholder="Enter your financial context and question here...",
            )

            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Temperature",
                    info="Lower values are more deterministic.",
                )

                max_tokens_slider = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Max New Tokens",
                )

        with gr.Column(scale=5):
            output_display = gr.Textbox(
                label="Financial Solution",
                lines=18,
                elem_id="output_box",
                interactive=False,
                buttons=["copy"],
            )

    with gr.Row():
        submit_btn = gr.Button("Analyze & Solve", variant="primary")
        clear_btn = gr.ClearButton([prompt_input, output_display], value="Clear")

    submit_btn.click(
        fn=generate_response,
        inputs=[
            prompt_input,
            temperature_slider,
            max_tokens_slider,
        ],
        outputs=output_display,
    )

    gr.Examples(
        examples=default_examples,
        inputs=[
            prompt_input,
            temperature_slider,
            max_tokens_slider,
        ],
        outputs=output_display,
        fn=generate_response,
        cache_examples=False,
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting Gradio server...")
    print("=" * 70)

    try:
        demo.queue(
            max_size=20,
            default_concurrency_limit=1,
        ).launch(
            share=True,
            server_name="0.0.0.0",
            theme=gr.themes.Soft(),
            css=custom_css,
        )
    finally:
        wandb.finish()
