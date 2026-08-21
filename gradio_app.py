from threading import Thread
import gradio as gr
import torch
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel

# ==========================================
# 1. LOAD MODEL & TOKENIZER
# ==========================================
print("Loading fine-tuned financial model...")
max_seq_length = 4096
dtype = torch.bfloat16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="gemma2_9b_fincot_adapters",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable Unsloth 2x fast inference
FastLanguageModel.for_inference(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# 2. STREAMING INFERENCE FUNCTION
# ==========================================
def generate_response(prompt: str, temperature: float, max_tokens: int):
    if not prompt.strip():
        yield "Please enter a financial problem or context to analyze."
        return

    messages = [{"role": "user", "content": prompt}]

    # Format with Gemma chat template
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    # Setup threaded streamer for real-time output in UI
    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=0.9,
        do_sample=temperature > 0.0,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    partial_text = ""
    for new_token in streamer:
        partial_text += new_token
        yield partial_text


# ==========================================
# 3. GRADIO UI DEFINITION
# ==========================================
default_examples = [
    [
        (
            "Please answer the given financial question based on the context.\n\n"
            "Context: amortization expense , which is included in selling , general and administrative expenses , "
            "was $ 13.0 million , $ 13.9 million and $ 8.5 million for the years ended december 31 , 2016 , 2015 and 2014 , respectively . "
            "The estimated amortization expense is: 2017: $12.5M, 2018: $11.0M, 2019: $9.2M, 2020: $8.0M.\n\n"
            "Question: What is the cumulative amortization expense estimated for the two-year period from 2017 to 2018?"
        ),
        0.1,
        1024,
    ],
    [
        (
            "Please answer the given financial question based on the context.\n\n"
            "Context: In FY2023, Company Alpha reported total revenue of $850 million, cost of goods sold (COGS) of $510 million, "
            "and operating expenses of $170 million. In FY2022, total revenue was $700 million with an operating margin of 18%.\n\n"
            "Question: What is Company Alpha's operating margin for FY2023, and by how many basis points did it expand or contract compared to FY2022?"
        ),
        0.1,
        1024,
    ],
    [
        (
            "Please answer the given financial question based on the context.\n\n"
            "Context: As of Q4, ABC Corp holds $450 million in Total Debt and $50 million in Cash & Cash Equivalents. "
            "Its trailing twelve months (TTM) Adjusted EBITDA is $100 million.\n\n"
            "Question: Calculate the company's Net Debt and its Net Debt-to-EBITDA leverage ratio."
        ),
        0.1,
        1024,
    ],
]

custom_css = """
#output_box { font-family: monospace; font-size: 14px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 📊 Gemma-2-9B Financial Chain-of-Thought (FinCoT)")
    gr.Markdown(
        "Interact with the fine-tuned financial reasoning model. Select a sample below or paste your own financial problem."
    )

    with gr.Row():
        with gr.Column(scale=5):
            prompt_input = gr.Textbox(
                label="Financial Question / Context",
                lines=8,
                placeholder="Enter financial text and question here...",
            )
            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Temperature (Lower = more precise math)",
                )
                max_tokens_slider = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Max New Tokens",
                )
            with gr.Row():
                submit_btn = gr.Button("Analyze & Solve", variant="primary")
                clear_btn = gr.ClearButton()

        with gr.Column(scale=5):
            output_display = gr.Textbox(
                label="Step-by-Step Chain of Thought & Solution",
                lines=14,
                elem_id="output_box",
                interactive=False,
            )

    clear_btn.add([prompt_input, output_display])

    submit_btn.click(
        fn=generate_response,
        inputs=[prompt_input, temperature_slider, max_tokens_slider],
        outputs=output_display,
    )

    gr.Examples(
        examples=default_examples,
        inputs=[prompt_input, temperature_slider, max_tokens_slider],
        outputs=output_display,
        fn=generate_response,
        cache_examples=False,
    )

# ==========================================
# 4. LAUNCH WITH PUBLIC GRADIO LIVE LINK
# ==========================================
if __name__ == "__main__":
    # share=True provides a public gradio.live URL accessible anywhere
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)
