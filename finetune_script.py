FINAL_ANSWER_MARKERS = ["Final Answer:"]

SYSTEM_PROMPT = (
    "You are a financial reasoning assistant. Work through the problem "
    "step by step under \"Reasoning:\", then give the polished final answer "
    "under \"Final Answer:\"."
)

def split_reasoning_and_answer(text):
    """Split on 'Final Answer:'; strip a leading 'Reasoning:' header."""
    idx = text.find("Final Answer:")
    if idx == -1:
        reasoning = text
        final_answer = ""
    else:
        reasoning = text[:idx]
        final_answer = text[idx + len("Final Answer:"):].strip()

    # Drop a leading "Reasoning:" header from the reasoning panel
    reasoning = reasoning.strip()
    if reasoning.startswith("Reasoning:"):
        reasoning = reasoning[len("Reasoning:"):].strip()

    return reasoning, final_answer
