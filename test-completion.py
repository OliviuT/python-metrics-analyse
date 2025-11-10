import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# 1) Get ready: load .env and grab the API key we need to talk to OpenAI.
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("Set OPENAI_API_KEY in your environment or .env file before running this script.")

# 2) Describe the conversation we want the model to have.
messages = [
    {"role": "system", "content": "You are a patient teacher who explains physics in plain English."},
    {"role": "user", "content": "Explain the theory of relativity in simple terms."},
]

# 3) Talk to the Chat Completions API with streaming turned on.
client = OpenAI(api_key=api_key)
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

print("Assistant:\n")
usage_summary = None


def extract_text(delta) -> str:
    """Make the streaming payload human-friendly regardless of SDK shape."""
    if delta is None:
        return ""
    content = getattr(delta, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    pieces = []
    for block in content:
        # Support both dicts and typed objects
        text = getattr(block, "text", None)
        if isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            pieces.append(text)
        elif text is not None:
            value = getattr(text, "value", None)
            if not value and isinstance(text, dict):
                value = text.get("value")
            if value:
                pieces.append(value)
    return "".join(pieces)


try:
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.4,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.choices:
            choice = chunk.choices[0]
            piece = extract_text(choice.delta)
            if piece:
                print(piece, end="", flush=True)

        if chunk.usage:
            usage_summary = chunk.usage

    print("\n\n--- streaming complete ---")

except Exception as exc:  # pragma: no cover
    sys.exit(f"Chat completion failed: {exc}")

# 4) Show the total tokens the API says we used (input vs. output).
if usage_summary:
    print(
        f"Tokens used - input (prompt): {usage_summary.prompt_tokens}, "
        f"output (completion): {usage_summary.completion_tokens}, "
        f"total: {usage_summary.total_tokens}"
    )
else:
    print("Token usage was not returned by the API.")
