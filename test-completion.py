import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("Missing OPENAI_API_KEY. Add it to your environment or .env file before running.")

model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=api_key)

try:
    with client.responses.stream(
        model=model,
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain the theory of relativity in simple terms."},
        ],
        temperature=0.2,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.completed":
                print("\n--- response complete ---")
            elif event.type == "response.error":
                raise RuntimeError(event.error)

        final_response = stream.get_final_response()
        # Keep final response available for assertions/logging during tests.
        print(f"\nFull response ID: {final_response.id}")
except Exception as exc:  # pragma: no cover
    sys.exit(f"Streaming request failed: {exc}")
