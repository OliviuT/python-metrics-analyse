import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


def load_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY in your environment or .env file before running this script.")
    return OpenAI(api_key=api_key)


def chat_loop() -> None:
    client = load_client()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    transcript = [
        {"role": "system", "content": "You are a concise, friendly assistant."},
    ]

    while True:
        try:
            user_prompt = input("\nYou (type 'end chat' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat ended.")
            break

        if user_prompt.lower() == "end chat":
            print("Chat ended.")
            break
        if not user_prompt:
            print("Please type something or 'end chat'.")
            continue

        transcript.append({"role": "user", "content": user_prompt})

        print("Assistant:\n")
        assistant_text = []
        usage_summary = None

        # ✅ Chat Completions streaming: iterate over the generator
        stream = client.chat.completions.create(
            model=model_name,
            messages=transcript,
            stream=True,
            # include_usage is supported for streamed chat completions
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            # Each chunk has choices; delta contains incremental text
            if chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    assistant_text.append(text)
                    print(text, end="", flush=True)

            # The very last event can include usage if stream_options enabled
            if getattr(chunk, "usage", None):
                usage_summary = chunk.usage

        print("\n")

        # Persist the assistant message in the transcript for conversation memory
        final_text = "".join(assistant_text)
        if final_text:
            transcript.append({"role": "assistant", "content": final_text})

        # Show usage if returned; some models may omit it
        if usage_summary:
            prompt_tokens = getattr(usage_summary, "prompt_tokens", 0)
            completion_tokens = getattr(usage_summary, "completion_tokens", 0)
            total_tokens = getattr(usage_summary, "total_tokens", prompt_tokens + completion_tokens)
            print(f"Usage → prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")
        else:
            print("Usage summary: not returned by API")


if __name__ == "__main__":
    chat_loop()
