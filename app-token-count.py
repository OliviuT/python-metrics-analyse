import os
import sys
import csv
import math
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI


# ---- Token limits & helpers ----
# Rough heuristic: ~4 characters per token for English-ish text.
# This is NOT exact but good enough to avoid huge prompts.
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "2000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "3500"))
MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", "5500"))  # safety cap


def _estimate_tokens_from_messages(messages: list[dict]) -> int:
    text = " ".join(m.get("content", "") for m in messages)
    # very rough: chars / 4
    return max(1, len(text) // 4)


def load_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY in your environment or .env file before running this script.")
    return OpenAI(api_key=api_key)

def load_rows(path: str) -> list[dict]:
    """Load CSV file and return list of rows as dicts."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def compute_stats(rows: list[dict]) -> dict:
    """Return a dict like:
    {
      "total_requests": int,
      "by_endpoint": {
        "/login": {
          "count": int,
          "errors": int,
          "error_rate": float,
          "avg_latency": float,
          "p95_latency": float,
        },
        ...
      }
    }
    """
    stats = {
        "total_requests": len(rows),
        "by_endpoint": {}
    }

    endpoint_data: Dict[str, List[float]] = {}

    for row in rows:
        endpoint = row.get("endpoint", "unknown")
        latency = float(row.get("latency_ms", 0))
        status_code = int(row.get("status_code", 200))

        if endpoint not in endpoint_data:
            endpoint_data[endpoint] = []

        endpoint_data[endpoint].append((latency, status_code))

    for endpoint, data in endpoint_data.items():
        count = len(data)
        errors = sum(1 for _, status in data if status >= 400)
        latencies = [lat for lat, _ in data]
        avg_latency = sum(latencies) / count if count > 0 else 0
        p95_latency = sorted(latencies)[math.ceil(0.95 * count) - 1] if count > 0 else 0

        stats["by_endpoint"][endpoint] = {
            "count": count,
            "errors": errors,
            "error_rate": errors / count if count > 0 else 0,
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
        }

    return stats

def build_summary_prompt(stats: dict) -> list[dict]:
    """Return chat messages for the model."""
    prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert data analyst. Given the following API metrics statistics, "
                "provide a concise summary highlighting key insights, potential issues, "
                "and recommendations for improvement."
            )
        },
        {
            "role": "user",
            "content": f"Here are the API metrics statistics:\n{stats}"
        }
    ]
    return prompt


def summarize_with_openai(stats: dict, stream: bool = True) -> str:
    """Call Chat Completions and return the summary text.
    If stream=True, print tokens as they arrive.
    Also prints final token usage and enforces a simple input budget.
    """
    client = load_client()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    messages = build_summary_prompt(stats)

    # ---- pre-flight token budget check ----
    est_input_tokens = _estimate_tokens_from_messages(messages)
    est_total_tokens = est_input_tokens + MAX_OUTPUT_TOKENS

    print(
        f"[Token estimate] input≈{est_input_tokens}, "
        f"max_output={MAX_OUTPUT_TOKENS}, "
        f"total≈{est_total_tokens} (limit={MAX_TOTAL_TOKENS})"
    )

    if est_input_tokens > MAX_INPUT_TOKENS:
        raise RuntimeError(
            f"Refusing to call model: estimated input tokens {est_input_tokens} "
            f"exceed MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}. "
            "Consider summarizing or sampling your stats before sending them."
        )

    if est_total_tokens > MAX_TOTAL_TOKENS:
        raise RuntimeError(
            f"Refusing to call model: estimated total tokens {est_total_tokens} "
            f"exceed MAX_TOTAL_TOKENS={MAX_TOTAL_TOKENS}. "
            "Reduce prompt size or MAX_OUTPUT_TOKENS."
        )

    if not stream:
        # Non-streaming path (original behavior)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage:
            print(
                f"\n[Token usage] prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, "
                f"total={usage.total_tokens}"
            )
        return content

    # --- Streaming path ---
    collected_chunks: list[str] = []
    final_usage = None

    stream_resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
        stream_options={"include_usage": True},  # include a final usage event
    )

    for chunk in stream_resp:
        # usage only appears on the final event when include_usage=True
        if getattr(chunk, "usage", None):
            final_usage = chunk.usage

        # some events (e.g. the final usage event) have no choices
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta
        if delta and delta.content:
            text = delta.content
            print(text, end="", flush=True)
            collected_chunks.append(text)

    if final_usage:
        print(
            f"\n\n[Token usage] prompt={final_usage.prompt_tokens}, "
            f"completion={final_usage.completion_tokens}, "
            f"total={final_usage.total_tokens}"
        )

    return "".join(collected_chunks)

def main():
    """Wire everything together, accept a CSV path from argv."""
    if len(sys.argv) != 2:
        print("Usage: python metrics-app-analyse.py <path_to_metrics_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_rows(csv_path)
    stats = compute_stats(rows)

    #print("Computed stats:")
    #print(stats)
    print("\n=== API Metrics Summary (streaming) ===\n")

    # This will print tokens as they arrive
    summary = summarize_with_openai(stats, stream=True)

    #print("\n\n[Full response]\n")
    #print(summary)
    print("\n===========================\n")


if __name__ == "__main__":
    main()