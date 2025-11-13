import os
import sys
import csv
import math
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI
from streamlit import json

import time
from dataclasses import dataclass, asdict



@dataclass
class LatencyMetrics:
    ttft: float           # Time to first token (seconds)
    tbt: float            # Average time between tokens (seconds)
    total_latency: float  # Total latency from start to last token (seconds)
    token_count: int      # Number of streamed tokens/chunks

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------
# OpenAI client helper
# -----------------------------

def _load_client() -> OpenAI:
    """Create an OpenAI client using environment variables (.env supported)."""
    load_dotenv()
    return OpenAI()  # uses OPENAI_API_KEY from env


def compute_metrics(start: float, timestamps: List[float], last: float) -> LatencyMetrics:
    """
    Compute latency metrics from timing data.

    Args:
        start:      timestamp (seconds) when the request started
        timestamps: list of timestamps (seconds) for each received token/chunk
        last:       timestamp (seconds) when the stream finished

    Returns:
        LatencyMetrics with ttft, tbt, total_latency, token_count
    """
    token_count = len(timestamps)

    if token_count == 0:
        # No tokens received – avoid division by zero, everything is 0.
        return LatencyMetrics(
            ttft=0.0,
            tbt=0.0,
            total_latency=0.0,
            token_count=0,
        )

    # Time to first token
    ttft = timestamps[0] - start

    # Total latency from start to last token
    total_latency = last - start

    # Average time between tokens:
    # span between first and last token divided by (N - 1).
    if token_count == 1:
        tbt = 0.0
    else:
        tbt = (timestamps[-1] - timestamps[0]) / (token_count - 1)

    return LatencyMetrics(
        ttft=ttft,
        tbt=tbt,
        total_latency=total_latency,
        token_count=token_count,
    )

def measure_latency(prompt: str) -> LatencyMetrics:
    """
    Call the OpenAI Chat Completions API with streaming enabled and
    measure latency-related metrics.

    Returns a LatencyMetrics dataclass with:
        ttft: float
        tbt: float
        total_latency: float
        token_count: int
    """
    client = _load_client()
    model = "gpt-4o-mini"  # or whichever model you want to test

    # Record start time right before we send the request
    t_start = time.monotonic()

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    token_timestamps: List[float] = []

    # Iterate over the streaming response
    for chunk in stream:
        # Some events may be empty (no choices), skip them
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if delta and delta.content:
            # We treat each content-bearing chunk as a "token"
            token_timestamps.append(time.monotonic())

    # If we got tokens, last is timestamp of final token; otherwise, use start.
    t_last = token_timestamps[-1] if token_timestamps else t_start

    # Delegate metric math to the pure helper
    return compute_metrics(start=t_start, timestamps=token_timestamps, last=t_last)

def main(argv: list[str]) -> None:
    """Simple CLI to measure latency for a given prompt."""
    if len(argv) < 2:
        print(
            "Usage:\n  python stream_latency.py \"Your prompt here\"\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Join all args after the script name into a single prompt string
    prompt = " ".join(argv[1:])

    metrics = measure_latency(prompt)
    # Pretty-print as JSON so it matches the example in the task
    print(metrics.to_dict())

if __name__ == "__main__":
    main(sys.argv)

