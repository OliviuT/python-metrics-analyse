import os
import sys
import csv
import math
import io
from typing import Dict, List
import pandas as pd


import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ---- Token limits & helpers ----
# Rough heuristic: ~4 characters per token for English-ish text.
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
        st.error("OPENAI_API_KEY is not set. Add it to your .env file.")
        st.stop()
    return OpenAI(api_key=api_key)


def load_rows_from_text(csv_text: str) -> list[dict]:
    """Load CSV from an in-memory string and return list of rows as dicts."""
    f = io.StringIO(csv_text)
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

def stats_to_dataframe(stats: dict) -> pd.DataFrame:
    """Convert the stats dict into a pandas DataFrame indexed by endpoint."""
    rows = []
    for endpoint, ep in stats.get("by_endpoint", {}).items():
        rows.append({
            "endpoint": endpoint,
            "count": ep["count"],
            "errors": ep["errors"],
            "error_rate": ep["error_rate"],         # fraction (0–1)
            "avg_latency_ms": ep["avg_latency"],
            "p95_latency_ms": ep["p95_latency"],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("endpoint")
    return df


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


def summarize_with_openai(stats: dict) -> str:
    """Call Chat Completions and return the summary text.
    Here we use non-streaming for simplicity in the web UI.
    """
    client = load_client()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    messages = build_summary_prompt(stats)

    # ---- pre-flight token budget check ----
    est_input_tokens = _estimate_tokens_from_messages(messages)
    est_total_tokens = est_input_tokens + MAX_OUTPUT_TOKENS

    if est_input_tokens > MAX_INPUT_TOKENS:
        raise RuntimeError(
            f"Estimated input tokens {est_input_tokens} exceed MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}. "
            "Consider summarizing or sampling your stats before sending them."
        )

    if est_total_tokens > MAX_TOTAL_TOKENS:
        raise RuntimeError(
            f"Estimated total tokens {est_total_tokens} exceed MAX_TOTAL_TOKENS={MAX_TOTAL_TOKENS}. "
            "Reduce prompt size or MAX_OUTPUT_TOKENS."
        )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    if usage:
        # You can also display this in the UI if you like
        print(
            f"[Token usage] prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}"
        )

    return content


# =======================
# 🌐 Streamlit Frontend
# =======================

st.set_page_config(page_title="Metrics Analyzer", layout="wide")

st.title("📊 Metrics Analyzer with OpenAI")
st.write(
    "Upload a CSV with API metrics, and this app will compute statistics and ask an OpenAI "
    "model to generate a human-readable analysis."
)

uploaded_file = st.file_uploader("Upload your metrics CSV", type=["csv"])

csv_text = ""
stats = None
analysis_text = ""

# Left and right columns for nicer layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ CSV Contents")

    if uploaded_file is not None:
        # Read bytes → decode to string
        csv_text = uploaded_file.getvalue().decode("utf-8")
        st.text_area("File contents", csv_text, height=300)
    else:
        st.info("Upload a CSV file to see its contents here.")

with col2:
    st.subheader("2️⃣ AI Analysis")

    if uploaded_file is not None:
        if st.button("Analyze with OpenAI"):
            try:
                rows = load_rows_from_text(csv_text)
                if not rows:
                    st.error("The CSV seems to be empty or has no data rows.")
                else:
                    stats = compute_stats(rows)
                    st.write("Computed stats:")
                    st.json(stats)

                    rows = load_rows_from_text(csv_text)
                    if not rows:
                        st.error("The CSV seems to be empty or has no data rows.")
                    else:
                        stats = compute_stats(rows)
                        st.write("Computed stats:")
                        st.json(stats)

                        # 🔢 Convert stats to DataFrame for charting
                        df = stats_to_dataframe(stats)

                        if df.empty:
                            st.warning("No endpoints found in stats to chart.")
                        else:
                            st.subheader("3️⃣ Visualizations by Endpoint")

                            st.markdown("**Requests per endpoint**")
                            st.bar_chart(df["count"])

                            st.markdown("**Errors per endpoint**")
                            st.bar_chart(df["errors"])

                            st.markdown("**Error rate per endpoint (%)**")
                            st.bar_chart(df["error_rate"] * 100.0)

                            st.markdown("**Average latency per endpoint (ms)**")
                            st.bar_chart(df["avg_latency_ms"])

                            st.markdown("**95th percentile latency per endpoint (ms)**")
                            st.bar_chart(df["p95_latency_ms"])
                            
                    with st.spinner("Calling OpenAI for analysis..."):
                        analysis_text = summarize_with_openai(stats)

                    st.text_area("AI prompt response / analysis", analysis_text, height=300)
            except Exception as exc:
                st.error(f"Error while processing: {exc}")
    else:
        st.info("Upload a CSV file first, then click 'Analyze with OpenAI'.")
