# python-metrics-analyse

A small collection of scripts and a Streamlit UI to analyze API metrics from a CSV file, compute basic stats (counts, error rates, latency averages/p95), and generate a concise human-readable summary using OpenAI Chat Completions. It also includes simple examples for streaming responses, token budgeting, and an interactive chat loop.

## Features
- Streamlit app (`metrics_ui.py`) to upload a CSV and see:
  - Parsed CSV preview
  - Computed stats by endpoint (count, errors, error rate, avg latency, p95 latency)
  - Live-streamed AI analysis using your OpenAI model
- CLI tools for quick summaries from the terminal:
  - `metrics-app-analyse.py`: streams a summary to stdout
  - `app-token-count.py`: adds rough token budgeting and prints final usage
- Examples and utilities:
  - `test-completion.py`: minimal non-streaming chat completion
  - `test-streaming.py`: streaming completion demo with prompt input
  - `looping-completions.py`: simple interactive chat loop
- `.env.example`: configuration template for API key, model, and optional settings

## Requirements
- Python 3.10+ recommended
- Packages: `openai`, `python-dotenv`, `streamlit`, `pandas`

Install dependencies (using a virtual environment is recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U openai python-dotenv streamlit pandas
```

## Setup
1) Copy the example env file and set your OpenAI API key:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY and (optionally) OPENAI_MODEL
```

Important environment variables:
- `OPENAI_API_KEY`: your key (required)
- `OPENAI_MODEL`: default model (e.g., `gpt-4o-mini`)
- Token budgeting (used by some scripts):
  - `MAX_INPUT_TOKENS` (default 2000)
  - `MAX_OUTPUT_TOKENS` (default 3500)
  - `MAX_TOTAL_TOKENS` (default 5500)

## Usage

### Streamlit UI
Run the UI and upload a CSV of metrics:

```bash
streamlit run python-metrics-analyse/metrics_ui.py
```

The app will:
- Show the CSV contents
- Compute per-endpoint stats
- Call OpenAI and stream a human-readable analysis at the top of the page

### CLI: Streamed Summary
```bash
python python-metrics-analyse/metrics-app-analyse.py python-metrics-analyse/metrics.csv
```
Prints computed stats and then streams the model’s summary to stdout.

### CLI: Token Budget + Usage
```bash
python python-metrics-analyse/app-token-count.py python-metrics-analyse/metrics.csv
```
Performs a rough pre-flight token estimate, streams the response, and prints final token usage (when provided by the API).

### Quick Tests and Demos
- Non-streaming sample:
  ```bash
  python python-metrics-analyse/test-completion.py
  ```
- Streaming sample (prompts for input):
  ```bash
  python python-metrics-analyse/test-streaming.py
  ```
- Simple interactive chat loop:
  ```bash
  python python-metrics-analyse/looping-completions.py
  ```

## CSV Format
Scripts expect a CSV with at least these columns:
- `endpoint` (string)
- `status_code` (integer HTTP status)
- `latency_ms` (numeric latency in milliseconds)

Optional columns like `timestamp` and `user_id` are fine and will be ignored by the stats logic. A sample file is included at `python-metrics-analyse/metrics.csv`:

```csv
timestamp,endpoint,status_code,latency_ms,user_id
2025-11-10T10:00:01Z,/login,200,120,u1
2025-11-10T10:00:02Z,/login,401,95,u2
2025-11-10T10:00:03Z,/login,200,110,u3
2025-11-10T10:00:04Z,/items,200,250,u1
2025-11-10T10:00:05Z,/items,500,800,u4
2025-11-10T10:00:06Z,/items,500,900,u5
2025-11-10T10:00:07Z,/items,200,260,u2
2025-11-10T10:00:08Z,/checkout,200,450,u3
2025-11-10T10:00:09Z,/checkout,502,1200,u4
2025-11-10T10:00:10Z,/checkout,502,1500,u5
```

## Notes and Troubleshooting
- Missing API key: scripts will exit or the UI will show an error if `OPENAI_API_KEY` is not set.
- Large prompts: token budgeting env vars help prevent oversize requests; reduce stats size or lower `MAX_OUTPUT_TOKENS` if needed.
- Streamlit tips: if you don’t see streamed text, check your network/console for errors and ensure the key/model are set correctly.

## Why this exists
This project provides a compact, reproducible example of:
- Turning raw API request logs into actionable per-endpoint stats
- Driving a model to produce a concise summary with recommendations
- Using streaming in both CLI and UI to improve responsiveness

Feel free to adapt the CSV schema and prompts to your own metrics and analysis needs.
