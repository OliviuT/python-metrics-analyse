import os
import sys
import csv
import math
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI


# ---------- OpenAI client ----------
def load_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY in your environment or .env file before running this script.")
    return OpenAI(api_key=api_key)


# ---------- CSV helpers (no pandas) ----------
def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _read_csv_head(path: str, max_rows: int = 5) -> Tuple[List[str], List[List[str]], int, bool]:
    """
    Returns: (header, first_rows, total_rows_estimate, truncated)
    Note: we count rows in one pass for speed; total_rows_estimate is exact here.
    """
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], [], 0, False

        rows: List[List[str]] = []
        total = 0
        for r in reader:
            total += 1
            if len(rows) < max_rows:
                rows.append(r)
        truncated = total > max_rows
        return header, rows, total, truncated


def _summarize_columns(path: str, header: List[str], max_values_for_cats: int = 10, max_scan_rows: int = 5000):
    """
    For each column:
      - Detect numeric vs categorical
      - For numeric: count, mean, std, min, max
      - For categorical: #unique (up to max_values_for_cats), show some top values (by appearance order)
    Scans up to max_scan_rows for speed.
    """
    n_cols = len(header)
    if n_cols == 0:
        return []

    # Stats containers
    num_count = [0] * n_cols
    num_sum = [0.0] * n_cols
    num_sumsq = [0.0] * n_cols
    num_min = [math.inf] * n_cols
    num_max = [-math.inf] * n_cols
    cat_values: List[Dict[str, int]] = [dict() for _ in range(n_cols)]
    is_numeric_possible = [True] * n_cols

    scanned = 0
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        # skip header
        try:
            next(reader)
        except StopIteration:
            return []

        for row in reader:
            scanned += 1
            if scanned > max_scan_rows:
                break
            # pad or trim to header length
            if len(row) < n_cols:
                row = row + [""] * (n_cols - len(row))
            elif len(row) > n_cols:
                row = row[:n_cols]

            for i, cell in enumerate(row):
                cell = cell.strip()
                if cell == "":
                    continue
                if is_numeric_possible[i] and _is_float(cell):
                    v = float(cell)
                    num_count[i] += 1
                    num_sum[i] += v
                    num_sumsq[i] += v * v
                    if v < num_min[i]:
                        num_min[i] = v
                    if v > num_max[i]:
                        num_max[i] = v
                else:
                    is_numeric_possible[i] = False
                    # treat as categorical
                    if len(cat_values[i]) < 5000:  # guard
                        cat_values[i][cell] = cat_values[i].get(cell, 0) + 1

    summary = []
    for i, col in enumerate(header):
        if is_numeric_possible[i] and num_count[i] > 0:
            mean = num_sum[i] / num_count[i]
            # population std for simplicity
            variance = (num_sumsq[i] / num_count[i]) - (mean * mean)
            std = math.sqrt(variance) if variance > 0 else 0.0
            summary.append(
                f"- {col} (numeric): count={num_count[i]}, mean={mean:.4f}, std={std:.4f}, min={num_min[i]:.4f}, max={num_max[i]:.4f}"
            )
        else:
            if cat_values[i]:
                # show up to max_values_for_cats most common (simple sort by count desc)
                top = sorted(cat_values[i].items(), key=lambda kv: kv[1], reverse=True)[:max_values_for_cats]
                total_unique = len(cat_values[i])
                tops = ", ".join([f"{k} ({v})" for k, v in top])
                summary.append(f"- {col} (categorical): unique≈{total_unique}; top: {tops}")
            else:
                summary.append(f"- {col}: (mostly empty/mixed)")
    return summary


def build_csv_attachment_text(path: str) -> str:
    """
    Produces a compact, model-friendly text block describing the CSV.
    Includes: filename, header, row count, quick column stats, and a small sample.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    header, sample_rows, total_rows, truncated_head = _read_csv_head(path, max_rows=5)
    col_stats = _summarize_columns(path, header, max_values_for_cats=8, max_scan_rows=5000)

    lines = []
    lines.append(f"FILE: {os.path.basename(path)}")
    try:
        size_bytes = os.path.getsize(path)
        lines.append(f"SIZE: {size_bytes} bytes")
    except Exception:
        pass

    lines.append(f"ROWS (excluding header): {total_rows}")
    if header:
        lines.append(f"COLUMNS ({len(header)}): {', '.join(header)}")
    else:
        lines.append("COLUMNS: (none)")

    if col_stats:
        lines.append("\nCOLUMN STATS:")
        lines.extend(col_stats)

    # Add small sample
    if header:
        lines.append("\nSAMPLE (first 5 rows):")
        fence = "```"
        lines.append(f"{fence}csv")
        lines.append(",".join(header))
        for r in sample_rows:
            # pad/truncate row length to header
            r2 = (r + [""] * (len(header) - len(r)))[: len(header)]
            # escape commas minimally by quoting if needed
            safe = []
            for cell in r2:
                if any(ch in cell for ch in [",", '"', "\n"]):
                    safe.append('"' + cell.replace('"', '""') + '"')
                else:
                    safe.append(cell)
            lines.append(",".join(safe))
        lines.append(fence)
        if truncated_head and total_rows > 5:
            lines.append(f"(… {total_rows - 5} more rows not shown)")
    else:
        lines.append("\n(no sample available; file appears empty)")

    return "\n".join(lines)


# ---------- Chat loop with attachments ----------
def chat_loop() -> None:
    client = load_client()
