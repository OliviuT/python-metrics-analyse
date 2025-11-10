import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_PROMPT = "Explain the theory of relativity in simple terms."
DEFAULT_SYSTEM = "You are a helpful coding assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick Responses API prompt for interview dry runs."
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="Prompt text to send to the model (overrides default).",
    )
    parser.add_argument(
        "-f",
        "--prompt-file",
        type=Path,
        help="Path to a file containing the prompt text.",
    )
    parser.add_argument(
        "-s",
        "--system",
        default=os.getenv("OPENAI_SYSTEM_PROMPT", DEFAULT_SYSTEM),
        help="System instruction for the assistant.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model name to call.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        help="Sampling temperature (0.0-2.0).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "512")),
        help="Maximum output tokens to request.",
    )
    return parser.parse_args()


def load_prompt_text(args: argparse.Namespace) -> str:
    if args.prompt_file:
        try:
            return args.prompt_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            sys.exit(f"Failed to read prompt file: {exc}")
    if args.prompt:
        return args.prompt.strip()
    return DEFAULT_PROMPT


def create_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Missing OPENAI_API_KEY. Add it to your environment or .env file before running.")
    return OpenAI(api_key=api_key)


def run_prompt(client: OpenAI, *, prompt: str, system: str, model: str, temperature: float, max_output_tokens: int) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return response.output_text


def main() -> None:
    args = parse_args()
    prompt = load_prompt_text(args)
    client = create_client()
    try:
        output_text = run_prompt(
            client,
            prompt=prompt,
            system=args.system,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    except Exception as exc:  # pragma: no cover - SDK raises numerous subclasses
        sys.exit(f"OpenAI request failed: {exc}")

    print(output_text)


if __name__ == "__main__":
    main()
