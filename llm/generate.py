from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .inference import DEFAULT_MODEL, LocalIncidentLLM


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an incident response from a RAG prompt using a small local LLM."
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the RAG prompt from a UTF-8 file. If omitted, read stdin.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    prompt = (
        args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file
        else sys.stdin.read()
    )
    if not prompt.strip():
        parser.error("Provide a non-empty prompt through --prompt-file or stdin.")

    response = LocalIncidentLLM(
        args.model,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    ).generate(prompt)
    print(response)


if __name__ == "__main__":
    main()
