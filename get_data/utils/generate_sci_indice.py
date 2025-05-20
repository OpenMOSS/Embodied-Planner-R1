#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import re
import argparse
from pathlib import Path


def clean_task(raw_task: str) -> str:
    """
    Remove the leading "task-XX-" prefix and strip parentheses, keeping their contents.
    """
    # 1. Drop the first two hyphen-separated segments ("task" and the number/label)
    parts = raw_task.split('-', 2)
    if len(parts) < 3:
        remainder = raw_task[len("task-"):] if raw_task.startswith("task-") else raw_task
    else:
        remainder = parts[2]

    # 2. Remove parentheses but keep their contents
    cleaned = re.sub(r'[()]', '', remainder)
    return cleaned


def transform(input_path: Path, output_path: Path) -> None:
    # Load the original JSON: list of [raw_task, var]
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # Build the new list of dicts
    output_data = []
    for entry in data:
        if not (isinstance(entry, list) and len(entry) == 2):
            # Skip malformed entries
            continue
        raw_task, var = entry
        cleaned = clean_task(raw_task)
        output_data.append({"task": cleaned, "var": var})

    # Ensure parent dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write out the transformed JSON
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a list of [raw_task, var] to cleaned JSON dicts."
    )
    parser.add_argument(
        "-i", "--input", metavar="FILE", type=Path,
        help="Path to source JSON"
    )
    parser.add_argument(
        "-o", "--output", metavar="FILE", type=Path,
        help="Path to destination JSON"
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    input_json = args.input
    output_json = args.output

    transform(input_json, output_json)
    print(f"Converted {input_json} -> {output_json}")