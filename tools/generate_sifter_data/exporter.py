"""JSONL serialization — standalone functions, not a class.

Owns:           write_jsonl(), to_training_line(), export_training_set().
Depends on:     json, pathlib, os (fsync).
Provides:       write_jsonl(path, samples) → int,
                to_training_line(sample) → str,
                export_training_set(samples, path) → int.
Invariants:     One JSON object per line. fsync after batch write (GAP-03).
                Creates parent directories on first write.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel


def write_jsonl(
    samples: Iterable[BaseModel],
    output_path: Path,
    mode: str = "a",
) -> int:
    """Write Pydantic model instances as JSONL.

    Opens the file, writes one JSON line per sample, and fsyncs.
    Creates parent directories if they do not exist.

    Args:
        samples: Iterable of Pydantic BaseModel instances.
        output_path: Destination file path.
        mode: File open mode — "w" for overwrite, "a" for append.

    Returns:
        Number of samples written.

    Raises:
        IOError: If file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, mode) as f:
        for sample in samples:
            line = sample.model_dump_json()
            f.write(line + "\n")
            count += 1
        f.flush()
        os.fsync(f.fileno())

    return count


def to_training_line(sample: BaseModel) -> str:
    """Convert a sample to {"text": str, "label": int} JSONL line.

    label = 1 if is_manipulation else 0.
    Uses keyword_regression total > 0 as fallback when is_manipulation not present.

    Args:
        sample: Any product sample BaseModel.

    Returns:
        JSON string with {"text": str, "label": int}.
    """
    data = sample.model_dump()

    text = data.get("text", "")
    if "is_manipulation" in data:
        label = 1 if data["is_manipulation"] else 0
    elif "expected_total" in data:
        label = 1 if data["expected_total"] > 0 else 0
    else:
        label = 0

    return json.dumps({"text": text, "label": label})


def export_training_set(
    samples: Iterable[BaseModel],
    output_path: Path,
) -> int:
    """Export samples as training-format JSONL for train_tfidf_classifier.py.

    Converts each sample to {"text": str, "label": int} and writes to file.

    Args:
        samples: Iterable of product samples.
        output_path: Destination file path.

    Returns:
        Number of samples exported.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w") as f:
        for sample in samples:
            line = to_training_line(sample)
            f.write(line + "\n")
            count += 1
        f.flush()
        os.fsync(f.fileno())

    return count
