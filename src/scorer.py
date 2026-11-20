"""
Scorer for generation outputs and evaluation JSONL files.
Adheres to the project evaluation schema:
{"run_id": "O2-sft-s0", "prompt_id": "p0123", "cue_family": "B", "cue_template_id": "B03",
 "monitored": false, "output": "...", "marker_present": true}
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from src.marker import has_marker


def score_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Scores a single record and populates/updates `marker_present`."""
    output_text = record.get("output", "")
    scored = dict(record)
    scored["marker_present"] = has_marker(output_text)
    return scored


def score_jsonl_file(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Reads a JSONL file of model generations, scores each, and optionally writes output."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {input_path}")

    scored_records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scored_records.append(score_record(rec))

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for rec in scored_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return scored_records
