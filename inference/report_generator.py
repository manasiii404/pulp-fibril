"""
report_generator.py — CSV Metric Report Output
================================================
Saves per-fibril quantification metrics to a CSV file.

Output columns:
  fibril_id, total_length_um, mean_segment_um, tortuosity,
  branching_count, mean_intensity, estimated_depth_um,
  fibrillation_index, num_segments
"""

import csv
import json
from pathlib import Path
from typing import List, Dict


REPORT_COLUMNS = [
    "fibril_id",
    "total_length_um",
    "mean_segment_um",
    "tortuosity",
    "branching_count",
    "mean_intensity",
    "estimated_depth_um",
    "fibrillation_index",
    "num_segments",
]


def save_report(metrics_list: List[Dict], output_path: str):
    """
    Save per-fibril metrics to a CSV file.

    Args:
        metrics_list: List of metric dicts (from mask_to_metrics)
        output_path:  Save path for the CSV
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for m in metrics_list:
            writer.writerow({col: m.get(col, 0) for col in REPORT_COLUMNS})

    # Also print a summary table
    print(f"\n  {'ID':>4} | {'Len(μm)':>8} | {'Tort':>6} | {'Branch':>6} | {'FI':>6}")
    print(f"  {'-'*42}")
    for m in metrics_list:
        print(
            f"  {m['fibril_id']:>4} | "
            f"{m['total_length_um']:>8.2f} | "
            f"{m['tortuosity']:>6.3f} | "
            f"{m['branching_count']:>6d} | "
            f"{m['fibrillation_index']:>6.2f}"
        )

    # Image-level summary
    if metrics_list:
        avg_fi = sum(m["fibrillation_index"] for m in metrics_list) / len(metrics_list)
        avg_len = sum(m["total_length_um"] for m in metrics_list) / len(metrics_list)
        print(f"\n  Summary: {len(metrics_list)} fibrils | "
              f"Avg Length={avg_len:.1f}μm | "
              f"Avg FI={avg_fi:.2f}")


def save_summary_json(metrics_list: List[Dict], output_path: str):
    """Save a JSON summary alongside the CSV for programmatic use."""
    summary = {
        "total_fibrils": len(metrics_list),
        "fibrils": metrics_list,
        "image_summary": {
            "avg_length_um": round(
                sum(m["total_length_um"] for m in metrics_list) / max(len(metrics_list), 1), 3
            ),
            "avg_tortuosity": round(
                sum(m["tortuosity"] for m in metrics_list) / max(len(metrics_list), 1), 4
            ),
            "avg_fibrillation_index": round(
                sum(m["fibrillation_index"] for m in metrics_list) / max(len(metrics_list), 1), 3
            ),
            "total_branching_points": sum(m["branching_count"] for m in metrics_list),
        },
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
