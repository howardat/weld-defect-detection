"""Compare baseline, Optuna ceiling, Experiment A, and Experiment C."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "json_output"
A_FILE = OUT_DIR / "experiment_a_results.json"
C_FILE = OUT_DIR / "experiment_c_results.json"
COMPARE_FILE = OUT_DIR / "porosity_comparison.json"


def build_comparison(a_results: dict, c_results: dict, baseline_f1: float = 0.40) -> list[dict]:
    return [
        {"method": "Baseline global sweep", "mean_f1": float(baseline_f1)},
        {"method": "Per-image Optuna Ceiling (GT, non-deployable)",
         "mean_f1": float(a_results["mean_ceiling_f1"])},
        {"method": "Experiment A (predictor, LOOCV)", "mean_f1": float(a_results["mean_f1"])},
        {"method": "Experiment C (warm-start + proxy, LOOCV)", "mean_f1": float(c_results["mean_f1"])},
    ]


def format_table(rows: list[dict]) -> str:
    width = max(len(r["method"]) for r in rows)
    lines = [f"{'Method'.ljust(width)}  Mean F1", f"{'-' * width}  -------"]
    for r in rows:
        lines.append(f"{r['method'].ljust(width)}  {r['mean_f1']:.4f}")
    return "\n".join(lines)


def main() -> None:
    with open(A_FILE) as f:
        a = json.load(f)
    with open(C_FILE) as f:
        c = json.load(f)
    rows = build_comparison(a, c)
    print(format_table(rows))
    with open(COMPARE_FILE, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nComparison -> {COMPARE_FILE}")


if __name__ == "__main__":
    main()
