"""
3D surface plot of parameter sweep results from sweep_results.json.
Run after sweep_test.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3D projection
from scipy.interpolate import RectBivariateSpline

# ─── Editable constants ────────────────────────────────────────────────────────
INPUT_JSON  = Path("./sweep_results.json")
OUTPUT_PNG  = Path("./accuracy_surface.png")
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    if not INPUT_JSON.exists():
        print(f"Input file not found: {INPUT_JSON}")
        print("Run sweep_test.py first to generate it.")
        sys.exit(1)

    with open(INPUT_JSON) as f:
        data = json.load(f)

    results = data["results"]

    otsu_vals   = sorted({r["otsu_multiplier"]         for r in results})
    thresh_vals = sorted({r["discontinuity_threshold"] for r in results})

    # Build lookup for fast grid construction
    acc_lookup = {
        (r["otsu_multiplier"], r["discontinuity_threshold"]): r["accuracy"]
        for r in results
    }

    # Construct the grid — shape (len(thresh_vals), len(otsu_vals))
    otsu_arr   = np.array(otsu_vals)
    thresh_arr = np.array(thresh_vals)
    Z = np.array([
        [acc_lookup.get((ox, ty), float("nan")) for ox in otsu_vals]
        for ty in thresh_vals
    ])

    # Upsample to a denser grid for a smooth surface
    spline   = RectBivariateSpline(thresh_arr, otsu_arr, Z)
    otsu_fine   = np.linspace(otsu_arr.min(),   otsu_arr.max(),   200)
    thresh_fine = np.linspace(thresh_arr.min(), thresh_arr.max(), 200)
    X, Y = np.meshgrid(otsu_fine, thresh_fine)
    Z = spline(thresh_fine, otsu_fine)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Accuracy")

    ax.set_xlabel("otsu_multiplier")
    ax.set_ylabel("discontinuity_threshold")
    ax.set_zlabel("Accuracy")
    ax.set_title("Discontinuity Detection Accuracy\nvs. otsu_multiplier × threshold")

    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {OUTPUT_PNG}")

    top = sorted(results, key=lambda r: r["accuracy"], reverse=True)[:5]
    print("\n─── Top results ─────────────────────────────────────────────────")
    print(f"{'Rank':<6}{'otsu_multiplier':<18}{'threshold':<12}{'accuracy':<10}{'correct/total'}")
    for rank, r in enumerate(top, start=1):
        print(f"{rank:<6}{r['otsu_multiplier']:<18}{r['discontinuity_threshold']:<12}"
              f"{r['accuracy']:<10}{r['correct']}/{r['total']}")

    plt.show()


if __name__ == "__main__":
    main()
