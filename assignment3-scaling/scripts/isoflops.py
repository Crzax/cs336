"""IsoFLOPs scaling-law fitting (assignment 3, problem chinchilla_isoflops).

Method (Hoffmann et al. "Chinchilla", simplified per the handout):

  1. Group the runs in data/isoflops_curves.json by compute budget C_i.
  2. For each budget, take the run with the LOWEST final loss as the optimum
     (no quadratic profile fit), giving the points <C_i, N_opt(C_i)>.
     The dataset size follows from the Chinchilla approximation C ~ 6ND:
         D_opt(C_i) = C_i / (6 * N_opt(C_i)).
  3. Fit power laws  N_opt = a_N * C^b_N  and  D_opt = a_D * C^b_D  by linear
     least squares in log10 space (np.polyfit; the handout allows any
     curve-fitting method, so scipy is not required).
  4. Extrapolate to 1e23 / 1e24 FLOPs and plot.

Outputs (written under docs/figures/):
  isoflops_profiles.png     - the 9 IsoFLOPs profiles (loss vs model size),
                              minima marked: justifies the N_opt points.
  isoflops_model_size.png   - N_opt(C) scaling law with extrapolation (part a).
  isoflops_dataset_size.png - D_opt(C) scaling law with extrapolation (part b).

Run:  python3 scripts/isoflops.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "isoflops_curves.json"
FIG_DIR = REPO_ROOT / "docs" / "figures"

TARGET_BUDGETS = [1e23, 1e24]


def fit_power_law(C: np.ndarray, Y: np.ndarray) -> tuple[float, float, float]:
    """Fit Y = a * C**b by least squares in log10 space.

    Returns (a, b, R^2) where R^2 is computed on the log scale.
    """
    logC, logY = np.log10(C), np.log10(Y)
    b, loga = np.polyfit(logC, logY, 1)
    residuals = logY - (loga + b * logC)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((logY - logY.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return 10.0**loga, float(b), r2


def predict(C, a: float, b: float):
    return a * np.asarray(C, dtype=np.float64) ** b


def fmt_params(n: float) -> str:
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def fmt_tokens(d: float) -> str:
    return f"{d / 1e12:.2f}T" if d >= 1e12 else f"{d / 1e9:.1f}B"


def main() -> None:
    runs = json.loads(DATA_PATH.read_text())
    groups: dict[float, list[dict]] = defaultdict(list)
    for run in runs:
        groups[run["compute_budget"]].append(run)

    budgets = sorted(groups)
    n_opt = np.array([min(groups[c], key=lambda r: r["final_loss"])["parameters"] for c in budgets])
    d_opt = np.array([c / (6.0 * n) for c, n in zip(budgets, n_opt)])
    C = np.array(budgets, dtype=np.float64)

    print("=== <C_i, N_opt(C_i), D_opt(C_i)> from the runs ===")
    for c, n, d in zip(C, n_opt, d_opt):
        print(f"  C={c:.2e}  N_opt={n:.3e} ({fmt_params(n)})  D_opt={d:.3e} ({fmt_tokens(d)})")

    a_n, b_n, r2_n = fit_power_law(C, n_opt)
    a_d, b_d, r2_d = fit_power_law(C, d_opt)
    print(f"\nN_opt = {a_n:.3g} * C^{b_n:.4f}   (R^2 = {r2_n:.6f})")
    print(f"D_opt = {a_d:.3g} * C^{b_d:.4f}   (R^2 = {r2_d:.6f})")

    print("\n=== Extrapolations ===")
    preds = {}
    for target in TARGET_BUDGETS:
        n_hat = predict(target, a_n, b_n)
        d_hat = predict(target, a_d, b_d)
        preds[target] = (n_hat, d_hat)
        print(f"  C={target:.0e}: N_opt={n_hat:.3e} ({fmt_params(n_hat)} params), "
              f"D_opt={d_hat:.3e} ({fmt_tokens(d_hat)} tokens)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    c_max_data = C.max()

    # ---------------------------------------------------------------- profiles
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(budgets):
        pts = sorted(groups[c], key=lambda r: r["parameters"])
        xs = [p["parameters"] for p in pts]
        ys = [p["final_loss"] for p in pts]
        color = cmap(i / max(1, len(budgets) - 1))
        ax.plot(xs, ys, "o-", ms=4, lw=1, color=color, label=f"{c:.0e}")
        best = min(pts, key=lambda r: r["final_loss"])
        ax.plot([best["parameters"]], [best["final_loss"]], "*", ms=12, color=color, mec="k", mew=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("model size N (parameters)")
    ax.set_ylabel("final training loss")
    ax.set_title("IsoFLOPs profiles (stars: minimum-loss run per budget)")
    ax.legend(fontsize=7, title="compute budget", ncol=3)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "isoflops_profiles.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------- scaling-law plots
    for name, ydata, (a, b, r2), unit, fmt in [
        ("model_size", n_opt, (a_n, b_n, r2_n), "parameters", fmt_params),
        ("dataset_size", d_opt, (a_d, b_d, r2_d), "tokens", fmt_tokens),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        # Fit line: solid over the data range, dashed over the extrapolation.
        c_grid_data = np.logspace(np.log10(C.min()), np.log10(c_max_data), 50)
        c_grid_extra = np.logspace(np.log10(c_max_data), np.log10(2e24), 50)
        ax.plot(c_grid_data, predict(c_grid_data, a, b), "-", lw=1.2, color="tab:blue",
                label=f"fit: {a:.2g}·C^{b:.3f} (R²={r2:.4f})")
        ax.plot(c_grid_extra, predict(c_grid_extra, a, b), "--", lw=1.2, color="tab:blue")
        ax.axvspan(c_max_data, 2e24, color="tab:blue", alpha=0.06)
        ax.plot(C, ydata, "o", ms=6, color="k", label=f"<C_i, {'N' if 'model' in name else 'D'}_opt(C_i)>")
        for target in TARGET_BUDGETS:
            y_hat = preds[target][0 if "model" in name else 1]
            ax.plot([target], [y_hat], "*", ms=14, color="tab:red", mec="k", mew=0.4)
            ax.annotate(f"C={target:.0e}\n{fmt(y_hat)}", xy=(target, y_hat),
                        xytext=(-8, 10), textcoords="offset points", fontsize=9,
                        ha="right", color="tab:red")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("compute budget C (FLOPs)")
        ax.set_ylabel(f"compute-optimal {'model size' if 'model' in name else 'dataset size'} ({unit})")
        ax.set_title(f"IsoFLOPs scaling law: {'N_opt' if 'model' in name else 'D_opt'}(C)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"isoflops_{name}.png", dpi=150)
        plt.close(fig)

    print(f"\nwrote 3 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
