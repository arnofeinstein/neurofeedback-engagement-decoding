"""
Stability metrics and plots for decoder coefficients (β) over time.

Designed for Arno's DMS decoding project (Elastic Net + z-score).
Place this file as `neuron_analysis/stability.py` and import in notebooks:

    from neuron_analysis.stability import (
        compute_stability_metrics,
        plot_rank_vs_sign_scatter,
        plot_topn_stability_bar,
        plot_sign_heatmap,
        analyze_stability,
    )

Inputs follow the convention used in your pipeline:
- betas: np.ndarray of shape (n_windows, n_neurons)
- centers: np.ndarray of shape (n_windows,) — window centers in ms
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, kendalltau


def _robust_minmax01(x: np.ndarray, lo_pct: float = 5, hi_pct: float = 95) -> np.ndarray:
    """Percentile-based min–max scaling to [0,1], robust to outliers."""
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def compute_stability_metrics(
    betas: np.ndarray,
    centers: np.ndarray,
    top_ks: Tuple[int, ...] = (5, 10, 20),
    *,
    kendall_reference: str = "global",  # "global" (mean |β| rank) or "first" (first window)
) -> pd.DataFrame:
    """
    Compute rank- and sign-based stability metrics per neuron.

    Parameters
    ----------
    betas : (T, N)
        Decoder coefficients over time (can be negative or positive).
    centers : (T,)
        Window centers (ms), only used for plotting / alignment downstream.
    top_ks : tuple of int
        K values for Top-K persistence metrics.
    kendall_reference : {"global", "first"}
        Reference ranking for Kendall's tau:
          - "global": rank of mean |β| across time (default; more stable)
          - "first": rank in the first time window (sensitive to early epoch)

    Returns
    -------
    df : DataFrame (N x metrics)
        Columns: [neuron, mean_abs_beta, cv_absbeta, rank_std, topK_persist..., 
                  sign_stability, sign_majority, sign_flips, kendall_tau, stability_score]
    """
    B = np.asarray(betas)  # (T, N)
    T, N = B.shape
    absB = np.abs(B)

    # Ranks per window: 1 = largest |β| (use average ties)
    ranks = np.zeros_like(absB)
    for t in range(T):
        ranks[t] = rankdata(-absB[t], method="average")

    # Reference ranking for Kendall τ
    if kendall_reference == "global":
        ref_order = rankdata(-absB.mean(axis=0), method="average")
    elif kendall_reference == "first":
        ref_order = ranks[0]
    else:
        raise ValueError("kendall_reference must be 'global' or 'first'")

    # Top-K masks per window
    top_masks: Dict[int, np.ndarray] = {K: (ranks <= K).astype(float) for K in top_ks}

    rows = []
    signs = np.sign(B)  # -1, 0, +1
    for i in range(N):
        r_i = ranks[:, i]
        a_i = absB[:, i]
        s_i = signs[:, i]

        # CV of |β|
        cv_absbeta = a_i.std() / (a_i.mean() + 1e-12)

        # Rank stability
        rank_std = r_i.std()
        # Kendall τ of the neuron's rank series vs reference ranks across neurons is ill-posed
        # Instead, compute τ between the neuron-specific rank trajectory and a constant ref value
        # is not meaningful. Use τ between ranks of all neurons at each time vs the reference,
        # then assign the neuron's mean τ across time.
    
    # Compute τ(time) between the full rank vector and the reference, then attribute to neurons
    taus_per_time = []
    for t in range(T):
        tau, _ = kendalltau(ranks[t], ref_order)
        taus_per_time.append(tau if np.isfinite(tau) else 0.0)
    mean_tau_over_time = float(np.nanmean(taus_per_time))

    rows = []
    for i in range(N):
        r_i = ranks[:, i]
        a_i = absB[:, i]
        s_i = signs[:, i]

        cv_absbeta = a_i.std() / (a_i.mean() + 1e-12)
        rank_std = r_i.std()

        # Top-K persistence per K
        persist = {f"top{K}_persist": top_masks[K][:, i].mean() for K in top_ks}

        # Sign stability (% windows in majority sign)
        pos_frac = (s_i > 0).mean()
        neg_frac = (s_i < 0).mean()
        sign_stability = max(pos_frac, neg_frac)
        majority_sign = 1 if pos_frac >= neg_frac else -1

        # Sign flips: fill zeros with previous sign to avoid artificial flips
        s_fill = s_i.copy()
        for t in range(1, T):
            if s_fill[t] == 0:
                s_fill[t] = s_fill[t-1]
        sign_flips = int(np.sum(s_fill[1:] != s_fill[:-1]))

        rows.append({
            "neuron": i,
            "mean_abs_beta": a_i.mean(),
            "cv_absbeta": float(cv_absbeta),
            "rank_std": float(rank_std),
            **persist,
            "sign_stability": float(sign_stability),
            "sign_majority": int(majority_sign),
            "sign_flips": sign_flips,
            # Same τ for all neurons (global pattern stability over time)
            "kendall_tau": mean_tau_over_time,
        })

    df = pd.DataFrame(rows)

    # Composite score favouring persistent, coherent, and low-variance contributors
    s_top10 = _robust_minmax01(df.get("top10_persist", pd.Series(np.zeros(len(df)))).values)
    s_sign  = _robust_minmax01(df["sign_stability"].values)
    s_rstd  = 1 - _robust_minmax01(df["rank_std"].values)
    s_cv    = 1 - _robust_minmax01(df["cv_absbeta"].values)

    df["stability_score"] = 0.35*s_top10 + 0.35*s_sign + 0.15*s_rstd + 0.15*s_cv

    return df.sort_values("stability_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rank_vs_sign_scatter(df: pd.DataFrame, *, size_by: str = "mean_abs_beta", top_annotate: int = 10):
    """Scatter: Top-10 persistence vs Sign-stability (size = mean |β| by default)."""
    x = df.get("top10_persist")
    y = df.get("sign_stability")
    if x is None or y is None:
        raise ValueError("DataFrame must contain 'top10_persist' and 'sign_stability'.")

    sizes = 200 * (df[size_by].values / (df[size_by].max() + 1e-9) + 0.1)
    plt.figure(figsize=(7.5, 6))
    plt.scatter(x, y, s=sizes, alpha=0.6, edgecolor="k", linewidth=0.3)
    plt.xlabel("Top-10 persistence (fraction of windows)")
    plt.ylabel("Sign stability (fraction in majority sign)")
    plt.title("Rank- vs Sign-stability (size = mean |β|)")
    # Annotate top neurons by stability_score
    top = df.head(top_annotate)
    for _, row in top.iterrows():
        plt.annotate(int(row["neuron"]), (row["top10_persist"], row["sign_stability"]),
                     textcoords="offset points", xytext=(3,3), fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()


def plot_topn_stability_bar(df: pd.DataFrame, *, topn: int = 15):
    """Horizontal bar plot of the top-N neurons by composite stability score."""
    topN = df.head(topn).sort_values("stability_score", ascending=True)
    plt.figure(figsize=(7, 6))
    plt.barh([f"n{int(i)}" for i in topN["neuron"]], topN["stability_score"])
    plt.xlabel("Composite stability score")
    plt.title(f"Top-{topn} stable neurons (rank + sign + CV)")
    plt.tight_layout()


def plot_sign_heatmap(betas: np.ndarray, centers: np.ndarray, df: pd.DataFrame, *, n_show: int = 12):
    """Heatmap of sign(β) over time for the top stable neurons."""
    sel = df.head(n_show)["neuron"].astype(int).tolist()
    signs = np.sign(np.asarray(betas))  # (T, N)
    signs_sel = signs[:, sel].T         # (n_show, T)

    # Map {-1, 0, +1} to {0, 0.5, 1} for visualization
    mapping = { -1: 0.0, 0: 0.5, 1: 1.0 }
    signs_plot = np.vectorize(mapping.get)(signs_sel)

    plt.figure(figsize=(9, 4.8))
    im = plt.imshow(signs_plot, aspect="auto", origin="lower",
                    extent=[centers[0], centers[-1], 0, len(sel)])
    cbar = plt.colorbar(im, ticks=[0.0, 0.5, 1.0])
    cbar.ax.set_yticklabels(["-1", "0", "+1"])  # type: ignore[attr-defined]
    plt.yticks(np.arange(len(sel)) + 0.5, [f"n{i}" for i in sel])
    plt.xlabel("Time (ms)")
    plt.ylabel("Neurons (top stability)")
    plt.title("Sign stability over time (−1 / 0 / +1)")
    plt.tight_layout()



def analyze_stability(
    betas: np.ndarray,
    centers: np.ndarray,
    *,
    save_csv: Optional[str] = "stability_metrics.csv",
    top_ks: Tuple[int, ...] = (5, 10, 20),
    kendall_reference: str = "global",
    plots: bool = True,
    topn_bar: int = 15,
    n_show_heatmap: int = 12,
):
    """
    Compute stability metrics and optionally render the three diagnostic plots.
    Returns the DataFrame of metrics.
    """
    df = compute_stability_metrics(
        betas=betas,
        centers=centers,
        top_ks=top_ks,
        kendall_reference=kendall_reference,
    )

    if save_csv:
        pd.DataFrame(df).to_csv(save_csv, index=False)

    if plots:
        plot_rank_vs_sign_scatter(df)
        plot_topn_stability_bar(df, topn=topn_bar)
        plot_sign_heatmap(betas, centers, df, n_show=n_show_heatmap)

    return df
