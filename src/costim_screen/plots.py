# src/costim_screen/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def volcano_plot(
    df: pd.DataFrame,
    *,
    x_col: str = "logFC",
    y_col: str = "neglog10_q",
    label_col: str = "motif",
    q_col: str = "qvalue",
    q_thresh: float = 0.10,
    lfc_thresh: float = 1.0,
    title: Optional[str] = None,
    top_n_labels: int = 10,
    outpath: Optional[str | Path] = None,
    dpi: int = 200,
):
    """
    Basic volcano plot: log2FC vs -log10(qvalue).

    By default:
      - labels top_n_labels most significant points (smallest qvalue)
      - draws threshold lines at q_thresh and +/- lfc_thresh
      - saves if outpath is provided

    Returns (fig, ax).
    """
    if df.empty:
        raise ValueError("volcano_plot received an empty DataFrame.")

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=12, alpha=0.7)

    # Threshold lines
    y_line = -np.log10(max(q_thresh, 1e-300))
    ax.axhline(y_line, linestyle="--", linewidth=1)
    ax.axvline(+lfc_thresh, linestyle="--", linewidth=1)
    ax.axvline(-lfc_thresh, linestyle="--", linewidth=1)

    ax.set_xlabel(f"{x_col} (typically log2 fold change)")
    ax.set_ylabel(f"{y_col} (typically -log10 q)")
    if title:
        ax.set_title(title)

    # Label top N by qvalue (and finite)
    if top_n_labels and q_col in df.columns:
        sub = df[np.isfinite(df[q_col])].sort_values(q_col, ascending=True).head(int(top_n_labels))
        for _, r in sub.iterrows():
            ax.text(float(r[x_col]), float(r[y_col]), str(r[label_col]), fontsize=8)

    ax.margins(0.05)
    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

    return fig, ax
