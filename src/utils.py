"""Small IO and plotting helpers shared by the training script."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_outdir(path: str | Path) -> Path:
    """Create `path` (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write `df` to `path` as CSV, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def plot_roc(fpr: Sequence[float], tpr: Sequence[float], out: str | Path) -> Path:
    """Plot an ROC curve (fpr vs tpr) with a diagonal reference line and save it."""
    out = Path(out)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
