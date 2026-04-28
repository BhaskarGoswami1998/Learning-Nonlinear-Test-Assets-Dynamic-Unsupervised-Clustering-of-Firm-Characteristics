"""
plot_utils.py
-------------
Plotting style and shared helpers for all replication notebooks.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────

def set_paper_style():
    """Apply paper-quality matplotlib style."""
    plt.rcParams.update({
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          11,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.frameon":     True,
        "legend.framealpha":  0.9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "lines.linewidth":    1.5,
        "figure.constrained_layout.use": True,
    })


# ── Colour palettes ───────────────────────────────────────────────────────────

CLUSTER_CMAP   = "RdYlGn"
HEATMAP_CMAP   = "viridis"
REGIME_COLORS  = {"Top": "#2166AC", "Btm": "#D6604D", "Com": "#4DAC26"}

FACTOR_COLORS  = {
    "CAPM":   "#1f77b4",
    "FF3":    "#aec7e8",
    "FFC":    "#ffbb78",
    "FF5":    "#2ca02c",
    "FFPS":   "#98df8a",
    "DHS":    "#d62728",
    "Q5":     "#ff9896",
    "PCA5":   "#9467bd",
    "RPPCA5": "#c5b0d5",
    "Cluster":"#e6550d",
}


# ── Convenience wrappers ──────────────────────────────────────────────────────

def add_recession_shading(ax, recessions=None):
    """
    Add NBER recession shading to a time-series axis.

    Parameters
    ----------
    ax         : matplotlib Axes
    recessions : list of (start, end) tuples as datetime-like strings.
                 Defaults to major US recessions in the sample period.
    """
    import pandas as pd
    if recessions is None:
        recessions = [
            ("1980-01-01", "1980-07-01"),
            ("1981-07-01", "1982-11-01"),
            ("1990-07-01", "1991-03-01"),
            ("2001-03-01", "2001-11-01"),
            ("2007-12-01", "2009-06-01"),
        ]
    for start, end in recessions:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="grey", alpha=0.15, linewidth=0)


def format_pct_axis(ax, axis="y", decimals=0):
    """Format axis tick labels as percentages."""
    fmt = f"{{:.{decimals}f}}%"
    if axis == "y":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: fmt.format(x * 100))
        )
    else:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: fmt.format(x * 100))
        )


def plot_cluster_gradient_bar(cluster_mean_rets, ax=None, figsize=(12, 3.5)):
    """
    Bar chart of cluster mean returns coloured by magnitude (green=high, red=low).
    Matches Table 1 visual summary.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    labels = [f"L{i+1:02d}" for i in range(len(cluster_mean_rets))]
    vals   = cluster_mean_rets.values
    colors = plt.cm.RdYlGn(np.linspace(0.05, 0.95, len(vals)))

    ax.bar(labels, vals * 100, color=colors, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cluster (ranked by mean return)")
    ax.set_ylabel("Mean excess return (% / month)")
    ax.set_title("Monthly Out-of-Sample Value-Weighted Cluster Returns")

    # Show only every 5th label
    for i, label in enumerate(ax.get_xticklabels()):
        if (i % 5) != 0:
            label.set_visible(False)

    return fig, ax
