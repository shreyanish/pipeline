"""
generate_analysis_plots.py
--------------------------
Generate publication-ready analysis figures from pipeline_results/rppg_results.csv.

Plots produced:
  1. snr_heatmap.png        — ROI × Algorithm SNR heatmap
  2. top5_bot5_snr.png      — Top-5 vs Bottom-5 ROI bar chart (per algorithm)
  3. algo_snr_boxplot.png   — Algorithm SNR distribution across all ROIs
  4. skin_vs_face_bar.png   — Skin-ROI vs Face-ROI metrics comparison
  5. roi_ranking.png        — ROIs ranked by mean SNR with error bars

Usage:
    python spo2/generate_analysis_plots.py \
        --results pipeline_results/rppg_results.csv \
        --output  spo2/results/analysis_plots
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 180,
})

SKIN_REGIONS = [
    'upper_medial_forehead', 'right_upper_lateral_forehead',
    'left_upper_lateral_forehead', 'lower_medial_forehead',
    'right_lower_lateral_forehead', 'left_lower_lateral_forehead',
    'glabella', 'upper_nasal_dorsum', 'right_mid_nasal_sidewall',
    'left_mid_nasal_sidewall', 'right_lower_nasal_sidewall',
    'left_lower_nasal_sidewall', 'lower_nasal_dorsum',
    'left_upper_lip', 'right_upper_lip', 'philtrum',
    'lower_nasal_sidewall', 'right_nasolabial_fold', 'left_nasolabial_fold',
    'chin', 'right_marionette_fold', 'left_marionette_fold',
    'right_malar', 'left_malar', 'right_lower_cheek', 'left_lower_cheek',
]
FACE_REGIONS = ['right_eye', 'left_eye', 'nasal_tip',
                'right_temporal_lobe', 'left_temporal_lobe']

ALGO_ORDER = ['GREEN', 'POS', 'CHROM', 'ICA', 'SSR',
              'PCA', 'PBV', 'LGI', 'SAMC', '2SR', 'OMIT']


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_roi(name: str) -> str:
    return name.replace('_', ' ').title()


def _save(fig, path: str) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {Path(path).name}")


# ── Plot 1: SNR heatmap  ROI × Algorithm ─────────────────────────────────────

def plot_snr_heatmap(df: pd.DataFrame, out: str) -> None:
    pivot = (df.groupby(["roi", "algorithm"])["SNR"]
               .mean()
               .unstack("algorithm")
               .reindex(columns=[a for a in ALGO_ORDER if a in df["algorithm"].unique()]))

    # Sort ROIs by overall mean SNR descending
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    roi_labels = [_fmt_roi(r) for r in pivot.index]
    algo_labels = list(pivot.columns)

    fig, ax = plt.subplots(figsize=(len(algo_labels) * 1.1 + 1.5, len(roi_labels) * 0.38 + 1.5))

    vmin, vmax = pivot.values.min(), pivot.values.max()
    vcen = (vmin + vmax) / 2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcen, vmax=vmax)

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", norm=norm)

    ax.set_xticks(range(len(algo_labels)))
    ax.set_xticklabels(algo_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(roi_labels)))
    ax.set_yticklabels(roi_labels, fontsize=8)

    # Annotate cells
    for i in range(len(roi_labels)):
        for j in range(len(algo_labels)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6.5, color="black" if abs(val - vcen) < (vmax - vmin) * 0.3 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean SNR (dB)", fontsize=9)

    ax.set_title("Mean SNR by Facial Region and Algorithm", fontweight="bold", pad=12)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Facial Region")
    ax.grid(False)

    _save(fig, out)


# ── Plot 2: Top-5 vs Bottom-5 ROI bar chart ──────────────────────────────────

def plot_top5_bot5(df: pd.DataFrame, out: str) -> None:
    roi_snr = df.groupby("roi")["SNR"].mean().sort_values(ascending=False)
    top5 = roi_snr.head(5)
    bot5 = roi_snr.tail(5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    colors_top = plt.cm.Greens(np.linspace(0.45, 0.85, 5))
    colors_bot = plt.cm.Reds(np.linspace(0.45, 0.85, 5))

    # Per-algo breakdown for Top-5 and Bottom-5
    for ax, rois, palette, title in [
        (axes[0], top5.index, colors_top, "Top-5 Regions — Mean SNR per Algorithm"),
        (axes[1], bot5.index, colors_bot, "Bottom-5 Regions — Mean SNR per Algorithm"),
    ]:
        sub = df[df["roi"].isin(rois)]
        pivot = sub.groupby(["roi", "algorithm"])["SNR"].mean().unstack("algorithm")
        pivot = pivot.reindex(index=rois,
                              columns=[a for a in ALGO_ORDER if a in pivot.columns])

        x = np.arange(len(pivot.index))
        n_algos = len(pivot.columns)
        width = 0.7 / n_algos

        for j, algo in enumerate(pivot.columns):
            vals = pivot[algo].values
            ax.bar(x + j * width - 0.35 + width / 2, vals,
                   width=width * 0.9, label=algo, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([_fmt_roi(r) for r in rois], rotation=25,
                           ha="right", fontsize=8.5)
        ax.set_ylabel("Mean SNR (dB)")
        ax.set_title(title, fontweight="bold")
        if ax == axes[0]:
            ax.legend(ncol=3, fontsize=7, loc="lower right")

    fig.suptitle("Top-5 vs Bottom-5 Facial Regions by Signal Quality (SNR)",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out)


# ── Plot 3: Algorithm SNR distribution boxplot ────────────────────────────────

def plot_algo_boxplot(df: pd.DataFrame, out: str) -> None:
    algos = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]
    data = [df.loc[df["algorithm"] == a, "SNR"].dropna().values for a in algos]

    medians = [np.median(d) for d in data]
    order = np.argsort(medians)[::-1]
    algos_sorted = [algos[i] for i in order]
    data_sorted  = [data[i]  for i in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    palette = plt.cm.tab20(np.linspace(0, 0.9, len(algos_sorted)))

    bp = ax.boxplot(data_sorted, patch_artist=True, notch=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                    widths=0.55)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(algos_sorted) + 1))
    ax.set_xticklabels(algos_sorted, fontsize=10)
    ax.set_ylabel("SNR (dB)", fontsize=11)
    ax.set_xlabel("Algorithm", fontsize=11)
    ax.set_title("rPPG Algorithm SNR Distribution Across All Facial Regions\n"
                 f"(N = {df['video'].nunique()} videos × 31 ROIs)",
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    _save(fig, out)


# ── Plot 4: Skin vs Face comparison ──────────────────────────────────────────

def plot_skin_vs_face(df: pd.DataFrame, out: str) -> None:
    df = df.copy()
    df["group"] = df["roi"].apply(
        lambda r: "Face" if r in FACE_REGIONS else "Skin")

    metrics = ["SNR", "NSQI", "Variance"]
    algos = [a for a in ALGO_ORDER if a in df["algorithm"].unique()]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for ax, metric in zip(axes, metrics):
        grp = df.groupby(["algorithm", "group"])[metric].mean().unstack("group")
        grp = grp.reindex(index=[a for a in algos if a in grp.index])

        x = np.arange(len(grp))
        w = 0.35
        skin_vals = grp.get("Skin", pd.Series(dtype=float)).values
        face_vals = grp.get("Face", pd.Series(dtype=float)).values

        b1 = ax.bar(x - w / 2, skin_vals, width=w, label="Skin ROIs",
                    color="#4CAF50", alpha=0.8)
        b2 = ax.bar(x + w / 2, face_vals, width=w, label="Face ROIs",
                    color="#F44336", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(grp.index, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(metric)
        ax.set_title(f"Mean {metric}: Skin vs Face ROIs", fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Skin ROI vs Face ROI — Signal Quality by Algorithm",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out)


# ── Plot 5: ROI ranking with CI ───────────────────────────────────────────────

def plot_roi_ranking(df: pd.DataFrame, out: str) -> None:
    grp = df.groupby("roi")["SNR"].agg(["mean", "std", "count"]).reset_index()
    grp["se"] = grp["std"] / np.sqrt(grp["count"])
    grp = grp.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 10))

    colors = ["#F44336" if r in FACE_REGIONS else "#4CAF50" for r in grp["roi"]]
    y = np.arange(len(grp))
    ax.barh(y, grp["mean"], xerr=grp["se"], color=colors, alpha=0.8,
            error_kw=dict(elinewidth=0.8, capsize=3, ecolor="gray"), height=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels([_fmt_roi(r) for r in grp["roi"]], fontsize=8.5)
    ax.set_xlabel("Mean SNR (dB) ± SE", fontsize=10)
    ax.set_title("Facial Regions Ranked by Mean SNR\n(error bars = standard error)",
                 fontweight="bold")

    skin_patch = mpatches.Patch(color="#4CAF50", alpha=0.8, label="Skin ROI")
    face_patch = mpatches.Patch(color="#F44336", alpha=0.8, label="Face ROI")
    ax.legend(handles=[skin_patch, face_patch], loc="lower right", fontsize=9)

    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    _save(fig, out)


# ── Plot 6: NSQI per algorithm per region (grouped bar, top 10 vs bottom 5) ──

def plot_nsqi_comparison(df: pd.DataFrame, out: str) -> None:
    """Signal quality index comparison for skin optimal vs face noisy regions."""
    top_rois = (df.groupby("roi")["SNR"].mean()
                  .sort_values(ascending=False)
                  .head(5).index.tolist())
    bot_rois = (df.groupby("roi")["SNR"].mean()
                  .sort_values(ascending=True)
                  .head(5).index.tolist())

    selected = top_rois + bot_rois
    sub = df[df["roi"].isin(selected)]
    pivot = sub.groupby(["roi", "algorithm"])["NSQI"].mean().unstack("algorithm")
    pivot = pivot.reindex(index=selected,
                          columns=[a for a in ALGO_ORDER if a in pivot.columns])

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(pivot.index))
    n = len(pivot.columns)
    width = 0.75 / n

    cmap = plt.cm.tab20
    for j, algo in enumerate(pivot.columns):
        offset = j * width - 0.375 + width / 2
        ax.bar(x + offset, pivot[algo].values, width=width * 0.9,
               label=algo, color=cmap(j / n), zorder=3)

    ax.axvline(len(top_rois) - 0.5, color="gray", linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.text(len(top_rois) / 2 - 0.5, ax.get_ylim()[0] * 0.95,
            "← Top-5 regions", ha="center", fontsize=8.5, color="#2e7d32")
    ax.text(len(top_rois) + len(bot_rois) / 2 - 0.5, ax.get_ylim()[0] * 0.95,
            "Bottom-5 regions →", ha="center", fontsize=8.5, color="#c62828")

    ax.set_xticks(x)
    ax.set_xticklabels([_fmt_roi(r) for r in pivot.index],
                       rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("Mean NSQI")
    ax.set_title("Normalised Signal Quality Index — Top-5 vs Bottom-5 Regions",
                 fontweight="bold")
    ax.legend(ncol=6, fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    _save(fig, out)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis plots from rppg_results.csv")
    parser.add_argument(
        "--results", default="./pipeline_results/rppg_results.csv",
        help="Path to rppg_results.csv (default: ./pipeline_results/rppg_results.csv)")
    parser.add_argument(
        "--output", default="./spo2/results/analysis_plots",
        help="Output directory for plots (default: ./spo2/results/analysis_plots)")
    args = parser.parse_args()

    print(f"Loading: {args.results}")
    df = pd.read_csv(args.results)
    print(f"  {len(df):,} rows | {df['video'].nunique()} videos | "
          f"{df['roi'].nunique()} ROIs | {df['algorithm'].nunique()} algorithms")

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving plots to: {args.output}\n")

    plot_snr_heatmap(df,     os.path.join(args.output, "snr_heatmap.png"))
    plot_top5_bot5(df,       os.path.join(args.output, "top5_bot5_snr.png"))
    plot_algo_boxplot(df,    os.path.join(args.output, "algo_snr_boxplot.png"))
    plot_skin_vs_face(df,    os.path.join(args.output, "skin_vs_face_bar.png"))
    plot_roi_ranking(df,     os.path.join(args.output, "roi_ranking.png"))
    plot_nsqi_comparison(df, os.path.join(args.output, "nsqi_top5_bot5.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
