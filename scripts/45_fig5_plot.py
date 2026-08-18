#!/usr/bin/env python3
"""
Figure 5 of Chambers et al.: wrist kinematic features against infant age,
one regression overlay per risk group.

    python scripts/45_fig5_plot.py \
        --reference kinematics/data/interim/fig5_youtube_vitpose.pkl \
        --clinical  kinematics/data/interim/fig5_clinical.pkl \
        --out       outputs/figures/features_wrists.pdf

A 3x4 grid, one panel per feature in feature_list_wrists, age_in_weeks on x and
the feature value on y. Each panel overlays four regressions: the YouTube
reference infants (risk 0) plus the clinical low / moderate / high risk groups.

RUNS WITHOUT THE CLINICAL HALF. Pass --reference alone and you get the same
grid with the reference group only -- useful for checking the reference data
before the clinical features exist. The legend says which groups are present,
so a one-group plot cannot be mistaken for the full figure.

COLOURS: the published caption and the repo's own notebook disagree (the repo
uses grey/green/blue/red). We follow the repo, since that is what actually
produced the released figure; --palette paper switches to the caption's.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: compute nodes have no display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FEATURES = [
    "Wrist_medianx", "Wrist_mediany", "Wrist_IQRx", "Wrist_IQRy",
    "Wrist_medianvelx", "Wrist_medianvely", "Wrist_IQRvelx", "Wrist_IQRvely",
    "Wrist_IQRaccx", "Wrist_IQRaccy",
    "Wrist_lrCorr_x", "Wrist_meanent",
]

# Axis labels. The notebook pulls these from a feature_label column on the
# merged set; we spell them out so the figure is readable without it.
LABELS = {
    "Wrist_medianx":    "median x position",
    "Wrist_mediany":    "median y position",
    "Wrist_IQRx":       "IQR x position",
    "Wrist_IQRy":       "IQR y position",
    "Wrist_medianvelx": "median x velocity",
    "Wrist_medianvely": "median y velocity",
    "Wrist_IQRvelx":    "IQR x velocity",
    "Wrist_IQRvely":    "IQR y velocity",
    "Wrist_IQRaccx":    "IQR x acceleration",
    "Wrist_IQRaccy":    "IQR y acceleration",
    "Wrist_lrCorr_x":   "left-right correlation (x)",
    "Wrist_meanent":    "mean entropy",
}

GROUPS = [(0, "reference"), (1, "low risk"), (2, "moderate risk"), (3, "high risk")]
PALETTES = {
    "repo":  {0: "grey", 1: "green", 2: "blue", 3: "red"},
    "paper": {0: "grey", 1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"},
}


def load(reference: Path, clinical: Path | None) -> pd.DataFrame:
    ref = pd.read_pickle(reference)
    ref["risk"] = 0
    frames = [ref]
    if clinical is not None:
        clin = pd.read_pickle(clinical)
        if "risk" not in clin.columns:
            raise SystemExit(
                f"{clinical} has no `risk` column -- it must carry 1/2/3 for "
                "low/moderate/high (Chambers codes these 0/1/2, so add one).")
        frames.append(clin)
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "age_in_weeks" not in df.columns:
        raise SystemExit("no `age_in_weeks` column -- run scripts/44 first")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True,
                    help="YouTube reference features (risk 0)")
    ap.add_argument("--clinical", type=Path, default=None,
                    help="clinical features carrying risk 1/2/3; omit to plot "
                         "the reference group alone")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/figures/features_wrists.pdf"))
    ap.add_argument("--palette", choices=sorted(PALETTES), default="repo")
    ap.add_argument("--order", type=int, default=1,
                    help="polynomial order for the regression fit")
    ap.add_argument("--label", action="store_true",
                    help="number every point and write a <out>_labels.csv "
                         "mapping number -> video, so an outlier on the plot "
                         "can be traced back to the video that produced it")
    args = ap.parse_args()

    df = load(args.reference, args.clinical)

    # Stable point numbers, assigned once over the whole frame and sorted by
    # group then video, so the same video carries the same number in every
    # panel AND across re-runs. Numbering per-panel would be useless -- the
    # whole point is tracing one dot through all twelve.
    if args.label:
        idcol = "video" if "video" in df.columns else df.columns[0]
        df = df.sort_values(["risk", idcol]).reset_index(drop=True)
        df["point"] = range(1, len(df) + 1)

    present = sorted(df["risk"].dropna().unique())
    print(f"{len(df)} rows, risk groups present: {present}")
    for r, name in GROUPS:
        n = int((df["risk"] == r).sum())
        if n:
            sub = df[df["risk"] == r]["age_in_weeks"]
            print(f"  risk {r} ({name}): {n} rows, age {sub.min():.0f}-{sub.max():.0f} wk")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"missing feature column(s): {missing}")

    colours = PALETTES[args.palette]
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))

    for ax, feat in zip(axes.ravel(), FEATURES):
        for risk, name in GROUPS:
            sub = df[df["risk"] == risk]
            if sub.empty:
                continue
            sns.regplot(x="age_in_weeks", y=feat, data=sub, ax=ax,
                        order=args.order, color=colours[risk],
                        scatter_kws={"s": 18, "alpha": 0.6},
                        line_kws={"linewidth": 2},
                        label=f"{name} (n={len(sub)})")
            if args.label:
                for _, row in sub.iterrows():
                    if pd.isna(row[feat]):
                        continue
                    ax.annotate(str(int(row["point"])),
                                (row["age_in_weeks"], row[feat]),
                                fontsize=6, alpha=0.9,
                                xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("age (weeks)")
        ax.set_ylabel(LABELS.get(feat, feat))
        ax.set_title(feat, fontsize=10)

    # One legend for the whole figure rather than twelve identical ones.
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Wrist kinematic features by age and risk group", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\n{args.out}\n{args.out.with_suffix('.png')}")

    if args.label:
        idcol = "video" if "video" in df.columns else df.columns[0]
        keep = [c for c in ("point", idcol, "age_in_weeks", "risk", "infant_id")
                if c in df.columns]
        table = args.out.with_name(args.out.stem + "_labels.csv")
        df[keep].to_csv(table, index=False)
        print(table)


if __name__ == "__main__":
    main()
