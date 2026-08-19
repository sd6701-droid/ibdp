#!/usr/bin/env python3
"""
Bayesian surprise per infant, and the reference-vs-risk-group comparison that
is the point of it -- Chambers et al.'s scoring, on OUR ViTPose features.

    python scripts/50_surprise.py \
        --reference kinematics/data/interim/fig5_youtube_vitpose_clips.pkl \
        --clinical  kinematics/data/interim/fig5_clinical.pkl \
        --out       outputs/surprise

WHAT THE SCORE IS. Fit a normal to each feature over the REFERENCE infants
alone, within an age bracket, then score every infant by the log-likelihood of
its features under those reference normals, summed over the 38 features of
Chambers' final set. The result answers "how ordinary does this infant's
movement look next to typically-developing infants of the same age?".

SIGN, because the name invites the opposite reading: Chambers' column
`minus_log_pfeature` is the LOG-LIKELIHOOD (the two minus signs cancel), so
HIGHER = more typical and LOWER = more surprising. z is against the reference
group, so a genuine risk effect shows up as NEGATIVE z. If your high-risk
infants come out with positive z, something is inverted -- do not report it as
a finding.

INPUTS are the two fig5 pickles, not the raw feature tables: scripts/44 and 47
already did the per-joint renaming and the left/right averaging that
merge_data_sets.py does at the end, so starting there reuses tested code
instead of re-deriving it. Chambers' own merge_data_sets.py is 2019 pandas
(DataFrame.append, bare groupby().mean()) and does not run on pandas 2.x.

ONE INFANT, ONE POINT. A clinical infant recorded at several sessions is
several data points, but two videos of the same infant at the same age are one
-- so rows are collapsed to unique (infant, age), matching Chambers.

READ THE POSE-ESTIMATOR CAVEAT PRINTED AT THE END. It is not a footnote.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Chambers' final feature set: 38 features, the ones the surprise is summed
# over. Anything outside this list is dropped, so a typo here silently changes
# the score rather than erroring -- hence the count assertion below.
FEATURE_LIST = [
    "Ankle_medianx", "Wrist_medianx", "Ankle_mediany", "Wrist_mediany",
    "Knee_mean_angle", "Elbow_mean_angle",
    "Ankle_IQRx", "Wrist_IQRx", "Ankle_IQRy", "Wrist_IQRy",
    "Knee_stdev_angle", "Elbow_stdev_angle",
    "Ankle_medianvelx", "Wrist_medianvelx", "Ankle_medianvely", "Wrist_medianvely",
    "Knee_median_vel_angle", "Elbow_median_vel_angle",
    "Ankle_IQRvelx", "Wrist_IQRvelx", "Ankle_IQRvely", "Wrist_IQRvely",
    "Knee_IQR_vel_angle", "Elbow_IQR_vel_angle",
    "Ankle_IQRaccx", "Wrist_IQRaccx", "Ankle_IQRaccy", "Wrist_IQRaccy",
    "Knee_IQR_acc_angle", "Elbow_IQR_acc_angle",
    "Ankle_meanent", "Wrist_meanent", "Knee_entropy_angle", "Elbow_entropy_angle",
    "Ankle_lrCorr_x", "Wrist_lrCorr_x", "Knee_lrCorr_angle", "Elbow_lrCorr_angle",
]
assert len(FEATURE_LIST) == 38, len(FEATURE_LIST)

RISK_NAMES = {0: "reference", 1: "low risk", 2: "moderate risk", 3: "high risk"}


def label_infants(ref: pd.DataFrame, clin: pd.DataFrame) -> pd.DataFrame:
    """Both halves -> one frame with `infant`, `category`, `risk`, `age_in_weeks`.

    Infant ids are prefixed per source because the two metadata tables number
    their infants independently from 1 -- unprefixed, clinical infant 6 and
    YouTube infant 6 would collapse into one row. The clinical id also carries
    the age, since a session is its own data point.
    """
    ref = ref.copy()
    clin = clin.copy()

    idcol = "infant_id" if "infant_id" in ref.columns else "infant"
    if idcol not in ref.columns:
        raise SystemExit(
            "the reference table has no infant_id/infant column -- rerun "
            "scripts/44_fig5_features.py, which attaches it from meta_data_yt.pkl")
    ref["infant"] = "yt_" + ref[idcol].astype(float).astype(int).astype(str)
    ref["category"] = 0
    ref["risk"] = 0

    if "infant" not in clin.columns:
        raise SystemExit(
            "the clinical table has no infant column -- rerun "
            "scripts/47_fig5_clinical.py, which carries it from meta_data_clin.pkl")
    clin["category"] = 1
    clin["infant"] = ("clin_" + clin["infant"].astype(float).astype(int).astype(str)
                      + "_" + clin["age_in_weeks"].round().astype(int).astype(str))

    keep = ["infant", "category", "risk", "age_in_weeks"] + FEATURE_LIST
    for name, df in (("reference", ref), ("clinical", clin)):
        missing = [c for c in FEATURE_LIST if c not in df.columns]
        if missing:
            raise SystemExit(
                f"{name} table is missing {len(missing)} of the 38 scored "
                f"features, e.g. {missing[:5]}")
    both = pd.concat([ref[keep], clin[keep]], ignore_index=True)

    # Several videos of one infant at one age are one observation.
    n_before = len(both)
    both = both.groupby("infant", as_index=False).mean(numeric_only=True)
    if len(both) != n_before:
        print(f"collapsed {n_before} rows -> {len(both)} unique infant-sessions")
    return both


def compute_surprise(df: pd.DataFrame, age_threshold: float) -> tuple:
    """Chambers' scoring. Returns (per-infant surprise, per-feature long table)."""
    long = df.melt(id_vars=["infant", "category", "risk", "age_in_weeks"],
                   value_vars=FEATURE_LIST,
                   var_name="feature", value_name="Value")
    long["age_bracket"] = (long["age_in_weeks"] > age_threshold).astype(int)

    ref = long[long["risk"] == 0]
    if ref.empty:
        raise SystemExit("no reference infants (risk 0) -- nothing to fit against")

    # norm.fit is the MLE: the mean and the population sd (ddof=0).
    stats = (ref.groupby(["feature", "age_bracket"])["Value"]
                .agg(mean_ref="mean", sd_ref=lambda s: s.std(ddof=0),
                     n_ref="count").reset_index())
    stats["var_ref"] = stats["sd_ref"] ** 2
    thin = stats[(stats["n_ref"] < 3) | (stats["sd_ref"] <= 0)]
    if len(thin):
        print(f"\nWARNING: {len(thin)} feature x age-bracket cell(s) have <3 "
              "reference infants or zero spread. A normal fitted to those is "
              "meaningless and the affected features are DROPPED:")
        for _, r in thin.head(10).iterrows():
            print(f"  {r['feature']} (bracket {r['age_bracket']}): "
                  f"n={int(r['n_ref'])}, sd={r['sd_ref']:.4g}")
    stats = stats[(stats["n_ref"] >= 3) & (stats["sd_ref"] > 0)]

    long = long.merge(stats, on=["feature", "age_bracket"], how="inner")
    var = long["sd_ref"] ** 2
    # log N(Value; mean_ref, var). Chambers calls this minus_log_pfeature; it
    # is the log-likelihood, so higher = more typical. See the module docstring.
    long["loglik"] = -(0.5 * np.log(2 * np.pi * var)
                       + (long["Value"] - long["mean_ref"]) ** 2 / (2 * var))

    # LEAVE-ONE-OUT. The line above scores reference infants against a normal
    # fitted to THEMSELVES and clinical infants against one fitted without
    # them, so the reference wins on in-sample fit alone. Simulated from a
    # single distribution, that bias alone puts an outside group near z = -0.6
    # at p < 0.05 -- a significant "finding" from nothing. Refitting each
    # reference infant's cell without that infant puts both groups
    # out-of-sample and removes it. Clinical rows are already out-of-sample.
    n = long["n_ref"]
    S = n * long["mean_ref"]                       # cell sum
    Q = n * (long["var_ref"] + long["mean_ref"] ** 2)   # cell sum of squares
    x = long["Value"]
    n_loo = n - 1
    m_loo = (S - x) / n_loo
    v_loo = (Q - x ** 2) / n_loo - m_loo ** 2
    is_ref = long["risk"] == 0
    ok = is_ref & (n_loo >= 3) & (v_loo > 0)
    long["loglik_loo"] = long["loglik"]
    long.loc[ok, "loglik_loo"] = -(
        0.5 * np.log(2 * np.pi * v_loo[ok])
        + (x[ok] - m_loo[ok]) ** 2 / (2 * v_loo[ok]))
    dropped = int((is_ref & ~ok).sum())
    if dropped:
        print(f"\n{dropped} reference row(s) could not be scored "
              "leave-one-out (cell too small); they keep the in-sample value.")

    long = long[long["Value"].notna() & np.isfinite(long["loglik"])
                & np.isfinite(long["loglik_loo"])]

    surprise = (long.groupby(["infant", "category", "risk", "age_in_weeks",
                              "age_bracket"], as_index=False)
                    .agg(loglik=("loglik", "sum"),
                         loglik_loo=("loglik_loo", "sum"),
                         n_features=("loglik", "size")))

    # An infant scored on fewer features has fewer negative terms in the sum
    # and so looks artificially typical. Summing over unequal feature counts is
    # the quiet way to manufacture a group difference, so say it out loud.
    counts = surprise["n_features"].value_counts()
    if len(counts) > 1:
        print(f"\nWARNING: infants are scored on DIFFERENT numbers of features "
              f"{dict(counts)}. The sum is not comparable across them; treat "
              "any group difference as unproven until this is one number.")

    for src, dst in (("loglik", "z"), ("loglik_loo", "z_loo")):
        base = surprise[surprise["risk"] == 0][src]
        surprise[dst] = (surprise[src] - base.mean()) / base.std()
    return surprise, long


def compare(surprise: pd.DataFrame) -> None:
    """Reference vs each risk group, on the score and on a rank test."""
    from scipy import stats as st

    is_ref = surprise["risk"] == 0

    print("\n  Chambers = reference scored in-sample (reproduces the paper).")
    print("  LOO      = reference refit without each infant; the honest test.")
    print(f"\n{'group':16s} {'n':>3s} | {'mean z':>8s} {'p':>8s} "
          f"| {'mean z LOO':>10s} {'p LOO':>8s}")
    print("-" * 62)
    for r in sorted(surprise["risk"].dropna().unique()):
        sub = surprise[surprise["risk"] == r]
        cells = []
        for src, zcol in (("loglik", "z"), ("loglik_loo", "z_loo")):
            if r == 0:
                cells.append((sub[zcol].mean(), None))
                continue
            try:
                p = st.mannwhitneyu(surprise.loc[is_ref, src], sub[src],
                                    alternative="two-sided")[1]
            except ValueError:
                p = float("nan")
            cells.append((sub[zcol].mean(), p))
        (z1, p1), (z2, p2) = cells
        f1 = "      --" if p1 is None else f"{p1:8.4f}"
        f2 = "      --" if p2 is None else f"{p2:8.4f}"
        print(f"{RISK_NAMES.get(int(r), r):16s} {len(sub):3d} | {z1:8.2f} {f1} "
              f"| {z2:10.2f} {f2}")

    clin = surprise[~is_ref]
    if len(clin) and is_ref.sum() > 1:
        print()
        for src, zcol, name in (("loglik", "z", "Chambers"),
                                ("loglik_loo", "z_loo", "LOO     ")):
            p = st.mannwhitneyu(surprise.loc[is_ref, src], clin[src],
                                alternative="two-sided")[1]
            print(f"all clinical vs reference [{name}]: mean z="
                  f"{clin[zcol].mean():6.2f}, p={p:.4f}")
        direction = ("LOWER (more surprising) -- the expected direction"
                     if clin["z_loo"].mean() < 0 else
                     "HIGHER (more typical) than the reference, which is the "
                     "OPPOSITE of the expected direction")
        print(f"  clinical infants score {direction}")
        if clin["z"].mean() < 0 <= clin["z_loo"].mean():
            print("  NOTE: the effect is present in-sample and GONE under LOO. "
                  "It was the fitting bias, not the infants.")


def plot(surprise: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {0: "grey", 1: "green", 2: "blue", 3: "red"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for r in sorted(surprise["risk"].dropna().unique()):
        sub = surprise[surprise["risk"] == r]
        label = f"{RISK_NAMES.get(int(r), r)} (n={len(sub)})"
        # z_loo, not z: the in-sample score puts any outside group below the
        # reference on fitting bias alone, which is a picture of the method
        # rather than of the infants.
        axes[0].scatter(np.full(len(sub), r) + np.linspace(-.18, .18, len(sub)),
                        sub["z_loo"], color=colours.get(int(r), "black"),
                        alpha=.75, s=26, label=label)
        axes[1].scatter(sub["age_in_weeks"], sub["z_loo"],
                        color=colours.get(int(r), "black"), alpha=.75, s=26,
                        label=label)

    axes[0].axhline(0, color="black", lw=.8, ls="--")
    axes[0].set_xticks(sorted(surprise["risk"].dropna().unique().astype(int)))
    axes[0].set_xticklabels([RISK_NAMES.get(int(r), r).replace(" ", "\n")
                             for r in sorted(surprise["risk"].dropna().unique())])
    axes[0].set_ylabel("surprise z, leave-one-out (lower = more atypical)")
    axes[0].set_title("Surprise by risk group")

    axes[1].axhline(0, color="black", lw=.8, ls="--")
    axes[1].set_xlabel("age (weeks)")
    axes[1].set_ylabel("surprise z, leave-one-out")
    axes[1].set_title("Surprise against age")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Bayesian surprise vs the ViTPose reference infants")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\n{out}\n{out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True,
                    help="fig5 pickle from scripts/44 (risk 0)")
    ap.add_argument("--clinical", type=Path, required=True,
                    help="fig5 pickle from scripts/47 (risk 1/2/3)")
    ap.add_argument("--out", type=Path, default=Path("outputs/surprise"))
    ap.add_argument("--age-threshold", type=float, default=10.0,
                    help="age bracket split in weeks (Chambers: 10)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    for p in (args.reference, args.clinical):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    ref = pd.read_pickle(args.reference)
    clin = pd.read_pickle(args.clinical)
    print(f"reference: {len(ref)} rows    clinical: {len(clin)} rows")

    df = label_infants(ref, clin)
    surprise, long = compute_surprise(df, args.age_threshold)

    for r in sorted(surprise["risk"].dropna().unique()):
        n = int((surprise["risk"] == r).sum())
        print(f"  risk {int(r)} ({RISK_NAMES.get(int(r), r)}): {n} infants")

    compare(surprise)

    args.out.mkdir(parents=True, exist_ok=True)
    surprise.to_pickle(args.out / "bayes_surprise.pkl")
    surprise.to_csv(args.out / "bayes_surprise.csv", index=False)
    long.to_pickle(args.out / "final_feature_set.pkl")
    print(f"\n{args.out / 'bayes_surprise.pkl'}"
          f"\n{args.out / 'bayes_surprise.csv'}"
          f"\n{args.out / 'final_feature_set.pkl'}")

    if not args.no_plot:
        # The tables above are the result; the plot is a convenience. Losing
        # it to a missing matplotlib should not look like the run failed,
        # since the pickles are already on disk by this point.
        try:
            plot(surprise, args.out / "surprise.pdf")
        except ImportError as e:
            print(f"\nno plot ({e}) -- the scores above are still written. "
                  "Re-run with --no-plot to silence this.")

    print("""
CAVEAT -- READ BEFORE REPORTING ANY OF THIS
The reference features come from OUR ViTPose run; the clinical features come
from Chambers' own OpenPose estimates, because those videos were never
released. The two halves are therefore scored across a pose-estimator change
as well as a clinical one, and a normal fitted to ViTPose noise does not
describe OpenPose noise. Any clinical-vs-reference gap here is confounded and
cannot be read as a movement finding on its own.

To separate the two effects, rerun with Chambers' OWN YouTube pose estimates
as the reference (data/pose_estimates/youtube/py/pose_estimates.pkl inside
infant_movement_assessment_repo_files.zip, through the same feature pipeline).
If the group separation survives on their reference but vanishes on ours, the
difference is the pose estimator, not the infants.""")


if __name__ == "__main__":
    main()
