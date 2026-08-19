#!/usr/bin/env python3
"""
Turn Chambers' clinical feature table into the Figure 5 risk groups.

    python scripts/47_fig5_clinical.py \
        --features kinematics/data/interim/features_clinical.pkl

The clinical infants supply the low / moderate / high risk lines. We use THEIR
pose estimates for these, not ViTPose -- the clinical videos were never
released, so re-posing them is not an option and never will be. Their pose
pickle ships inside infant_movement_assessment_repo_files.zip at
data/pose_estimates/clinical/py/pose_estimates.pkl; run it through the same
feature pipeline as the YouTube pose, then point this script at the result.

TWO CONVERSIONS:

1. RISK. meta_data_clin.pkl codes risk 0/1/2 for low/moderate/high. Figure 5
   reserves 0 for the YouTube reference group, so these shift to 1/2/3.
   Getting this wrong silently paints the low-risk infants as the reference.

2. AGE. Stored as separate month and day columns; the plot wants weeks.
   4.345 weeks per month, not 4 -- over a 12-month range the difference is
   about a month and a half.

Chronological age is the default because the reference infants' ages are also
chronological. --corrected switches to corrected age (adjusted for prematurity),
which is arguably the better developmental measure but is NOT comparable to
the YouTube ages.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

def _load_transform():
    """Reuse the EXACT renaming applied to the reference group. Re-implementing
    it here is how the two halves of a figure end up on subtly different
    columns. Loaded by path because the module name starts with a digit, which
    a plain import statement cannot express."""
    import importlib.util
    p = Path(__file__).resolve().parent / "44_fig5_features.py"
    spec = importlib.util.spec_from_file_location("fig5_features", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.to_joint_naming


WEEKS_PER_MONTH = 4.345


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=Path("kinematics/data/interim/features_clinical.pkl"),
                    help="clinical features, same 104-column shape as the "
                         "YouTube ones")
    ap.add_argument("--meta", type=Path,
                    default=Path("Chamber-etal-dataset/meta_data_clin.pkl"))
    ap.add_argument("--out", type=Path,
                    default=Path("kinematics/data/interim/fig5_clinical.pkl"))
    ap.add_argument("--corrected", action="store_true",
                    help="use corrected age/risk (adjusted for prematurity) "
                         "instead of chronological. NOT comparable to the "
                         "YouTube reference ages, which are chronological.")
    args = ap.parse_args()

    for p in (args.features, args.meta):
        if not p.is_file():
            raise SystemExit(
                f"missing {p}\n"
                "meta_data_clin.pkl and the clinical pose estimates live inside\n"
                "infant_movement_assessment_repo_files.zip:\n"
                "  data/video_meta_data/meta_data_clin.pkl\n"
                "  data/pose_estimates/clinical/py/pose_estimates.pkl")

    feats = pd.read_pickle(args.features)
    print(f"in : {feats.shape[0]} sessions x {feats.shape[1] - 1} features")

    joint = _load_transform()(feats)
    wrist = [c for c in joint.columns if c.startswith("Wrist_")]
    print(f"     -> {len(joint.columns) - 1} joint-level features "
          f"({len(wrist)} wrist)")

    meta = pd.read_pickle(args.meta)
    suffix = "corr" if args.corrected else "chron"
    months, days = f"Months_{suffix}", f"Days_{suffix}"
    riskcol = f"Risk_low0_mod1_high2_{suffix}"
    for c in (months, days, riskcol, "video"):
        if c not in meta.columns:
            raise SystemExit(f"meta_data_clin.pkl has no column {c!r}; "
                             f"it has {list(meta.columns)}")

    meta = meta.copy()
    meta["age_in_weeks"] = meta[months] * WEEKS_PER_MONTH + meta[days] / 7.0
    # 0/1/2 -> 1/2/3: risk 0 is the YouTube reference group in Figure 5.
    meta["risk"] = pd.to_numeric(meta[riskcol], errors="coerce") + 1

    cols = ["video", "age_in_weeks", "risk", "infant"]
    cols = [c for c in cols if c in meta.columns]
    merged = joint.merge(meta[cols], on="video", how="inner")

    dropped = len(joint) - len(merged)
    print(f"out: {len(merged)} sessions matched to metadata "
          f"({dropped} unmatched)")
    if len(merged):
        for r, name in ((1, "low"), (2, "moderate"), (3, "high")):
            sub = merged[merged["risk"] == r]
            if len(sub):
                print(f"     risk {r} ({name:8s}): {len(sub):2d} sessions, "
                      f"age {sub['age_in_weeks'].min():.0f}"
                      f"-{sub['age_in_weeks'].max():.0f} wk")
        print(f"     using {'CORRECTED' if args.corrected else 'chronological'} age")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_pickle(args.out)
    merged.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"\n{args.out}\n{args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
