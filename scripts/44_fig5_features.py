#!/usr/bin/env python3
"""
Reshape our per-side kinematic features into the Chambers et al. naming, and
attach each video's infant age -- the two things standing between
features_youtube_vitpose.pkl and the Figure 5 plotting code.

    python scripts/44_fig5_features.py \
        --features kinematics/data/interim/features_youtube_vitpose.pkl \
        --chambers Chamber-etal-dataset

TWO TRANSFORMS, both mechanical but easy to get subtly wrong:

1. NAMING. Our extractor emits one column per side (`IQRaccx_LWrist`,
   `IQRaccx_RWrist`); the paper's feature_list_wrists wants one per joint
   (`Wrist_IQRaccx`). We average the two sides, which is what "Wrist" means in
   their feature set -- NOT concatenate, and not left-only.
   `lrCorr_x_Wrist` is already a two-limb quantity and is only renamed.

2. AGE. Ages live in meta_data_yt.pkl keyed by `video_number` and an
   anonymised `video` ("video_000000"), NOT by YouTube id. The bridge is
   URL_pose_dataset.csv, which carries video_number -> url -> the 11-char id.
   Only videos in THAT list have an age, so the output is smaller than the
   input and the script says by how much.

risk=0 is set for every row: these are the YouTube reference infants. The
clinical rows (risk 1/2/3) come from Chambers' own pose estimates and are
merged separately -- we cannot pose those videos, they were never released.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def _shim_legacy_pandas():
    """meta_data_yt.pkl was pickled with 2019-era pandas and names
    pandas.core.indexes.numeric.Int64Index, which pandas 2.x deleted. Without
    this, read_pickle dies on an import of a module that no longer exists --
    an error that looks like a corrupt file rather than a version skew.
    Mapping the old names onto the modern Index is enough to unpickle."""
    import sys
    import types
    if "pandas.core.indexes.numeric" in sys.modules:
        return
    try:
        import pandas.core.indexes.numeric  # noqa: F401  (still present: nothing to do)
    except ImportError:
        from pandas import Index
        mod = types.ModuleType("pandas.core.indexes.numeric")
        mod.Int64Index = Index
        mod.Float64Index = Index
        mod.UInt64Index = Index
        mod.NumericIndex = Index
        sys.modules["pandas.core.indexes.numeric"] = mod


_shim_legacy_pandas()

# A side-specific column: <stat>_<L|R><Joint>. The stat itself may contain
# underscores (IQR_acc_angle), so anchor on the side letter before the joint.
SIDED = re.compile(r"^(?P<stat>.+?)_(?P<side>[LR])(?P<joint>Wrist|Ankle|Elbow|Knee|Hip|Shoulder|Ear|Eye)$")
# lrCorr_x_Wrist / lrCorr_angle_Elbow -- inherently two-limb, no side token.
LRCORR = re.compile(r"^lrCorr_(?P<axis>x|angle)_(?P<joint>\w+)$")
# Pose dirs cut by scripts/46 are named <video_id>_<start>-<end>s, so `video`
# carries the WINDOW, not the 11-char id meta_data_yt.pkl is keyed on. Anchor
# on the suffix rather than slicing 11 characters off the front: YouTube ids
# contain _ and - (2-qT--mHx_8, _eyl8uuoFcg), so a fixed-width slice is a trap.
WINDOW = re.compile(r"^(?P<vid>.+)_(?P<start>\d+)-(?P<end>\d+)s$")


def to_joint_naming(df: pd.DataFrame) -> pd.DataFrame:
    """Per-side columns -> per-joint columns, averaging left and right."""
    out = pd.DataFrame(index=df.index)
    if "video" in df.columns:
        out["video"] = df["video"]

    pairs: dict[tuple[str, str], dict[str, str]] = {}
    for col in df.columns:
        m = LRCORR.match(col)
        if m:
            out[f"{m.group('joint')}_lrCorr_{m.group('axis')}"] = df[col]
            continue
        m = SIDED.match(col)
        if m:
            key = (m.group("joint"), m.group("stat"))
            pairs.setdefault(key, {})[m.group("side")] = col

    unpaired = []
    for (joint, stat), sides in sorted(pairs.items()):
        name = f"{joint}_{stat}"
        if "L" in sides and "R" in sides:
            out[name] = df[[sides["L"], sides["R"]]].mean(axis=1)
        else:
            # One side only -- keep it, but say so. A silently half-populated
            # feature would look identical to a properly averaged one.
            only = next(iter(sides.values()))
            out[name] = df[only]
            unpaired.append(f"{name} (only {only})")
    if unpaired:
        print(f"WARNING: {len(unpaired)} feature(s) had one side only:",
              file=sys.stderr)
        for u in unpaired:
            print(f"  {u}", file=sys.stderr)
    return out


def collapse_windows(df: pd.DataFrame) -> pd.DataFrame:
    """<video_id>_<start>-<end>s rows -> one row per <video_id>.

    A no-op on a whole-video pose tree, where `video` is already the id.
    Chambers analysed a window per video and Figure 5 plots ONE point per
    video, so a video with several windows has to become one row; the numeric
    features are averaged unweighted, since build_features does not carry the
    per-window frame counts a weighted mean would need.
    """
    m = df["video"].astype(str).str.extract(WINDOW)
    if m["vid"].isna().all():
        return df                       # whole-video tree: nothing to strip
    if m["vid"].isna().any():
        bad = df.loc[m["vid"].isna(), "video"].tolist()[:5]
        raise SystemExit(
            "some rows are windowed and some are not -- refusing to guess "
            f"which tree this is. Unparsed: {bad}")

    out = df.copy()
    out["video"] = m["vid"]
    counts = out["video"].value_counts()
    multi = counts[counts > 1]
    if multi.empty:
        print(f"     stripped window suffixes ({len(out)} videos, "
              "one window each)")
        return out

    num = list(out.select_dtypes("number").columns)
    collapsed = out.groupby("video", as_index=False)[num].mean()
    print(f"     {len(out)} windows -> {len(collapsed)} videos; averaged "
          f"{len(multi)} multi-window video(s): "
          + ", ".join(f"{v} (x{n})" for v, n in multi.items()))
    return collapsed


def youtube_ids(url_pose_csv: Path) -> pd.DataFrame:
    """video_number -> 11-char YouTube id, from the pose dataset's URL list."""
    df = pd.read_csv(url_pose_csv)
    ids = df["url"].astype(str).str.extract(r"([0-9A-Za-z_-]{11})")[0]
    return pd.DataFrame({"video_number": df["video_number"], "yt_id": ids}).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=Path("kinematics/data/interim/features_youtube_vitpose.pkl"))
    ap.add_argument("--chambers", type=Path, default=Path("Chamber-etal-dataset"),
                    help="dir holding URL_pose_dataset.csv and meta_data_yt.pkl")
    ap.add_argument("--meta-yt", type=Path, default=None,
                    help="meta_data_yt.pkl (default: <chambers>/meta_data_yt.pkl)")
    ap.add_argument("--out", type=Path,
                    default=Path("kinematics/data/interim/fig5_youtube_vitpose.pkl"))
    args = ap.parse_args()

    feats = pd.read_pickle(args.features)
    print(f"in : {feats.shape[0]} videos x {feats.shape[1] - 1} features")

    joint = to_joint_naming(feats)
    wrist = [c for c in joint.columns if c.startswith("Wrist_")]
    print(f"     -> {len(joint.columns) - 1} joint-level features "
          f"({len(wrist)} wrist)")
    joint = collapse_windows(joint)

    url_pose = args.chambers / "URL_pose_dataset.csv"
    meta_yt = args.meta_yt or (args.chambers / "meta_data_yt.pkl")
    for p in (url_pose, meta_yt):
        if not p.is_file():
            raise SystemExit(
                f"missing {p}\n"
                "meta_data_yt.pkl lives inside infant_movement_assessment_repo_files.zip\n"
                "at data/video_meta_data/ -- unzip it, or pass --meta-yt.")

    bridge = youtube_ids(url_pose)
    ages = pd.read_pickle(meta_yt)[["video_number", "age_in_weeks", "infant_id"]]
    key = bridge.merge(ages, on="video_number", how="inner")

    merged = joint.merge(key, left_on="video", right_on="yt_id", how="inner")
    merged["risk"] = 0          # YouTube infants are the reference group

    dropped = len(joint) - len(merged)
    print(f"out: {len(merged)} videos with an age "
          f"({dropped} dropped -- not in URL_pose_dataset.csv)")
    if len(merged):
        print(f"     age_in_weeks {merged['age_in_weeks'].min():.0f}"
              f"-{merged['age_in_weeks'].max():.0f}, "
              f"{merged['infant_id'].nunique()} distinct infants")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_pickle(args.out)
    merged.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"\n{args.out}\n{args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
