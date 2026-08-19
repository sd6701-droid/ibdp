#!/usr/bin/env python3
'''
Bayesian surprise scores from kinematic features (Chambers et al. stage 4/5),
runnable WITHOUT their metadata pickles: one video = one infant, one age
bracket. The math is compute_surprise.py's, unchanged: fit a Gaussian per
(feature, body part) on the REFERENCE cohort, score every video's 38 selected
features by negative log-probability, sum, then z/p against the reference
surprise distribution.

    # score the reference cohort against itself (baseline / outlier scan)
    python kinematics/run_surprise.py \
        --reference kinematics/data/interim/features_chambers.pkl

    # score clinical videos against the YouTube reference
    python kinematics/run_surprise.py \
        --reference kinematics/data/interim/features_chambers.pkl \
        --target kinematics/data/interim/features_clinical.pkl \
        --out kinematics/data/interim/surprise_clinical.csv

Interpretation: despite its name (kept from Chambers' code), the summed
'minus_log_pfeature' is the LOG-LIKELIHOOD of the features under the
reference Gaussians -- so ATYPICAL movement shows up as a strongly NEGATIVE
z (low likelihood), and p is two-sided on |z|. Low p = atypical relative to
the reference population. This is Chambers' continuous risk signal, not a
low/mod/high label -- those need clinical (BINS) metadata to validate
against.
'''
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent

# The 38 features compute_surprise.py sums over (part_featurename form).
FEATURE_LIST = [
    'Ankle_medianx', 'Wrist_medianx', 'Ankle_mediany', 'Wrist_mediany',
    'Knee_mean_angle', 'Elbow_mean_angle',
    'Ankle_IQRx', 'Wrist_IQRx', 'Ankle_IQRy', 'Wrist_IQRy',
    'Knee_stdev_angle', 'Elbow_stdev_angle',
    'Ankle_medianvelx', 'Wrist_medianvelx', 'Ankle_medianvely', 'Wrist_medianvely',
    'Knee_median_vel_angle', 'Elbow_median_vel_angle',
    'Ankle_IQRvelx', 'Wrist_IQRvelx', 'Ankle_IQRvely', 'Wrist_IQRvely',
    'Knee_IQR_vel_angle', 'Elbow_IQR_vel_angle',
    'Ankle_IQRaccx', 'Wrist_IQRaccx', 'Ankle_IQRaccy', 'Wrist_IQRaccy',
    'Knee_IQR_acc_angle', 'Elbow_IQR_acc_angle',
    'Ankle_meanent', 'Wrist_meanent', 'Knee_entropy_angle', 'Elbow_entropy_angle',
    'Ankle_lrCorr_x', 'Wrist_lrCorr_x', 'Knee_lrCorr_angle', 'Elbow_lrCorr_angle',
]


def melt_features(df: pd.DataFrame) -> pd.DataFrame:
    '''Wide (one row per video) -> long (video, part, feature_name, Value),
    averaged across left/right sides -- merge_data_sets.py's reshape.'''
    value_cols = [c for c in df.columns if c != 'video']
    long = df.melt(id_vars=['video'], value_vars=value_cols,
                   var_name='feature', value_name='Value')
    toks = long.feature.str.split('_')
    # 'medianvelx_LAnkle' -> side L, part Ankle, name medianvelx
    # 'mean_angle_LKnee'  -> side L, part Knee,  name mean_angle
    # 'lrCorr_x_Ankle'    -> side '', part Ankle, name lrCorr_x
    long['part'] = [t[-1] if t[0] == 'lrCorr' else t[-1][1:] for t in toks]
    long['feature_name'] = ['_'.join(t[:-1]) for t in toks]
    return (long.groupby(['video', 'part', 'feature_name'], as_index=False)
                .Value.mean())


def surprise_scores(long: pd.DataFrame, ref_stats: pd.DataFrame) -> pd.DataFrame:
    m = pd.merge(long, ref_stats, on=['feature_name', 'part'], how='inner')
    m['feature'] = m.part + '_' + m.feature_name
    m = m[m.feature.isin(FEATURE_LIST) & m.Value.notna() & (m.var_ref > 0)]
    m['minus_log_pfeature'] = -1 * (
        .5 * np.log(2 * np.pi * m['var_ref'])
        + ((m['Value'] - m['mean_ref']) ** 2) / (2 * m['var_ref']))
    out = m.groupby('video').agg(
        surprise=('minus_log_pfeature', 'sum'),
        n_features=('minus_log_pfeature', 'size')).reset_index()
    # A video missing many features (bad tracking) sums fewer terms and looks
    # spuriously "normal"; normalise per feature as well so it is comparable.
    out['surprise_per_feature'] = out.surprise / out.n_features
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path,
                    default=HERE / 'data/interim/features_chambers.pkl',
                    help='features pkl of the reference (healthy) cohort')
    ap.add_argument('--target', type=Path, default=None,
                    help='features pkl to score (default: the reference itself)')
    ap.add_argument('--out', type=Path, default=None,
                    help='output csv (default: surprise_<target stem>.csv)')
    ap.add_argument('--min-features', type=int, default=19,
                    help='flag videos scoring on fewer than this many of the '
                         '38 features (half by default)')
    args = ap.parse_args()

    ref = pd.read_pickle(args.reference)
    tgt = pd.read_pickle(args.target) if args.target else ref
    tgt_name = (args.target or args.reference).stem.replace('features_', '')

    ref_long = melt_features(ref)
    tgt_long = melt_features(tgt) if args.target else ref_long

    # Gaussian per (feature, part) on the reference -- norm.fit is the MLE,
    # i.e. mean and ddof=0 std, as in compute_surprise.py.
    ref_stats = (ref_long.groupby(['feature_name', 'part'])
                 .Value.agg(mean_ref='mean', sd_ref=lambda x: x.std(ddof=0))
                 .reset_index())
    ref_stats['var_ref'] = ref_stats.sd_ref ** 2

    # Reference surprise distribution is what z is measured against
    # (Chambers: z against the risk==0 group -- here, the whole reference).
    ref_surprise = surprise_scores(ref_long, ref_stats)
    mu, sd = ref_surprise.surprise.mean(), ref_surprise.surprise.std()

    scores = (ref_surprise if args.target is None
              else surprise_scores(tgt_long, ref_stats))
    scores['z'] = (scores.surprise - mu) / sd
    scores['p'] = (norm.sf(np.abs(scores.z)) * 2).round(4)
    scores['low_coverage'] = scores.n_features < args.min_features
    # Most atypical first: lowest likelihood = most negative z.
    scores = scores.sort_values('z')

    out = args.out or HERE / 'data/interim' / f'surprise_{tgt_name}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)

    print(f'reference: {args.reference.name} ({ref.video.nunique()} videos, '
          f'surprise mu={mu:.1f} sd={sd:.1f})')
    print(f'scored   : {tgt_name} ({len(scores)} videos) -> {out}')
    print()
    cols = ['video', 'surprise', 'n_features', 'z', 'p', 'low_coverage']
    print(scores[cols].to_string(index=False,
                                 float_format=lambda v: f'{v:.3f}'))
    flagged = scores[scores.low_coverage]
    if len(flagged):
        print(f'\nNOTE: {len(flagged)} video(s) scored on < {args.min_features} '
              f'of 38 features (poor tracking) -- interpret with care.')


if __name__ == '__main__':
    main()
