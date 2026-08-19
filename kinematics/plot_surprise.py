#!/usr/bin/env python3
'''
Chambers-style risk-group figure from run_surprise.py output.

    python kinematics/plot_surprise.py \
        --scores kinematics/data/interim/surprise_clinical.csv \
        --out    kinematics/data/interim/surprise_clinical_plot.png

One labelled row per risk group, one dot per video at its surprise z; group
mean as a thin vertical tick; dashed reference at z = -1.96 (p = .05, the
atypicality threshold). Dots left of the dashed line are videos whose
movement is significantly unlikely under the reference population.
'''
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slots 1-3 of the validated default palette (all-pairs safe).
COLORS = {'low': '#2a78d6', 'moderate': '#eb6834', 'high': '#1baf7a'}
ORDER = ['low', 'moderate', 'high']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', type=Path, required=True,
                    help='surprise csv from run_surprise.py (needs z + risk)')
    ap.add_argument('--reference-scores', type=Path, default=None,
                    help='surprise csv of the reference cohort scored against '
                         'itself -- adds a baseline row to the figure')
    ap.add_argument('--out', type=Path, default=None,
                    help='output image (default: <scores>_plot.png)')
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    if 'risk' not in df.columns:
        raise SystemExit('no risk column -- rerun run_surprise.py with --labels')
    df = df[df.risk.isin(ORDER)]

    groups = {g: (df[df.risk == g], COLORS[g]) for g in ORDER}
    if args.reference_scores:
        ref = pd.read_csv(args.reference_scores)
        # Baseline population row first, in a neutral -- it is the yardstick,
        # not a risk group.
        groups = {'YouTube reference': (ref, '#8b897f'), **groups}

    def bands(g):
        a = int((g.z <= -1.96).sum())
        b = int(((g.z > -1.96) & (g.z <= -1)).sum())
        return a, b, len(g) - a - b

    fig, ax = plt.subplots(figsize=(9.5, 1.3 + 1.05 * len(groups)), dpi=150)
    # Segregation zones: atypical / borderline / typical, shared by every row.
    # Anchored to the DATA range, not a sentinel -- a huge span x would drag
    # autoscale out and squash every dot against the right edge.
    all_z = pd.concat([g.z for g, _ in groups.values()])
    x_lo = min(all_z.min(), -2.4) - .3
    x_hi = max(all_z.max(), .5) + .3
    ax.axvspan(x_lo, -1.96, color='#e34948', alpha=.06, zorder=0)
    ax.axvspan(-1.96, -1, color='#eda100', alpha=.07, zorder=0)
    ax.set_xlim(x_lo, x_hi)
    rng = np.random.default_rng(0)
    labels = []
    for i, (group, (g, color)) in enumerate(groups.items()):
        y = i + rng.uniform(-.13, .13, len(g))          # jitter within the row
        ax.scatter(g.z, y, s=64, color=color, alpha=.85,
                   edgecolors='white', linewidths=1.5, zorder=3)
        ax.plot([g.z.mean()] * 2, [i - .28, i + .28],
                color=color, lw=2, zorder=2)
        # every atypical dot gets named -- infant id on clinical rows, the
        # video id on the reference row (few enough there to stay legible)
        for _, r in g[g.z <= -1.96].iterrows():
            name = (str(r.get('infant', '')) if group in ORDER
                    else str(r.video)[:11])
            ax.annotate(name, (r.z, y[g.index.get_loc(r.name)]),
                        textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=7.5, color='#52514e')
        a, b, t = bands(g)
        labels.append(f'{group} (n={len(g)})\n{a} atyp · {b} bord · {t} typ')

    n_rows = len(groups)
    ax.axvline(-1.96, color='#9b9a93', lw=1, ls='--', zorder=1)
    ax.text(-1.96, n_rows - .38, 'z = −1.96 (p = .05)', ha='center',
            fontsize=8, color='#52514e')
    ax.axvline(0, color='#d6d5cd', lw=1, zorder=0)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_ylim(-.55, n_rows - .2)
    ax.invert_yaxis()
    ax.set_xlabel('Bayesian surprise, z vs YouTube reference '
                  '(← more atypical movement)', fontsize=9)
    ax.set_title('Movement atypicality by clinical risk group '
                 '(BINS, Chambers et al. cohort)', fontsize=11, loc='left')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis='x', color='#eceae4', lw=.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = args.out or args.scores.with_name(args.scores.stem + '_plot.png')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
