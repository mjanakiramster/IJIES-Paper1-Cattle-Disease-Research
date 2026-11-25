"""Figure 1, version 3: study workflow in a 3 x 3 snake layout; v3 widens boxes and shortens lines so no text touches a border (v1 overflowed, v2 was tight).
Writes results/paper1_v1/figures/Fig1_workflow_v3.png/.pdf. Run: python src/paper1/fig1_workflow_v2.py"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).resolve().parents[2] / 'results' / 'paper1_v1' / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans'})
INK, EDGE = '#2b2b28', '#6b6b64'
OUTSIDE, INSIDE = '#f3f3f0', '#e3edf8'   # grey: whole data / evaluation; blue: inside each training split

rows = [
    [('Data', 'ILRI household surveys\n3,340 households, 462 locations\n2012, 2014, 2015, 2018, 2019', OUTSIDE),
     ('Exclusions', 'Never used as predictors:\nWeight, survey year, region,\nyear-region code, identifiers', OUTSIDE),
     ('Spatial validation', '81 km blocks; 3 x 5 development\nfolds; locked 2 x 5 confirmatory\npartition, opened once', OUTSIDE)],
    [('Features', 'FS1: 5 environmental variables\nFS5: 39 features; rates\ntraining rows only,\nown 81 km block excluded', INSIDE),
     ('Models', 'ExtraTrees (FS1, FS5);\n23-model rank average;\ninner-selected single model', INSIDE),
     ('Calibration', 'Platt scaling and Youden\nthreshold fitted on inner\nout-of-fold predictions', INSIDE)],
    [('Evaluation', 'ROC-AUC, PR-AUC, Brier,\ncalibration; overall and\nwithin-survey-year AUC', OUTSIDE),
     ('Sensitivity', 'Label permutations, proximity\nbuffers, block size, forward\ntemporal test (2012-15 -> 2018-19)', OUTSIDE),
     ('Inference', 'Corrected resampled t,\nlocation-cluster bootstrap,\nHolm correction', OUTSIDE)],
]
W, H, GX, GY = 3.3, 1.2, 0.5, 0.45
fig, ax = plt.subplots(figsize=(7.4, 4.3)); ax.axis('off')
ax.set_xlim(-0.1, 3 * W + 2 * GX + 0.1); ax.set_ylim(-0.55, 3 * H + 2 * GY + 0.05)
centres = {}
for r, row in enumerate(rows):
    y = (2 - r) * (H + GY)
    for c, (head, body, fc) in enumerate(row):
        cc = c if r != 1 else 2 - c          # snake order: row 2 runs right to left
        x = cc * (W + GX)
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=EDGE, lw=0.8))
        ax.text(x + W / 2, y + H - 0.17, head, ha='center', va='center', fontsize=8.6, fontweight='bold', color=INK)
        ax.text(x + W / 2, y + H / 2 - 0.12, body, ha='center', va='center', fontsize=7.0, color=INK, linespacing=1.35)
        centres[(r, c)] = (x, y)
arrow = dict(arrowstyle='-|>', color=EDGE, lw=0.9, mutation_scale=9)
order = [(r, c) for r in range(3) for c in range(3)]
for a, b in zip(order[:-1], order[1:]):
    (xa, ya), (xb, yb) = centres[a], centres[b]
    if ya == yb:
        if xb > xa: ax.annotate('', xy=(xb, ya + H / 2), xytext=(xa + W, ya + H / 2), arrowprops=arrow)
        else: ax.annotate('', xy=(xb + W, yb + H / 2), xytext=(xa, ya + H / 2), arrowprops=arrow)
    else:
        ax.annotate('', xy=(xb + W / 2, yb + H), xytext=(xa + W / 2, ya), arrowprops=arrow)
ax.add_patch(FancyBboxPatch((0.15, -0.45), 0.35, 0.22, boxstyle='round,pad=0.01', fc=INSIDE, ec=EDGE, lw=0.6))
ax.text(0.6, -0.34, 'Steps repeated inside each training split; no outcome labels from the test fold are used', va='center', fontsize=7.2, color=INK)
fig.savefig(FIG / 'Fig1_workflow_v3.png', dpi=300, bbox_inches='tight'); fig.savefig(FIG / 'Fig1_workflow_v3.pdf', bbox_inches='tight')
print('saved Fig1_workflow_v3')
