"""Figure 1, version 7 (revision 2): workflow with the FS1-FS5 ablation, SHAP, area of applicability and campaign identification.
Earlier: v6 (revision 1) showed FS1/FS5 only. Same 3 x 3 snake layout as v6.
Writes results/paper1_rev2_v1/figures/Fig1_workflow_v7.png/.pdf. Run: python src/paper1/fig1_workflow_v7.py"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).resolve().parents[2] / 'results' / 'paper1_rev2_v1' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans'})
INK, EDGE = '#2b2b28', '#6b6b64'
OUTSIDE, INSIDE = '#f3f3f0', '#e3edf8'   # grey: whole data / evaluation; blue: inside each training split

rows = [
    [('Data', 'ILRI household surveys\n3,340 households, 462 locations\n5 campaigns, 2012-2019', OUTSIDE),
     ('Exclusions', 'Never used as predictors:\nWeight, survey year, region,\nyear-region code, identifiers', OUTSIDE),
     ('Validation design', '81 km blocks; 3 x 5 spatial\nfolds; random 5-fold CV\nas a reference', OUTSIDE)],
    [('Features (ablation)', 'FS1 environment; FS2 + climate;\nFS3 + coordinates; FS4 + rates\n(training rows, own block out);\nFS5 + sampling density', INSIDE),
     ('Models', 'ExtraTrees (FS1-FS5);\n23-model rank average;\ninner-selected single model', INSIDE),
     ('Calibration', 'Platt scaling and Youden\nthreshold fitted on inner\nout-of-fold predictions', INSIDE)],
    [('Evaluation', 'ROC-AUC, PR-AUC, Brier,\ncalibration with CIs; corrected t,\ncluster bootstrap, Holm', OUTSIDE),
     ('Leakage, applicability,\nexplainability', 'Leakage audit, permutations;\narea of applicability, buffers,\nblocks; SHAP, dependence', OUTSIDE),
     ('Campaigns and time', 'Within-year AUC, year\nidentification, campaign-\nspecific models, forward test', OUTSIDE)],
]
W, H, GX, GY = 3.3, 1.35, 0.5, 0.45
fig, ax = plt.subplots(figsize=(7.4, 4.7)); ax.axis('off')
ax.set_xlim(-0.1, 3 * W + 2 * GX + 0.1); ax.set_ylim(-0.55, 3 * H + 2 * GY + 0.05)
centres = {}
for r, row in enumerate(rows):
    y = (2 - r) * (H + GY)
    for c, (head, body, fc) in enumerate(row):
        cc = c if r != 1 else 2 - c          # snake order: row 2 runs right to left
        x = cc * (W + GX)
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=EDGE, lw=0.8))
        two = '\n' in head
        ax.text(x + W / 2, y + H - (0.27 if two else 0.17), head, ha='center', va='center', fontsize=8.2 if two else 8.6, fontweight='bold', color=INK, linespacing=1.1)
        ax.text(x + W / 2, y + H / 2 - (0.2 if two else 0.12), body, ha='center', va='center', fontsize=6.8, color=INK, linespacing=1.3)
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
fig.savefig(FIG / 'Fig1_workflow_v7.png', dpi=300, bbox_inches='tight'); fig.savefig(FIG / 'Fig1_workflow_v7.pdf', bbox_inches='tight')
print('saved Fig1_workflow_v7')
