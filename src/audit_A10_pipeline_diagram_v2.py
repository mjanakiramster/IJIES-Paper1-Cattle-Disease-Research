"""A10 pipeline diagram v2 (v1 text overflowed the boxes): two rows, wrapped labels."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT = Path(__file__).resolve().parents[1] / 'results/audit_v1/A10'
steps = [('Raw data\n3 Excel files\n(MD5-audited)', '#eeeeea'), ('Exclude Weight,\nYear, Region,\nYear_region', '#f6d7d6'), ('Outer split\n81 km spatial\nblocks', '#dbe8fa'),
         ('Label-free features\nclimate indices,\ncoordinates', '#eeeeea'), ('Disease-rate features\nfrom TRAINING rows,\nown block excluded', '#dbe8fa'),
         ('Inner spatial folds\ninside training\nrows only', '#dbe8fa'), ('Fit models; scaler,\nPlatt, threshold,\nECDF on inner OOF', '#dbe8fa'), ('Predict the\ntest fold once', '#d9f2e6'), ('Metrics\nROC-AUC, PR-AUC,\nBrier, MCC ...', '#eeeeea')]
fig, ax = plt.subplots(figsize=(12, 5.2)); ax.axis('off'); ax.set_xlim(0, 10.2); ax.set_ylim(0, 5)
pos = [(i * 2.0 + 0.2, 3.0) for i in range(5)] + [((8 - i * 2.0) + 0.2, 0.8) for i in range(4)]
for i, ((s, c), (x, y)) in enumerate(zip(steps, pos)):
    ax.add_patch(FancyBboxPatch((x, y), 1.7, 1.25, boxstyle='round,pad=0.03', fc=c, ec='#55554f', lw=0.8)); ax.text(x + 0.85, y + 0.625, s, ha='center', va='center', fontsize=8.5)
    if i < len(steps) - 1:
        x2, y2 = pos[i + 1]
        if y2 == y: ax.annotate('', xy=(x2 - 0.02, y + 0.625) if x2 > x else (x2 + 1.72, y + 0.625), xytext=(x + 1.72, y + 0.625) if x2 > x else (x - 0.02, y + 0.625), arrowprops=dict(arrowstyle='->', color='#55554f'))
        else: ax.annotate('', xy=(x2 + 0.85, y2 + 1.27), xytext=(x + 0.85, y - 0.02), arrowprops=dict(arrowstyle='->', color='#55554f'))
ax.text(0.2, 0.25, 'Blue = fitted or computed inside the training data of each split only. Test labels are used only for the final metrics.', fontsize=8.5, color='#444')
ax.set_title('A10: model pipeline and where each operation is fitted', loc='left', fontsize=10.5)
fig.savefig(OUT / 'A10_pipeline_diagram_v2.png', dpi=200, bbox_inches='tight')
