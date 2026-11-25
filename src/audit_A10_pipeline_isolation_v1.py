"""Feature-leakage audit A10 (2026-11-21): full pipeline isolation checklist and pipeline diagram.
Each item records the operation, what it is fitted on, the code evidence, and the executed evidence from other audit
stages where available. Run from the project root."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'results/audit_v1/A10'; OUT.mkdir(parents=True, exist_ok=True)
C = [
 ('Imputation', 'Not used: 0 missing cells (Stage 1); no imputer anywhere in the code (grep: Imputer/fillna = 0 hits)', 'n/a', 'Pass'),
 ('Scaling', 'StandardScaler inside make_pipeline for LR, GAM, kNN, MLP (run3 L231-L241) and SVM (stage 4 L55, stage 9 L49); GP scaler fitted on training locations (run3 L251)', 'training rows of each fit call', 'Pass'),
 ('Neighbourhood / cross-disease features', 'Built from Y[fit_idx] with own-block exclusion (features.py L41-L72; run3 L162-L198)', 'training rows only', 'Pass (A2, A3: test-label flips change 0 features)'),
 ('Fallback prior (old run-3 code)', 'p0 over all training rows incl. the query\'s own block (run3 L167, L185)', 'training rows only (no test labels)', 'Deviation from stated rule; consequence measured in A5'),
 ('Feature selection', 'Feature sets fixed in advance (B, E, EL, ELK, FS1-FS5); SGO/optimiser search used inner spatial folds of the outer training set only (stage5_worker.py)', 'inner folds of training set', 'Pass'),
 ('Hyperparameter tuning', 'None: hyperparameters fixed in advance, identical in every split (run3 L213; stage 4 CONFIG)', 'n/a', 'Pass'),
 ('Probability calibration', 'SVM Platt scaling fitted on inner out-of-fold decision values (stage 4 L92, L112)', 'inner OOF of training set', 'Pass'),
 ('Threshold selection', 'Youden index on inner out-of-fold predictions (experiment_config.yaml L57; stage 4 L94, L130)', 'inner OOF of training set', 'Pass'),
 ('Model selection, (b) best single', 'argmax of mean inner-fold AUC (stage 4 L118)', 'inner OOF of training set', 'Pass'),
 ('Ensemble construction, (a) rank average', 'Equal-weight mean of each model\'s ECDF rank; ECDF from training OOF scores (stage 4 L98-L103, L119; stage 9 ecdf_rank_average)', 'training OOF scores; no weights fitted', 'Pass'),
 ('Ensemble weighting (RB-HGWOPSO, run 3)', 'Weights optimised on inner OOF with bagged training locations (run3 L395, L452, L483)', 'inner OOF of training set', 'Pass'),
 ('Group-balanced weights', 'Year_region used only as training sample weights for one CatBoost model (run3 L261, L274)', 'training rows only', 'Pass (design variable not a feature)'),
 ('Prediction', 'Test rows receive features built from training rows; models applied once', 'n/a', 'Pass (A3: predictions bit-identical after test-label flips)'),
 ('Identical outer folds across models', 'All comparisons use the same development folds (stage3_folds_v1.csv); Stage 9 used the sealed confirmatory partition once, hash-checked', 'n/a', 'Pass'),
 ('Confirmatory partition', 'Opened once (2026-11-20T14:19:31Z) for pre-registered H1/H2; the audit uses development folds only', 'n/a', 'Pass'),
]
T = pd.DataFrame(C, columns=['Pipeline step', 'Implementation and code evidence', 'Fitted on', 'Audit result']); T.to_csv(OUT / 'A10_pipeline_isolation_checklist_v1.csv', index=False)

steps = [('Raw data\n3 Excel files (MD5-audited)', '#eeeeea'), ('Exclude Weight, Year,\nRegion, Year_region', '#f6d7d6'), ('Outer split\n(81 km spatial blocks)', '#dbe8fa'),
         ('Label-free features\n(climate, coordinates)', '#eeeeea'), ('Neighbourhood + cross-disease rates\nfrom TRAINING rows, own block excluded', '#dbe8fa'),
         ('Inner spatial folds\n(training rows only)', '#dbe8fa'), ('Fit models; scaler, Platt,\nthreshold, ECDF on inner OOF', '#dbe8fa'), ('Predict test fold\n(once)', '#d9f2e6'), ('Metrics\n(ROC-AUC, PR-AUC, ...)', '#eeeeea')]
fig, ax = plt.subplots(figsize=(15, 3.2)); ax.axis('off'); ax.set_xlim(0, len(steps) * 1.7); ax.set_ylim(0, 2)
for i, (s, c) in enumerate(steps):
    x = i * 1.7 + 0.08; ax.add_patch(FancyBboxPatch((x, 0.55), 1.45, 0.95, boxstyle='round,pad=0.02', fc=c, ec='#55554f', lw=0.8))
    ax.text(x + 0.725, 1.025, s, ha='center', va='center', fontsize=7.6)
    if i < len(steps) - 1: ax.annotate('', xy=(x + 1.62, 1.025), xytext=(x + 1.47, 1.025), arrowprops=dict(arrowstyle='->', color='#55554f'))
ax.text(0.08, 0.25, 'Blue = fitted or computed inside the training data of each split only.  Test labels are used only in the last step.', fontsize=8, color='#444')
ax.set_title('A10: model pipeline and where each operation is fitted', loc='left', fontsize=10)
fig.savefig(OUT / 'A10_pipeline_diagram_v1.png', dpi=200, bbox_inches='tight'); print(T.to_string(index=False))
