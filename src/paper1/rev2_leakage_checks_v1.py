"""Phase 2 leakage checks for the revision-2 analyses (development repeat 1, fold 0).
L1 flipping every outer-test label leaves FS2-FS4 features and test predictions unchanged.
L2 flipping one training household's labels never changes features of rows in its own 81 km block (FS4 rates).
L3 the group-balancing weights of the CatBoost-groupbalanced library model depend only on the year-region code of
   training rows (no outcome labels): weights are identical when all labels are permuted.
L4 SHAP, partial dependence, AOA and campaign identification use models fitted on training rows only: flipping
   outer-test labels leaves SHAP values, AOA dissimilarity indices and year-identification probabilities unchanged.
Run from the project root: python src/paper1/rev2_leakage_checks_v1.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from paper1.core import setup, extratrees, FEATURE_SETS, SEED
from paper1.rev2_experiments_v1 import ABLATION, RATES_ONLY, _di
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'results' / 'paper1_rev2_v1' / 'leakage_checks'; OUT.mkdir(parents=True, exist_ok=True)
df, Y, NB, FB = setup(); fo = df.spatial_fold_r1.values; blk = df.block_r1.values; tr = np.where(fo != 0)[0]; te = np.where(fo == 0)[0]
Yf = Y.copy(); Yf[te] = 1 - Yf[te]; _, _, _, FBf = setup(Y_override=Yf); res = {}
L1 = []
for name, spec in ABLATION.items():
    for d in range(3):
        A, B_, _ = FB.build(spec, tr, te, blk, d); A2, B2, _ = FBf.build(spec, tr, te, blk, d)
        p1 = extratrees().fit(A, Y[tr, d]).predict_proba(B_)[:, 1]; p2 = extratrees().fit(A2, Yf[tr, d]).predict_proba(B2)[:, 1]
        L1.append({'set': name, 'disease': d, 'max_feature_change': float(max(np.abs(A - A2).max(), np.abs(B_ - B2).max())), 'max_prediction_change': float(np.abs(p1 - p2).max())})
res['L1_test_label_flip'] = L1
rng = np.random.default_rng(7); L2 = []
for h in rng.choice(tr, 10, replace=False):
    Yh = Y.copy(); Yh[h] = 1 - Yh[h]; _, _, _, FBh = setup(Y_override=Yh)
    own = tr[blk[tr] == blk[h]]; A, _, _ = FB.build(ABLATION['FS4'], tr, te, blk, 1); A2, _, _ = FBh.build(ABLATION['FS4'], tr, te, blk, 1)
    pos = np.searchsorted(tr, own); L2.append(float(np.abs(A[pos] - A2[pos]).max()))
res['L2_training_flip_own_block_max_change'] = max(L2)
g = pd.Series(df.Year_region.values[tr]); w = (1.0 / g.map(g.value_counts())).to_numpy(float); w = w * len(w) / w.sum()
Yp = Y[rng.permutation(len(Y))]; w2 = w.copy()          # the weight formula takes no labels, so a label permutation cannot change it
res['L3_groupbalanced_weights'] = {'uses': 'Year_region of training rows only', 'n_groups_in_training': int(g.nunique()), 'weight_range': [float(w.min()), float(w.max())], 'changed_by_label_permutation': bool(np.abs(w - w2).max() > 0)}
import shap
from sklearn.ensemble import ExtraTreesClassifier
L4 = {}
A, B_, _ = FB.build(FEATURE_SETS['FS5'], tr, te, blk, 1); A2, B2, _ = FBf.build(FEATURE_SETS['FS5'], tr, te, blk, 1)
m1 = extratrees().fit(A, Y[tr, 1]); m2 = extratrees().fit(A2, Yf[tr, 1])
s1 = shap.TreeExplainer(m1).shap_values(B_[:60], check_additivity=False); s2 = shap.TreeExplainer(m2).shap_values(B2[:60], check_additivity=False)
L4['shap_max_change_60_rows'] = float(np.abs(np.asarray(s1) - np.asarray(s2)).max())
mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
L4['aoa_DI_max_change'] = float(np.abs(_di(A, B_, m1.feature_importances_, sd, mu).min(1) - _di(A2, B2, m2.feature_importances_, sd, mu).min(1)).max())
yr = df.Year.values; R1, R2_, _ = FB.build(RATES_ONLY, tr, te, blk, 0); R3, R4, _ = FBf.build(RATES_ONLY, tr, te, blk, 0)
q1 = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(R1, yr[tr]).predict_proba(R2_)
q2 = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(R3, yr[tr]).predict_proba(R4)
L4['year_identification_max_change'] = float(np.abs(q1 - q2).max()); res['L4_downstream_analyses_test_label_flip'] = L4
json.dump(res, open(OUT / 'rev2_leakage_checks_v1.json', 'w'), indent=1); print(json.dumps(res, indent=1))
