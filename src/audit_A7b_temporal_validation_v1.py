"""Feature-leakage audit A7b (2026-11-21), EXPLORATORY: forward temporal validation, the proper test of PROSPECTIVE use.
Train on households surveyed 2012-2015 (1,564), test on households surveyed 2018-2019 (1,776) — the dataset's existing
'temporal_set' split. Neighbourhood / cross-disease rates for every row come only from the 2012-2015 training households
(so no later report is ever used), and each query location still excludes its own 81 km block (development repeat 1
blocks). FS1 vs FS5, ExtraTrees-300; 95% CIs from a location-cluster bootstrap of the test set (1,000 resamples).
Run from the project root."""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); ROOT = SRC.parent
from common.data import load, labels, SEED, ENV
from common.features import Neighbourhood, climate_coords, CLIMATE_NAMES
from common.fitness import FeatureBuilder, auc_avg
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score
warnings.filterwarnings('ignore')
OUT = ROOT / 'results/audit_v1/A7'; OUT.mkdir(parents=True, exist_ok=True)
df = load(ROOT / 'data/raw'); Y = labels(df); FB = FeatureBuilder(df, Y, Neighbourhood(df, Y), climate_coords(df)); loc = df['loc'].values
tr = np.where(df.temporal_set.values == 'train')[0]; te = np.where(df.temporal_set.values == 'test')[0]
base = {'coordinates': False, 'kernel_rates': False, 'knn_rates': False, 'idw_rates': False, 'density_terms': False, 'cross_disease_rates': False,
        'bandwidths': (50.0, 150.0, 400.0), 'ks': (5, 15, 40), 'k': 15, 'prior': 1.0, 'idw_power': 2.0, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'complexity': 0}
FS = {'FS1': {**base, 'climate': ENV}, 'FS5': {**base, 'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'density_terms': True, 'cross_disease_rates': True}}
rng = np.random.default_rng(20261121); tl = loc[te]; ul = np.unique(tl); groups = {l: np.where(tl == l)[0] for l in ul}
boots = [np.concatenate([groups[l] for l in rng.choice(ul, len(ul), replace=True)]) for _ in range(1000)]
rows = []
for d, dn in enumerate(['CBPP', 'FMD', 'LSD']):
    P = {}
    for name, spec in FS.items():
        Xf, Xp, _ = FB.build(spec, tr, te, df.block_r1.values, d)
        P[name] = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(Xf, Y[tr, d]).predict_proba(Xp)[:, 1]
    yt = Y[te, d]; g = np.array([auc_avg(yt[b], P['FS5'][b]) - auc_avg(yt[b], P['FS1'][b]) for b in boots])
    rows.append({'disease': dn, 'train_prevalence': Y[tr, d].mean(), 'test_prevalence': yt.mean(), 'FS1_ROC_AUC': auc_avg(yt, P['FS1']), 'FS5_ROC_AUC': auc_avg(yt, P['FS5']),
                 'gain': auc_avg(yt, P['FS5']) - auc_avg(yt, P['FS1']), 'gain_CI_lo': np.nanquantile(g, 0.025), 'gain_CI_hi': np.nanquantile(g, 0.975),
                 'FS1_PR_AUC': average_precision_score(yt, P['FS1']), 'FS5_PR_AUC': average_precision_score(yt, P['FS5'])})
R = pd.DataFrame(rows); R.to_csv(OUT / 'A7b_forward_temporal_validation_v1.csv', index=False)
pd.set_option('display.width', 250); print(R.round(4).to_string(index=False))
