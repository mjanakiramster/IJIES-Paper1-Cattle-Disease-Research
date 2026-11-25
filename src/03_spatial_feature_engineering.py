"""Stage 3 — feature dictionary, feature-set comparison on DEVELOPMENT folds, mutual-information selection,
and the pre-registered surrogate fidelity check. The confirmatory partition is not touched.
Run from the project root:  python src/03_spatial_feature_engineering.py"""
import sys, json, time, warnings, datetime
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load, labels, DIS, ENV, SEED
from common.features import climate_coords, Neighbourhood, CLIMATE_NAMES
from common.fitness import FeatureBuilder, fit_predict, robust_fitness, auc_avg
from common.folds import inner_spatial_folds
from common.space import decode, random_positions
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

OUT = Path('results/stage3_v1'); (OUT / 'figures').mkdir(parents=True, exist_ok=False); (OUT / 'engineered_features').mkdir()
df = load('data/raw'); Y = labels(df); NB = Neighbourhood(df, Y); CC = climate_coords(df); FB = FeatureBuilder(df, Y, NB, CC)

# clarification of the fidelity rule, logged BEFORE the check is run
LOGF = Path('configs/decision_log.json'); LOG = json.load(open(LOGF))
LOG.setdefault('clarifications', []).append({'stage': 3, 'time_utc': datetime.datetime.utcnow().isoformat(timespec='seconds'),
    'item': 'surrogate fidelity rule', 'text': 'rho is computed per disease (30 candidates each, outer folds r1/f0, r1/f1, r1/f2 for CBPP, FMD, LSD); '
    'the location-level surrogate is used if the MEDIAN of the three per-disease Spearman rho values (vs the full ExtraTrees-300 fitness) is >= 0.7, otherwise the row-level surrogate.'})
json.dump(LOG, open(LOGF, 'w'), indent=1, default=str)

# ---------- 1. feature dictionary ----------
fd = [{'feature': v, 'group': 'A original environment', 'uses_labels': False, 'definition': f'{v} as supplied'} for v in ENV]
fd += [{'feature': 'SL_temperature', 'group': 'B climate transform', 'uses_labels': False, 'definition': 'Temperature + 0.0065 x Altitude (sea-level equivalent; lapse rate 6.5 C/km)'},
       {'feature': 'DeMartonne', 'group': 'B climate transform', 'uses_labels': False, 'definition': 'Precipitation / (Temperature + 10) (aridity index)'},
       {'feature': 'Temp_x_RH', 'group': 'B climate interaction', 'uses_labels': False, 'definition': 'Temperature x Relative_humidity / 100'},
       {'feature': 'Precip_x_RH', 'group': 'B climate interaction', 'uses_labels': False, 'definition': 'log(1 + Precipitation) x Relative_humidity / 100'},
       {'feature': 'log_Precip', 'group': 'B climate transform', 'uses_labels': False, 'definition': 'log(1 + Precipitation)'},
       {'feature': 'Wind_x_RH', 'group': 'B climate interaction', 'uses_labels': False, 'definition': 'Wind_speed x Relative_humidity / 100'},
       {'feature': 'Lat, Lon, LatPlusLon, LatMinusLon', 'group': 'B coordinates', 'uses_labels': False, 'definition': 'latitude, longitude and two 45-degree rotations'}]
leak = 'training rows only; the query location\'s own spatial block (and its labels) excluded, also from the prior rate'
fd += [{'feature': 'kern{b}_{d}', 'group': 'C neighbourhood (kernel)', 'uses_labels': True, 'definition': f'Gaussian-kernel disease rate, bandwidth b km, prior-shrunk; {leak}'},
       {'feature': 'knn{k}_{d}', 'group': 'C neighbourhood (nearest locations)', 'uses_labels': True, 'definition': f'disease rate among the k nearest training locations, prior-shrunk; {leak}'},
       {'feature': 'idw_{d}', 'group': 'C neighbourhood (inverse distance)', 'uses_labels': True, 'definition': f'inverse-distance-weighted disease rate (power p); {leak}'},
       {'feature': 'dens{b}, knndist{k}', 'group': 'C sampling density', 'uses_labels': False, 'definition': 'log kernel mass of training households; log distance to the k-th nearest training location'},
       {'feature': 'suffix _0/_1/_2', 'group': 'C cross-disease', 'uses_labels': True, 'definition': 'rates of CBPP (0), FMD (1), LSD (2); other-disease rates come only from training rows'}]
pd.DataFrame(fd).to_csv(OUT / 'feature_dictionary.csv', index=False)
CC.assign(ID=df.ID.values).to_csv(OUT / 'engineered_features' / 'climate_coords_v1.csv', index=False)

# ---------- 2. feature-set comparison (development folds, fixed ExtraTrees-300 as in run 3) ----------
base = {'coordinates': False, 'kernel_rates': False, 'knn_rates': False, 'idw_rates': False, 'density_terms': False, 'cross_disease_rates': False,
        'bandwidths': (50.0, 150.0, 400.0), 'ks': (5, 15, 40), 'k': 15, 'prior': 1.0, 'idw_power': 2.0, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'complexity': 0}
FSETS = {'FS1 original 5 variables': {**base, 'climate': ENV},
         'FS2 + climate transforms': {**base, 'climate': CLIMATE_NAMES},
         'FS3 + coordinates': {**base, 'climate': CLIMATE_NAMES, 'coordinates': True},
         'FS4 + same-disease neighbourhood rates': {**base, 'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'density_terms': True},
         'FS5 + cross-disease rates': {**base, 'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'density_terms': True, 'cross_disease_rates': True},
         'FS6 + inverse-distance rates': {**base, 'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'idw_rates': True, 'density_terms': True, 'cross_disease_rates': True}}
MI_NAME = 'FS7 mutual-information top 50% of FS6'
DEV = [(r, f) for r in (1, 2, 3) for f in range(5)]

def one_dev_split(r, f):
    warnings.filterwarnings('ignore')
    te = np.where(df[f'spatial_fold_r{r}'].values == f)[0]; tr = np.where(df[f'spatial_fold_r{r}'].values != f)[0]; blk = df[f'block_r{r}'].values
    rows, imp, mi = [], [], []
    for d in range(3):
        y, yt = Y[tr, d], Y[te, d]
        for name, spec in FSETS.items():
            Xf, Xp, names = FB.build(spec, tr, te, blk, d)
            m = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(Xf, y); p = m.predict_proba(Xp)[:, 1]
            rows.append({'rep': r, 'fold': f, 'disease': DIS[d], 'feature_set': name, 'n_features': Xf.shape[1], 'ROC_AUC': auc_avg(yt, p), 'PR_AUC': average_precision_score(yt, p)})
            if name.startswith('FS6'):
                imp += [{'rep': r, 'fold': f, 'disease': DIS[d], 'feature': n, 'importance': v} for n, v in zip(names, m.feature_importances_)]
                s = mutual_info_classif(Xf, y, random_state=SEED); keep = s >= np.median(s)
                mi += [{'rep': r, 'fold': f, 'disease': DIS[d], 'feature': n, 'mutual_information': v, 'kept': bool(k)} for n, v, k in zip(names, s, keep)]
                p2 = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(Xf[:, keep], y).predict_proba(Xp[:, keep])[:, 1]
                rows.append({'rep': r, 'fold': f, 'disease': DIS[d], 'feature_set': MI_NAME, 'n_features': int(keep.sum()), 'ROC_AUC': auc_avg(yt, p2), 'PR_AUC': average_precision_score(yt, p2)})
    return rows, imp, mi

t0 = time.time()
res = Parallel(n_jobs=2)(delayed(one_dev_split)(r, f) for r, f in DEV)
FM = pd.DataFrame([x for a, _, _ in res for x in a]); IMP = pd.DataFrame([x for _, b, _ in res for x in b]); MI = pd.DataFrame([x for _, _, c in res for x in c])
print(f'feature-set comparison: {(time.time() - t0) / 60:.1f} min')
FM.to_csv(OUT / 'feature_set_fold_metrics.csv', index=False)
ref = pd.read_csv('data/reference/run3_single_model_fold_metrics_v1.csv')
ref = ref[(ref.scheme == 'spatial') & (ref.model == 'ExtraTrees [ELK]')][['rep', 'fold', 'disease', 'ROC_AUC']].assign(fold=lambda t: t.fold.astype(int))

order = list(FSETS) + [MI_NAME]; ratio = np.mean([(df[f'spatial_fold_r{r}'] == f).mean() / (df[f'spatial_fold_r{r}'] != f).mean() for r, f in DEV])
def ct(d_):
    d_ = np.asarray(d_, float); J = len(d_); v = d_.var(ddof=1)
    return np.nan if v == 0 else 2 * stats.t.sf(abs(d_.mean() / np.sqrt((1 / J + ratio) * v)), J - 1)
summ = []
for d in DIS:
    piv = FM[FM.disease == d].pivot_table(index=['rep', 'fold'], columns='feature_set', values='ROC_AUC')
    rr = ref[ref.disease == d].set_index(['rep', 'fold']).ROC_AUC.reindex(piv.index)
    for i, fs in enumerate(order):
        prev = order[i - 1] if 0 < i < len(order) - 1 else (order[-2] if i == len(order) - 1 else None)
        summ.append({'disease': d, 'feature_set': fs, 'n_features': int(FM[(FM.disease == d) & (FM.feature_set == fs)].n_features.median()),
                     'ROC_AUC': piv[fs].mean(), 'PR_AUC': FM[(FM.disease == d) & (FM.feature_set == fs)].PR_AUC.mean(),
                     'dROC_vs_previous_set': (piv[fs] - piv[prev]).mean() if prev else np.nan, 'p_corrected_vs_previous': ct(piv[fs] - piv[prev]) if prev else np.nan,
                     'dROC_vs_FS1': (piv[fs] - piv[order[0]]).mean(), 'run3_ExtraTrees_ELK_reference': rr.mean()})
S = pd.DataFrame(summ); S.to_csv(OUT / 'feature_engineering_comparison.csv', index=False)
IMP.groupby(['disease', 'feature']).importance.mean().unstack('disease').sort_values('FMD', ascending=False).to_csv(OUT / 'feature_importance.csv')
MI.groupby(['disease', 'feature']).agg(mutual_information=('mutual_information', 'mean'), share_of_folds_kept=('kept', 'mean')).reset_index().to_csv(OUT / 'mutual_information_selection.csv', index=False)
te0 = np.where(df.spatial_fold_r1.values == 0)[0]; tr0 = np.where(df.spatial_fold_r1.values != 0)[0]
a, b, n = FB.build(FSETS['FS6 + inverse-distance rates'], tr0, te0, df.block_r1.values, 1)
pd.DataFrame(np.vstack([a, b]), columns=n).assign(ID=np.r_[df.ID.values[tr0], df.ID.values[te0]], role=['train'] * len(tr0) + ['test'] * len(te0)).to_csv(
    OUT / 'engineered_features' / 'example_FS6_FMD_dev_r1_fold0_v1.csv', index=False)

# ---------- 3. surrogate fidelity check (pre-registered) ----------
cands = random_positions(30, np.random.default_rng(SEED + 3))
def fid_one(d, c):
    warnings.filterwarnings('ignore')
    f = d; te = np.where(df.spatial_fold_r1.values == f)[0]; tr = np.where(df.spatial_fold_r1.values != f)[0]
    pairs = inner_spatial_folds(tr, df.block_r1.values, SEED + 100 * f + d, k=5, repeats=2); spec = decode(cands[c]); out = {'disease': DIS[d], 'candidate': c, 'complexity': spec['complexity']}
    for kind in ('loc', 'row', 'full'):
        t1 = time.time(); out[f'F_{kind}'] = robust_fitness(spec, FB, pairs, df.block_r1.values, d, kind=kind); out[f'sec_{kind}'] = time.time() - t1
    return out
t0 = time.time()
FID = pd.DataFrame(Parallel(n_jobs=2)(delayed(fid_one)(d, c) for d in range(3) for c in range(30)))
print(f'fidelity check: {(time.time() - t0) / 60:.1f} min')
FID.to_csv(OUT / 'surrogate_fidelity_candidates.csv', index=False)
fr = []
for d in DIS:
    s = FID[FID.disease == d]
    fr.append({'disease': d, 'rho_location_level_vs_full': stats.spearmanr(s.F_loc, s.F_full).correlation, 'rho_row_level_vs_full': stats.spearmanr(s.F_row, s.F_full).correlation,
               'sec_per_evaluation_location': s.sec_loc.mean(), 'sec_per_evaluation_row': s.sec_row.mean(), 'sec_per_evaluation_full': s.sec_full.mean()})
FR = pd.DataFrame(fr); FR.to_csv(OUT / 'surrogate_fidelity_summary.csv', index=False)
med_loc = float(FR.rho_location_level_vs_full.median()); med_row = float(FR.rho_row_level_vs_full.median())
chosen = 'location-level ExtraTrees 100' if med_loc >= 0.7 else 'row-level ExtraTrees 100'
LOG = json.load(open(LOGF)); LOG['stage3_decisions'] = {'surrogate': chosen, 'median_rho_location_level': med_loc, 'median_rho_row_level': med_row,
                                                        'rule': 'median per-disease rho >= 0.7 -> location-level, else row-level'}
json.dump(LOG, open(LOGF, 'w'), indent=1, default=str); json.dump(LOG, open(OUT / 'decision_log_snapshot.json', 'w'), indent=1, default=str)

# ---------- 4. figures ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
short = ['FS1 original', 'FS2 +climate', 'FS3 +coords', 'FS4 +nbr rates', 'FS5 +cross-dis.', 'FS6 +IDW', 'FS7 MI top 50%']
for ax, d in zip(axs, DIS):
    s = S[S.disease == d].set_index('feature_set').reindex(order)
    ax.bar(range(len(order)), s.ROC_AUC.values, color=['#c3c2b7', '#86b6ef', '#86b6ef', '#2a78d6', '#2a78d6', '#2a78d6', '#898781'], width=0.7)
    for i, v in enumerate(s.ROC_AUC.values): ax.text(i, v + 0.004, f'{v:.3f}', ha='center', fontsize=8)
    ax.axhline(s.run3_ExtraTrees_ELK_reference.iloc[0], color='#eb6834', ls='--', lw=1.2, label=f'Run 3 ExtraTrees [ELK] ({s.run3_ExtraTrees_ELK_reference.iloc[0]:.3f})')
    ax.set_xticks(range(len(order))); ax.set_xticklabels(short, rotation=35, ha='right', fontsize=8.5); ax.set_ylim(0.5, 0.75)
    ax.set_title(d, loc='left'); ax.legend(frameon=False, fontsize=8, loc='upper left')
axs[0].set_ylabel('Mean ROC-AUC, 15 development spatial folds')
fig.suptitle('Feature-set comparison with a fixed ExtraTrees-300 learner (development folds only)', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'feature_set_comparison_v1.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, d in zip(axs, DIS):
    s = FID[FID.disease == d]; r_ = FR.set_index('disease').loc[d]
    ax.scatter(s.F_full, s.F_loc, s=18, color='#2a78d6', label=f'location-level (rho {r_.rho_location_level_vs_full:.2f})')
    ax.scatter(s.F_full, s.F_row, s=18, color='#eb6834', marker='^', label=f'row-level (rho {r_.rho_row_level_vs_full:.2f})')
    lo_, hi_ = min(s[['F_full', 'F_loc', 'F_row']].min()), max(s[['F_full', 'F_loc', 'F_row']].max()); ax.plot([lo_, hi_], [lo_, hi_], color='#c3c2b7', lw=1)
    ax.set_title(d, loc='left'); ax.set_xlabel('Fitness with full ExtraTrees-300'); ax.legend(frameon=False, fontsize=8, loc='upper left')
axs[0].set_ylabel('Fitness with search surrogate'); fig.suptitle('Surrogate fidelity: 30 random SR-SGO candidates per disease (inner spatial CV only)', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'surrogate_fidelity_v1.png', dpi=200, bbox_inches='tight')

pd.set_option('display.width', 230)
print(S[['disease', 'feature_set', 'n_features', 'ROC_AUC', 'PR_AUC', 'dROC_vs_previous_set', 'p_corrected_vs_previous', 'dROC_vs_FS1', 'run3_ExtraTrees_ELK_reference']].round(4).to_string(index=False))
print(FR.round(3).to_string(index=False)); print('Surrogate chosen:', chosen)
