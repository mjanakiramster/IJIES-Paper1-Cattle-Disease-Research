"""Stage 4 — baselines: (1) verify run 3's 21-model library (cache + exact refit spot-check), (2) add LightGBM and SVM
with inner out-of-fold predictions on the same 26 splits and inner folds, (3) build comparators (a) simple rank average and
(b) inner-selected best single on the 23-model library, (c) hindsight-best single (labelled), (d) run-3 RB-HGWOPSO (reference).
Development, random and LORO splits only — the confirmatory partition is not touched.
Run from the project root:  python src/04_baseline_models.py"""
import sys, json, time, warnings, shutil
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
OUT = Path('results/stage4_v1'); (OUT / 'figures').mkdir(parents=True, exist_ok=False); (OUT / 'cache').mkdir()
t_start = time.time()

# ---------- 0. load run 3's frozen code (data, splits, feature sets, inner folds) into its own namespace ----------
src = open('data/reference/run3_cattle_v5_pipeline_v3.py').read()
src = src[:src.index("t0 = time.time()\nRES = sorted(")].replace("DATA = Path('data')", "DATA = Path('data/raw')").replace("OUT = next_free_dir('results')", "OUT = next_free_dir('_run3_setup_scratch')")
R3 = {'__name__': 'run3'}
exec(compile(src, 'run3_cattle_v5_pipeline_v3.py', 'exec'), R3)
shutil.rmtree(R3['OUT'], ignore_errors=True)
SPLITS, DIS, Y, SEED = R3['SPLITS'], R3['DIS'], R3['Y_ALL'], R3['SEED']
NAMES21 = list(R3['NAMES']); NEW = ['LightGBM [ELK]', 'SVM-RBF [ELK]']; NAMES = NAMES21 + NEW; C = len(NAMES)
LOCID = R3['df'].Location_id.values
cache = np.load('data/reference/run3_library_predictions_v1.npz')

# ---------- 1. verify the run-3 library ----------
ref = pd.read_csv('data/reference/run3_single_model_fold_metrics_v1.csv')
rows = []
for i, s in enumerate(SPLITS):
    P = cache[f'test_{i}']
    for di, d in enumerate(DIS):
        y = Y[s['test'], di]
        if len(np.unique(y)) < 2: continue
        for j, nm in enumerate(NAMES21): rows.append({'scheme': s['scheme'], 'rep': s['rep'], 'fold': str(s['fold']), 'disease': d, 'model': nm, 'ROC_AUC_from_cache': roc_auc_score(y, P[:, di, j])})
chk = pd.DataFrame(rows).merge(ref.assign(fold=ref.fold.astype(str))[['scheme', 'rep', 'fold', 'disease', 'model', 'ROC_AUC']], on=['scheme', 'rep', 'fold', 'disease', 'model'])
cache_check = {'fold_scores_compared': len(chk), 'max_abs_difference': float((chk.ROC_AUC_from_cache - chk.ROC_AUC).abs().max()),
               'within_1e-4': int(((chk.ROC_AUC_from_cache - chk.ROC_AUC).abs() < 1e-4).sum())}
i_spot = [k for k, s in enumerate(SPLITS) if s['scheme'] == 'spatial' and s['rep'] == 1 and s['fold'] == '0'][0]
t0 = time.time(); s = SPLITS[i_spot]; refit = R3['fit_predict_lib'](s['train'], s['test'], s['block'])
spot = {'split': f'spatial r1 fold 0 (index {i_spot})', 'models_refit': 21, 'max_abs_prediction_difference_vs_cache': float(np.abs(refit - cache[f'test_{i_spot}']).max()),
        'seconds': round(time.time() - t0, 1)}
json.dump({'cache_check': cache_check, 'exact_refit_spot_check': spot}, open(OUT / 'run3_reproduction_check.json', 'w'), indent=1)
print('cache check:', cache_check, '\nrefit spot check:', spot)

# ---------- 2. add LightGBM and SVM on ELK features, same splits and inner folds ----------
def lgbm(): return LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, verbose=-1, random_state=SEED)
def svm(): return make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale'))
CONFIG = {'LightGBM [ELK]': 'n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0',
          'SVM-RBF [ELK]': 'StandardScaler + SVC(C=1, gamma="scale"); ROC-AUC from decision function; probabilities by Platt scaling fitted on inner OOF decision values',
          'feature_set': 'ELK = run-3 E (climate + coordinates) + kernel rates 50/150/400 km + density + k-nearest rates (k = 5/15/40), from run-3 code',
          'inner_folds': 'identical to run 3 (cached fold arrays)'}
json.dump(CONFIG, open(OUT / 'new_model_configurations.json', 'w'), indent=1)

def new_preds(fit_idx, pred_idx, block):
    Xf, Xp = R3['featureset']('ELK', fit_idx, pred_idx, block); out = np.zeros((len(pred_idx), 3, 2))
    for di in range(3):
        y = Y[fit_idx, di]
        out[:, di, 0] = lgbm().fit(Xf, y).predict_proba(Xp)[:, 1]
        out[:, di, 1] = svm().fit(Xf, y).decision_function(Xp)
    return out

LIB = {}
t0 = time.time()
for i, s in enumerate(SPLITS):
    tr, te, fo = s['train'], s['test'], cache[f'fold_{i}']
    Pt_new = new_preds(tr, te, s['block']); Po_new = np.zeros((len(tr), 3, 2))
    for k in range(5):
        q = fo == k; Po_new[q] = new_preds(tr[~q], tr[q], s['block'])
    LIB[i] = {'test': np.concatenate([cache[f'test_{i}'], Pt_new], axis=2), 'oof': np.concatenate([cache[f'oof_spatial_{i}'], Po_new], axis=2), 'fold': fo}
print(f'LightGBM + SVM on 26 splits with inner OOF: {(time.time() - t0) / 60:.1f} min')
np.savez_compressed(OUT / 'cache' / 'library23_predictions_v1.npz', **{f'{k}_{i}': v[k].astype(np.float32) for i, v in LIB.items() for k in ('test', 'oof')},
                    **{f'fold_{i}': v['fold'] for i, v in LIB.items()})

# ---------- 3. metrics: single models, comparators ----------
def auc_avg(y, S):   # rows of S, average ranks for ties
    R = stats.rankdata(np.atleast_2d(S), axis=1); n1 = y.sum(); n0 = len(y) - n1
    return (R[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
def ecdf_scale(Po, Pt):
    n = len(Po); Ue = np.empty_like(Po); Ut = np.empty_like(Pt)
    for m in range(Po.shape[1]):
        srt = np.sort(Po[:, m]); Ue[:, m] = (stats.rankdata(Po[:, m]) - 0.5) / n
        Ut[:, m] = (np.searchsorted(srt, Pt[:, m], 'left') + np.searchsorted(srt, Pt[:, m], 'right')) / (2 * n)
    return Ue, Ut
def platt(oof_score, y, test_score):
    return LogisticRegression(C=1e6, max_iter=1000).fit(oof_score[:, None], y).predict_proba(test_score[:, None])[:, 1]
def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s); return thr[np.argmax(tpr - fpr)]
def thr_metrics(y, s, t):
    p = (s >= t).astype(int); tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1); f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    mcc = (tp * tn - fp * fn) / np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return {'threshold': t, 'accuracy': (tp + tn) / len(y), 'precision': prec, 'sensitivity': sens, 'specificity': spec, 'F1': f1, 'MCC': mcc,
            'balanced_accuracy': (sens + spec) / 2, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
ELIG = R3['elig']
def is_scored(scheme, fold, d): return scheme != 'loro' or ELIG[(fold, d)]   # LORO: only eligible region-disease pairs
SVM_J = NAMES.index('SVM-RBF [ELK]')

srow, crow, prow = [], [], []
for i, s in enumerate(SPLITS):
    tr, te = s['train'], s['test']; Pt, Po, fo = LIB[i]['test'], LIB[i]['oof'], LIB[i]['fold']
    for di, d in enumerate(DIS):
        y, yt = Y[tr, di], Y[te, di]; sc = is_scored(s['scheme'], s['fold'], d)
        po, pt = Po[:, di, :].copy(), Pt[:, di, :].copy()
        prob_t = pt.copy(); prob_t[:, SVM_J] = platt(po[:, SVM_J], y, pt[:, SVM_J])           # probabilities for Brier/calibration
        prob_o = po.copy(); prob_o[:, SVM_J] = 1 / (1 + np.exp(-po[:, SVM_J]))                   # (monotone; only ranks used from OOF)
        if len(np.unique(yt)) > 1:
            for j, nm in enumerate(NAMES):
                srow.append({'scheme': s['scheme'], 'rep': s['rep'], 'fold': str(s['fold']), 'disease': d, 'model': nm, 'scored': sc,
                             'ROC_AUC': roc_auc_score(yt, pt[:, j]), 'PR_AUC': average_precision_score(yt, pt[:, j]), 'Brier': brier_score_loss(yt, prob_t[:, j])})
        groups = [np.where(fo == k)[0] for k in range(5)]; groups = [g for g in groups if 0 < y[g].sum() < len(g)]
        inner = np.column_stack([auc_avg(y[g], po[g].T) for g in groups]).mean(1); jb = int(np.argmax(inner))
        Ue, Ut = ecdf_scale(po, pt); Ue21, Ut21 = ecdf_scale(po[:, :21], pt[:, :21])
        methods = {'(a) Simple rank average, 23 models': (Ut.mean(1), Ue.mean(1), None),
                   '(b) Best single, inner-selected (23 models)': (pt[:, jb], po[:, jb], prob_t[:, jb]),
                   'Simple probability average, 23 models': (prob_t.mean(1), None, prob_t.mean(1)),
                   'Simple rank average, run-3 21 models': (Ut21.mean(1), Ue21.mean(1), None)}
        for mname, (score_t, score_o, pr) in methods.items():
            rec = {'scheme': s['scheme'], 'rep': s['rep'], 'fold': str(s['fold']), 'disease': d, 'method': mname, 'scored': sc,
                   'selected_model': NAMES[jb] if mname.startswith('(b)') else ''}
            if len(np.unique(yt)) > 1:
                rec.update({'ROC_AUC': roc_auc_score(yt, score_t), 'PR_AUC': average_precision_score(yt, score_t), 'Brier': brier_score_loss(yt, pr) if pr is not None else np.nan})
                if score_o is not None: rec.update(thr_metrics(yt, score_t, youden(y, score_o)))
            crow.append(rec)
            if s['scheme'] == 'spatial': prow.append(pd.DataFrame({'rep': s['rep'], 'fold': s['fold'], 'idx': te, 'disease': d, 'method': mname, 'score': score_t, 'y': yt}))
        for j in (NAMES.index('ExtraTrees [ELK]'), NAMES.index('LightGBM [ELK]'), SVM_J, NAMES.index('CatBoost-village [EL]')):
            if s['scheme'] == 'spatial': prow.append(pd.DataFrame({'rep': s['rep'], 'fold': s['fold'], 'idx': te, 'disease': d, 'method': NAMES[j], 'score': pt[:, j], 'prob': prob_t[:, j], 'y': yt}))
SM = pd.DataFrame(srow); CM = pd.DataFrame(crow); PR = pd.concat(prow, ignore_index=True)
SM.to_csv(OUT / 'single_model_fold_metrics.csv', index=False); CM.to_csv(OUT / 'comparator_fold_metrics.csv', index=False)
PR.assign(Location_id=LOCID[PR.idx.values], ID=R3['df'].ID.values[PR.idx.values]).to_csv(OUT / 'development_test_predictions.csv.gz', index=False)

# single-model summary (23 models)
ss = SM[SM.scored].groupby(['scheme', 'disease', 'model']).ROC_AUC.mean().unstack(['scheme', 'disease'])
ss = ss[[(a, b) for a in ['spatial', 'random', 'loro'] for b in DIS]].reindex(NAMES); ss.round(4).to_csv(OUT / 'single_model_roc_auc_summary.csv')
best_h = SM[(SM.scheme == 'spatial')].groupby(['disease', 'model']).ROC_AUC.mean().groupby('disease').agg(['max', 'idxmax'])
best_h = pd.DataFrame({'disease': best_h.index, 'hindsight_best_model': best_h['idxmax'].map(lambda t: t[1]), 'ROC_AUC': best_h['max'].values})

# comparator summary with location-cluster bootstrap CI (development spatial folds, pooled per repeat)
def boot_ci(sub, B=1000, seed=SEED):
    piv = sub.pivot_table(index='idx', columns='rep', values='score'); yv = Y[piv.index.values, DIS.index(sub.disease.iloc[0])]
    lc = LOCID[piv.index.values]; groups = [np.where(lc == l)[0] for l in np.unique(lc)]; rg = np.random.default_rng(seed); vals = []
    pooled = lambda ix: np.mean([roc_auc_score(yv[ix], piv[c].values[ix]) for c in piv.columns])
    for _ in range(B): vals.append(pooled(np.concatenate([groups[k] for k in rg.integers(0, len(groups), len(groups))])))
    return pooled(np.arange(len(yv))), *np.percentile(vals, [2.5, 97.5])
cs = []
r3e = pd.read_csv('data/reference/run3_ensemble_fold_metrics_v1.csv')
for d in DIS:
    for m in CM.method.unique():
        sub = CM[(CM.scheme == 'spatial') & (CM.disease == d) & (CM.method == m)]
        pt_, lo_, hi_ = boot_ci(PR[(PR.disease == d) & (PR.method == m)])
        cs.append({'disease': d, 'method': m, 'ROC_AUC_mean_of_folds': sub.ROC_AUC.mean(), 'ROC_AUC_pooled': pt_, 'CI_lo': lo_, 'CI_hi': hi_, 'PR_AUC': sub.PR_AUC.mean(),
                   'Brier': sub.Brier.mean(), 'sensitivity': sub.sensitivity.mean(), 'specificity': sub.specificity.mean(), 'F1': sub.F1.mean(), 'MCC': sub.MCC.mean(),
                   'balanced_accuracy': sub.balanced_accuracy.mean(),
                   'random_CV_ROC_AUC': CM[(CM.scheme == 'random') & (CM.disease == d) & (CM.method == m)].ROC_AUC.mean(),
                   'LORO_ROC_AUC': CM[(CM.scheme == 'loro') & (CM.disease == d) & (CM.method == m) & CM.scored].ROC_AUC.mean()})
    h = best_h.set_index('disease').loc[d]
    cs.append({'disease': d, 'method': f'(c) Hindsight-best single: {h.hindsight_best_model}', 'ROC_AUC_mean_of_folds': h.ROC_AUC})
    r = r3e[(r3e.scheme == 'spatial') & (r3e.disease == d) & (r3e.method == 'RB-HGWOPSO ensemble (proposed)')]
    cs.append({'disease': d, 'method': '(d) RB-HGWOPSO, run 3 (21 models)', 'ROC_AUC_mean_of_folds': r.ROC_AUC.mean(), 'PR_AUC': r.PR_AUC.mean(), 'Brier': r.Brier.mean()})
CS = pd.DataFrame(cs); CS.to_csv(OUT / 'comparators_summary.csv', index=False)
sel = CM[(CM.scheme == 'spatial') & CM.method.str.startswith('(b)')].groupby(['disease', 'selected_model']).size().rename('folds_selected').reset_index()
sel.to_csv(OUT / 'inner_selected_models.csv', index=False)

# ---------- 4. figures ----------
COLS = {'(a) Simple rank average, 23 models': '#2a78d6', '(b) Best single, inner-selected (23 models)': '#eb6834', 'ExtraTrees [ELK]': '#0d3f86',
        'LightGBM [ELK]': '#1baf7a', 'SVM-RBF [ELK]': '#9b59b6'}
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
for c, d in enumerate(DIS):
    for m, col in COLS.items():
        sub = PR[(PR.disease == d) & (PR.method == m) & (PR.rep == 1)]
        fpr, tpr, _ = roc_curve(sub.y, sub.score); pr_, rc_, _ = precision_recall_curve(sub.y, sub.score)
        axs[0, c].plot(fpr, tpr, color=col, lw=1.6, label=f'{m.split(",")[0]} ({roc_auc_score(sub.y, sub.score):.3f})')
        axs[1, c].plot(rc_, pr_, color=col, lw=1.6, label=f'{m.split(",")[0]} ({average_precision_score(sub.y, sub.score):.3f})')
    axs[0, c].plot([0, 1], [0, 1], color='#c3c2b7', lw=1, ls='--'); axs[1, c].axhline(Y[:, c].mean(), color='#c3c2b7', lw=1, ls='--')
    axs[0, c].set_title(f'{d}: ROC (development repeat 1, pooled folds)', loc='left', fontsize=10); axs[1, c].set_title(f'{d}: precision-recall', loc='left', fontsize=10)
    axs[0, c].set_xlabel('False positive rate'); axs[0, c].set_ylabel('True positive rate'); axs[1, c].set_xlabel('Recall'); axs[1, c].set_ylabel('Precision')
    axs[0, c].legend(frameon=False, fontsize=7.5, loc='lower right'); axs[1, c].legend(frameon=False, fontsize=7.5, loc='upper right')
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'roc_pr_curves_development_v1.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, d in zip(axs, DIS):
    for m in ['ExtraTrees [ELK]', 'LightGBM [ELK]', 'SVM-RBF [ELK]', 'CatBoost-village [EL]']:
        sub = PR[(PR.disease == d) & (PR.method == m) & (PR.rep == 1)]
        fy, fx = calibration_curve(sub.y, sub.prob, n_bins=8, strategy='quantile')
        ax.plot(fx, fy, marker='o', ms=3.5, lw=1.4, color=COLS.get(m, '#898781'), label=m)
    ax.plot([0, 0.7], [0, 0.7], color='#c3c2b7', ls='--', lw=1); ax.set_xlabel('Mean predicted probability'); ax.set_title(d, loc='left')
axs[0].set_ylabel('Observed share positive'); axs[2].legend(frameon=False, fontsize=8, loc='upper left')
fig.suptitle('Calibration on development repeat 1 (8 quantile bins)', x=0.01, ha='left'); fig.tight_layout(); fig.savefig(OUT / 'figures' / 'calibration_curves_development_v1.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(2, 3, figsize=(11, 6.6))
for r_, m in enumerate(['(a) Simple rank average, 23 models', '(b) Best single, inner-selected (23 models)']):
    for c, d in enumerate(DIS):
        sub = CM[(CM.scheme == 'spatial') & (CM.disease == d) & (CM.method == m) & (CM.rep == 1)]
        M = np.array([[sub.TN.sum(), sub.FP.sum()], [sub.FN.sum(), sub.TP.sum()]]); ax = axs[r_, c]
        ax.imshow(M, cmap='Blues'); ax.set_xticks([0, 1]); ax.set_xticklabels(['pred 0', 'pred 1']); ax.set_yticks([0, 1]); ax.set_yticklabels(['true 0', 'true 1'])
        for a in range(2):
            for b in range(2): ax.text(b, a, f'{int(M[a, b])}', ha='center', va='center', fontsize=10, color='white' if M[a, b] > M.max() / 2 else 'black')
        ax.set_title(f'{d} — {m.split(",")[0][:28]}', loc='left', fontsize=9)
fig.suptitle('Confusion matrices at the Youden threshold chosen on inner OOF predictions (development repeat 1)', x=0.01, ha='left', fontsize=10)
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'confusion_matrices_development_v1.png', dpi=200, bbox_inches='tight')

def fs_colour(n): return '#eb6834' if n in NEW else '#c3c2b7' if '[B]' in n else '#86b6ef' if '[E]' in n else '#2a78d6' if '[EL]' in n else '#0d3f86'
fig, axs = plt.subplots(1, 3, figsize=(15, 6.8), sharey=True)
for ax, d in zip(axs, DIS):
    v = ss[('spatial', d)]; ax.barh(range(C), v.values, color=[fs_colour(n) for n in NAMES], height=0.65)
    a_ = CS[(CS.disease == d) & CS.method.str.startswith('(a)')].ROC_AUC_mean_of_folds.iloc[0]; ax.axvline(a_, color='#55554f', ls='--', lw=1.2, label=f'(a) rank average ({a_:.3f})')
    ax.set_xlim(0.5, 0.75); ax.set_title(d, loc='left'); ax.set_xlabel('Mean ROC-AUC, 15 development spatial folds'); ax.legend(frameon=False, fontsize=8, loc='lower right')
axs[0].set_yticks(range(C)); axs[0].set_yticklabels(NAMES, fontsize=8.5); axs[0].invert_yaxis()
fig.suptitle('23-model library on development folds (orange = new in Stage 4)', x=0.01, ha='left'); fig.tight_layout(); fig.savefig(OUT / 'figures' / 'single_models_development_v1.png', dpi=200, bbox_inches='tight')

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(ss.round(3).tail(4).to_string())
print(CS[['disease', 'method', 'ROC_AUC_mean_of_folds', 'ROC_AUC_pooled', 'CI_lo', 'CI_hi', 'PR_AUC', 'sensitivity', 'specificity', 'MCC', 'random_CV_ROC_AUC', 'LORO_ROC_AUC']].round(3).to_string(index=False))
print(sel.sort_values(['disease', 'folds_selected'], ascending=[True, False]).to_string(index=False))
print(f'Stage 4 total: {(time.time() - t_start) / 60:.1f} min')
