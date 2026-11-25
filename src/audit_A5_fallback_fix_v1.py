"""Feature-leakage audit A5 (2026-11-21): correct the OLD run-3 fallback prior and refit every affected model.

Issue: in run-3 lag_features / knn_features, the prior rate p0 = mean of ALL fit rows. For a TRAINING query location this
includes its own spatial block (and its own household), unlike the stated rule. Test labels are never involved.
Correction (preserved alongside the original, which is not modified): p0(x) = rate of fit rows OUTSIDE block(x), exactly
as in the new code. Affected = every library model using feature sets EL or ELK (13 of the 21 run-3 models) plus the two
Stage 4 additions (LightGBM [ELK], SVM-RBF [ELK]); 8 models (feature sets B, E, G) are unaffected and are taken from the
cache. For the 15 development spatial splits, affected models are refitted for the outer test fold AND the 5 inner
spatial folds (needed for the ECDF rank scaling), with the cached inner fold assignment. Then the simple rank average (a)
and all single models are re-scored and compared with the cached (original) results. Confirmatory data are not used.
Step 0 re-creates one split's original predictions from code to confirm that the cache matches the code.
Run from the project root: python src/audit_A5_fallback_fix_v1.py"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); os.environ['PYTHONPATH'] = str(SRC); ROOT = SRC.parent
warnings.filterwarnings('ignore')
OUT = ROOT / 'results/audit_v1/A5'; (OUT / 'parts').mkdir(parents=True, exist_ok=True)
CACHE = ROOT / 'results/stage4_v1/cache/library23_predictions_v1.npz'
_S = {}


def lgbm(SEED):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1, verbose=-1, random_state=SEED)


def svm():
    from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.svm import SVC
    return make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale'))


def setup(corrected):
    key = 'corr' if corrected else 'orig'
    if key in _S: return _S[key]
    from audit_common_v1 import load_run3
    R = load_run3(scratch=f'_audit_scratch_A5_{key}')
    if corrected:
        g = R; LOC, NL, Y_ALL, KERN, BWS, DLOC, KS = g['LOC'], g['NL'], g['Y_ALL'], g['KERN'], g['BWS'], g['DLOC'], g['KS']
        def p0_excl(fit_idx, same):
            n = np.bincount(LOC[fit_idx], minlength=NL).astype(float)
            pos = np.column_stack([np.bincount(LOC[fit_idx], weights=Y_ALL[fit_idx, k], minlength=NL) for k in range(3)])
            S_ = same.astype(float); return n, pos, (pos.sum(0)[None, :] - S_ @ pos) / np.maximum(n.sum() - S_ @ n, 1.0)[:, None]
        def lag_features(fit_idx, pred_idx, block_col):                  # identical to run-3 except p0 (per query location, own block excluded)
            bl = g['BLOCK_LOC'][block_col]; same = bl[:, None] == bl[None, :]; n, pos, p0 = p0_excl(fit_idx, same); feats = []
            for b in BWS:
                Km = np.where(same, 0.0, KERN[b]); feats.append((Km @ pos + p0) / ((Km @ n)[:, None] + 1.0))
            dens = np.column_stack([np.log1p(np.where(same, 0.0, KERN[b]) @ n) for b in BWS]); L = np.column_stack(feats + [dens])
            return L[LOC[fit_idx]], L[LOC[pred_idx]]
        def knn_features(fit_idx, pred_idx, block_col):
            bl = g['BLOCK_LOC'][block_col]; same = bl[:, None] == bl[None, :]; n, pos, p0 = p0_excl(fit_idx, same)
            Dm = np.where(same | (n[None, :] == 0), np.inf, DLOC); order = np.argsort(Dm, axis=1, kind='stable'); navail = np.isfinite(Dm).sum(1); rr = np.arange(NL)
            cn = np.cumsum(n[order], axis=1); cp = np.cumsum(pos[order], axis=1); Ds = np.take_along_axis(Dm, order, axis=1); feats = []
            for k in KS:
                j = np.clip(np.minimum(k, navail) - 1, 0, NL - 1)
                feats.append(np.where((navail > 0)[:, None], (cp[rr, j] + p0) / (cn[rr, j][:, None] + 1.0), p0))
            j15 = np.clip(np.minimum(15, navail) - 1, 0, NL - 1); dist = np.where(navail > 0, Ds[rr, j15], DLOC.max())
            Kf = np.column_stack(feats + [np.log1p(dist)]); return Kf[LOC[fit_idx]], Kf[LOC[pred_idx]]
        R['lag_features'] = lag_features; R['knn_features'] = knn_features
    _S[key] = R; return R


def affected_idx(R): return [j for j, m in enumerate(R['LIB']) if m[2] in ('EL', 'ELK')]


def predict_affected(R, fit, pred, block):
    full = R['LIB']; aff = affected_idx(R); R['LIB'] = [full[j] for j in aff]; R['C'] = len(aff)
    try: P = R['fit_predict_lib'](fit, pred, block)
    finally: R['LIB'] = full; R['C'] = len(full)
    Xf, Xp = R['featureset']('ELK', fit, pred, block); ex = np.zeros((len(pred), 3, 2)); Y = R['Y_ALL']
    for d in range(3):
        ex[:, d, 0] = lgbm(R['SEED']).fit(Xf, Y[fit, d]).predict_proba(Xp)[:, 1]; ex[:, d, 1] = svm().fit(Xf, Y[fit, d]).decision_function(Xp)
    return np.concatenate([P, ex], axis=2)


def run_split(i):
    out = OUT / 'parts' / f'split_{i:02d}.npz'
    if out.exists(): return str(out)
    warnings.filterwarnings('ignore')
    R = setup(True); s = R['SPLITS'][i]; tr, te = s['train'], s['test']; fo = np.load(CACHE)[f'fold_{i}']
    Pt = predict_affected(R, tr, te, s['block']); Po = np.zeros((len(tr), 3, Pt.shape[2]))
    for k in range(5):
        q = fo == k; Po[q] = predict_affected(R, tr[~q], tr[q], s['block'])
    np.savez_compressed(out, test=Pt, oof=Po); return str(out)


def ecdf_rank_average(Po, Pt):
    n = len(Po); U = np.empty_like(Pt, dtype=float)
    for m in range(Po.shape[1]):
        srt = np.sort(Po[:, m]); U[:, m] = (np.searchsorted(srt, Pt[:, m], 'left') + np.searchsorted(srt, Pt[:, m], 'right')) / (2 * n)
    return U.mean(1)


if __name__ == '__main__':
    t0 = time.time(); Ro = setup(False); C = np.load(CACHE)
    spatial = [i for i, s in enumerate(Ro['SPLITS']) if s['scheme'] == 'spatial']; assert len(spatial) == 15
    aff = affected_idx(Ro); NAMES = list(Ro['NAMES']) + ['LightGBM [ELK]', 'SVM-RBF [ELK]']; AFF23 = aff + [21, 22]
    # step 0: cache reproduces from the ORIGINAL code (one split, outer test predictions of the affected models)
    i0 = spatial[0]; s0 = Ro['SPLITS'][i0]; P0 = predict_affected(Ro, s0['train'], s0['test'], s0['block'])
    repro = float(np.abs(P0 - C[f'test_{i0}'][:, :, AFF23]).max()); print(f'step 0: max |recomputed original - cache| = {repro:.2e}', flush=True)
    for k, _ in enumerate(Parallel(n_jobs=2, return_as='generator_unordered')(delayed(run_split)(i) for i in spatial), 1):
        print(f'{k}/15 splits refitted, {(time.time() - t0) / 60:.1f} min', flush=True)
    Y = Ro['Y_ALL']; rows, prow = [], []
    for i in spatial:
        s = Ro['SPLITS'][i]; tr, te = s['train'], s['test']; Z = np.load(OUT / 'parts' / f'split_{i:02d}.npz')
        Pt_o, Po_o = C[f'test_{i}'].astype(float), C[f'oof_{i}'].astype(float)
        Pt_c, Po_c = Pt_o.copy(), Po_o.copy(); Pt_c[:, :, AFF23] = Z['test']; Po_c[:, :, AFF23] = Z['oof']
        for d, dn in enumerate(Ro['DIS']):
            yt = Y[te, d]
            for lab, Pt, Po in (('original', Pt_o, Po_o), ('corrected', Pt_c, Po_c)):
                a = ecdf_rank_average(Po[:, d, :], Pt[:, d, :])
                rows.append({'split': i, 'rep': s['rep'], 'fold': s['fold'], 'disease': dn, 'version': lab, 'model': '(a) simple rank average',
                             'ROC_AUC': stats.mannwhitneyu(a[yt == 1], a[yt == 0]).statistic / (yt.sum() * (len(yt) - yt.sum())),
                             'PR_AUC': __import__('sklearn.metrics', fromlist=['x']).average_precision_score(yt, a)})
                for m in range(23):
                    sc = Pt[:, d, m]; rows.append({'split': i, 'rep': s['rep'], 'fold': s['fold'], 'disease': dn, 'version': lab, 'model': NAMES[m],
                                                   'ROC_AUC': stats.mannwhitneyu(sc[yt == 1], sc[yt == 0]).statistic / (yt.sum() * (len(yt) - yt.sum())), 'PR_AUC': np.nan})
            for m in AFF23:
                prow.append({'split': i, 'disease': dn, 'model': NAMES[m], 'spearman_original_vs_corrected_test_scores': stats.spearmanr(Pt_o[:, d, m], Pt_c[:, d, m]).correlation,
                             'max_abs_prediction_change': float(np.abs(Pt_o[:, d, m] - Pt_c[:, d, m]).max())})
    M = pd.DataFrame(rows); M.to_csv(OUT / 'A5_fold_metrics_original_vs_corrected_v1.csv', index=False)
    PC = pd.DataFrame(prow); PC.to_csv(OUT / 'A5_prediction_changes_v1.csv', index=False)
    W = M.pivot_table(index=['split', 'disease', 'model'], columns='version', values='ROC_AUC').reset_index(); W['diff'] = W.corrected - W.original
    A = W[W.model == '(a) simple rank average']; summ = []
    for d, g in A.groupby('disease'):
        dd = g['diff'].values; n1, n2 = 668 * 4, 668                       # corrected resampled t (Nadeau-Bengio), 15 = 3 x 5
        t = dd.mean() / np.sqrt((1 / 15 + n2 / n1) * dd.var(ddof=1)) if dd.var(ddof=1) > 0 else 0.0
        summ.append({'disease': d, '(a) original': g.original.mean(), '(a) corrected': g.corrected.mean(), 'difference': dd.mean(),
                     'max_fold_abs_difference': np.abs(dd).max(), 'folds_corrected_higher': f'{(dd > 0).sum()}/15', 'p_corrected_t': 2 * stats.t.sf(abs(t), 14)})
    S = pd.DataFrame(summ); S.to_csv(OUT / 'A5_rank_average_summary_v1.csv', index=False)
    R_ = W.groupby(['disease', 'model'])[['original', 'corrected']].mean().reset_index(); rk = []
    for d, g in R_.groupby('disease'):
        g = g[g.model != '(a) simple rank average']
        rk.append({'disease': d, 'kendall_tau_single_model_ranking': stats.kendalltau(g.original, g.corrected).correlation,
                   'best_single_original': g.loc[g.original.idxmax(), 'model'], 'best_single_corrected': g.loc[g.corrected.idxmax(), 'model'],
                   'max_single_model_mean_AUC_change': float((g.corrected - g.original).abs().max())})
    RK = pd.DataFrame(rk); RK.to_csv(OUT / 'A5_model_ranking_change_v1.csv', index=False)
    json.dump({'step0_cache_reproduction_max_abs_diff': repro, 'affected_models': [NAMES[m] for m in AFF23], 'unaffected_models': [NAMES[m] for m in range(23) if m not in AFF23],
               'minutes': (time.time() - t0) / 60}, open(OUT / 'A5_meta_v1.json', 'w'), indent=1)
    pd.set_option('display.width', 250); print(S.round(4).to_string(index=False)); print(RK.to_string(index=False))
    print(PC.groupby('model')[['spearman_original_vs_corrected_test_scores', 'max_abs_prediction_change']].agg(['min', 'max']).round(4).to_string())
    print(f'A5 done in {(time.time() - t0) / 60:.1f} min')
