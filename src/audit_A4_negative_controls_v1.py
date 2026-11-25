"""Feature-leakage audit A4 (2026-11-21): permutation negative controls through the FULL pipeline.

For each permutation the labels are permuted FIRST; then neighbourhood/cross-disease features are rebuilt from the
permuted labels and the models are refitted and evaluated, exactly as with the real labels. Development repeat 1
(5 spatial folds, 81 km blocks); ExtraTrees-300 (min_samples_leaf 10, max_features sqrt), the Stage 3 / Stage 9 learner.
  FS1 = 5 original variables;  FS5 = climate indices + coordinates + neighbourhood + cross-disease rates + density terms.
Null schemes (100 permutations each):
  household: shuffle the 3,340 household label rows (joint 3-disease pattern kept, overall counts kept exactly);
  village:   permute whole villages between locations of similar size (size bins), so within-village clustering and
             the joint 3-disease pattern are kept but every link between a village's outcomes and its place/climate is
             broken. A donor of a different size is resampled to the recipient's size (counts preserved approximately).
Expected under a valid pipeline: ROC-AUC near 0.5, PR-AUC near prevalence, and FS5 - FS1 near 0 in both schemes.
perm = -1 is the observed (real-label) run on the same folds. Run from the project root."""
import os, sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); os.environ['PYTHONPATH'] = str(SRC)
warnings.filterwarnings('ignore')
ROOT = SRC.parent; OUT = ROOT / 'results/audit_v1/A4'; (OUT / 'parts').mkdir(parents=True, exist_ok=True)
N_PERM = 100; SEED0 = 20261121; SIZE_BINS = [0, 1, 2, 4, 6, 9, 14, 10_000]
_S = {}


def setup():
    if not _S:
        from common.data import load, labels
        df = load(ROOT / 'data/raw'); _S.update(df=df, Y=labels(df))
    return _S


def permute(Y, df, scheme, rng):
    if scheme == 'household': return Y[rng.permutation(len(Y))]
    loc = df['loc'].values; idx = {l: np.where(loc == l)[0] for l in np.unique(loc)}
    size = pd.Series({l: len(v) for l, v in idx.items()}); bins = pd.cut(size, SIZE_BINS)
    Yp = np.empty_like(Y)
    for _, members in size.groupby(bins, observed=True):
        rec = members.index.values; don = rng.permutation(rec)
        for r, d in zip(rec, don):
            src = rng.permutation(idx[d]); n = len(idx[r])
            take = src[:n] if len(src) >= n else np.concatenate([src, rng.choice(idx[d], n - len(src), replace=True)])
            Yp[idx[r]] = Y[take]
    return Yp


def one(scheme, perm):
    out = OUT / 'parts' / f'{scheme}_{perm:03d}.csv'
    if out.exists(): return str(out)
    from common.data import ENV, SEED
    from common.features import climate_coords, Neighbourhood, CLIMATE_NAMES
    from common.fitness import FeatureBuilder, auc_avg
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import average_precision_score
    S = setup(); df, Y = S['df'], S['Y']
    Yp = Y.copy() if perm < 0 else permute(Y, df, scheme, np.random.default_rng(SEED0 + (0 if scheme == 'household' else 50_000) + perm))
    FB = FeatureBuilder(df, Yp, Neighbourhood(df, Yp), climate_coords(df))
    base = {'coordinates': False, 'kernel_rates': False, 'knn_rates': False, 'idw_rates': False, 'density_terms': False, 'cross_disease_rates': False,
            'bandwidths': (50.0, 150.0, 400.0), 'ks': (5, 15, 40), 'k': 15, 'prior': 1.0, 'idw_power': 2.0, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'complexity': 0}
    FS = {'FS1': {**base, 'climate': ENV}, 'FS5': {**base, 'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True,
                                                     'density_terms': True, 'cross_disease_rates': True}}
    blk = df.block_r1.values; rows = []
    for f in range(5):
        te = np.where(df.spatial_fold_r1.values == f)[0]; tr = np.where(df.spatial_fold_r1.values != f)[0]
        for d, dn in enumerate(['CBPP', 'FMD', 'LSD']):
            r = {'scheme': scheme, 'perm': perm, 'fold': f, 'disease': dn, 'test_prevalence': Yp[te, d].mean()}
            for name, spec in FS.items():
                Xf, Xp, _ = FB.build(spec, tr, te, blk, d)
                p = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(Xf, Yp[tr, d]).predict_proba(Xp)[:, 1]
                r[f'{name}_ROC_AUC'] = auc_avg(Yp[te, d], p); r[f'{name}_PR_AUC'] = average_precision_score(Yp[te, d], p)
            rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False); return str(out)


if __name__ == '__main__':
    t0 = time.time()
    jobs = [('household', -1)] + [(s, p) for p in range(N_PERM) for s in ('household', 'village')]
    for k, _ in enumerate(Parallel(n_jobs=2, return_as='generator_unordered')(delayed(one)(s, p) for s, p in jobs), 1):
        if k % 10 == 0 or k == len(jobs): print(f'{k}/{len(jobs)} done, {(time.time() - t0) / 60:.1f} min', flush=True)
    A = pd.concat([pd.read_csv(p) for p in sorted((OUT / 'parts').glob('*.csv'))], ignore_index=True)
    A.to_csv(OUT / 'A4_all_runs_v1.csv', index=False)
    M = A.groupby(['scheme', 'perm', 'disease'])[['FS1_ROC_AUC', 'FS5_ROC_AUC', 'FS1_PR_AUC', 'FS5_PR_AUC', 'test_prevalence']].mean().reset_index()
    M['gain_ROC_AUC'] = M.FS5_ROC_AUC - M.FS1_ROC_AUC; M['gain_PR_AUC'] = M.FS5_PR_AUC - M.FS1_PR_AUC
    obs = M[M.perm == -1].set_index('disease'); rows = []
    for (s, d), g in M[M.perm >= 0].groupby(['scheme', 'disease']):
        o = obs.loc[d]; r = {'scheme': s, 'disease': d, 'permutations': len(g)}
        for c in ('FS1_ROC_AUC', 'FS5_ROC_AUC', 'gain_ROC_AUC', 'FS1_PR_AUC', 'FS5_PR_AUC', 'gain_PR_AUC'):
            r[f'{c}_null_mean'] = g[c].mean(); r[f'{c}_null_2.5%'] = g[c].quantile(0.025); r[f'{c}_null_97.5%'] = g[c].quantile(0.975)
            r[f'{c}_observed'] = o[c]; r[f'{c}_p_perm'] = (1 + (g[c] >= o[c]).sum()) / (1 + len(g))
        r['no_skill_PR_AUC (prevalence)'] = g.test_prevalence.mean(); rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(OUT / 'A4_null_summary_v1.csv', index=False)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for i, s in enumerate(('household', 'village')):
        for j, d in enumerate(['CBPP', 'FMD', 'LSD']):
            ax = axs[i, j]; g = M[(M.scheme == s) & (M.disease == d) & (M.perm >= 0)]
            ax.hist(g.FS5_ROC_AUC, bins=20, color='#2a78d6', alpha=0.85, label='FS5 under null')
            ax.hist(g.FS1_ROC_AUC, bins=20, color='#a9a8a0', alpha=0.7, label='FS1 under null')
            ax.axvline(obs.loc[d, 'FS5_ROC_AUC'], color='#0d3f86', lw=2, label=f'FS5 real labels ({obs.loc[d, "FS5_ROC_AUC"]:.3f})')
            ax.axvline(obs.loc[d, 'FS1_ROC_AUC'], color='#55554f', lw=2, ls='--', label=f'FS1 real labels ({obs.loc[d, "FS1_ROC_AUC"]:.3f})')
            ax.axvline(0.5, color='#e34948', lw=1, ls=':'); ax.set_title(f'{d}, {s}-level permutation', loc='left', fontsize=9.5)
            if i == 1: ax.set_xlabel('Mean spatial-fold ROC-AUC')
            if j == 0: ax.set_ylabel('Permutations')
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle('A4 negative controls: full pipeline rebuilt on permuted labels (100 per scheme), development repeat 1', x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(OUT / 'A4_negative_controls_v1.png', dpi=200, bbox_inches='tight')
    pd.set_option('display.width', 250)
    print(S[['scheme', 'disease', 'FS1_ROC_AUC_null_mean', 'FS5_ROC_AUC_null_mean', 'gain_ROC_AUC_null_mean', 'gain_ROC_AUC_null_2.5%', 'gain_ROC_AUC_null_97.5%',
             'gain_ROC_AUC_observed', 'gain_ROC_AUC_p_perm', 'FS5_PR_AUC_null_mean', 'no_skill_PR_AUC (prevalence)']].round(4).to_string(index=False))
    print(f'A4 done in {(time.time() - t0) / 60:.1f} min')
