"""Feature-leakage audit A6 + block-size sensitivity (2026-11-21). EXPLORATORY; development data only.
Question: does the FMD feature gain (FS5 - FS1) depend on training locations close to the test locations?
Conditions (FS1 and FS5, ExtraTrees-300 as in Stages 3/9; all 3 diseases):
  base81      : development partition (81 km blocks, 3 x 5) as used so far
  buffer B    : remove every training location within B = 25 / 50 / 100 km of ANY test location of the fold
  matched B   : remove the same number of training households, as whole randomly chosen locations that are > B km from
                every test location (separates 'nearby data' from 'less training data')
  blocks120/160: new label-free partitions with 120 / 160 km blocks, built with the original recipe (grid, balanced
                assignment by household count), 3 repeats x 5 folds; seed 20261121.
Neighbourhood features always exclude the query's own block and are built from the remaining training rows only.
Run from the project root: python src/audit_A6_proximity_blocksize_v1.py"""
import os, sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); os.environ['PYTHONPATH'] = str(SRC); ROOT = SRC.parent
warnings.filterwarnings('ignore')
OUT = ROOT / 'results/audit_v1/A6'; (OUT / 'parts').mkdir(parents=True, exist_ok=True)
SEED_A6 = 20261121; _S = {}


def setup():
    if not _S:
        from common.data import load, labels
        from common.features import Neighbourhood, climate_coords
        from common.fitness import FeatureBuilder
        df = load(ROOT / 'data/raw'); Y = labels(df); NB = Neighbourhood(df, Y)
        for B in (120.0, 160.0):                                   # label-free partitions, original recipe
            rng = np.random.default_rng(SEED_A6 + int(B))
            for r in (1, 2, 3):
                ox, oy = (0.0, 0.0) if r == 1 else rng.uniform(0, B, 2)
                b = (np.floor((df.x_km + ox) / B).astype(int).astype(str) + '_' + np.floor((df.y_km + oy) / B).astype(int).astype(str)).values
                sizes = pd.Series(b).value_counts().to_dict(); load_ = np.zeros(5); asg = {}
                for g in rng.permutation(np.array(list(sizes), dtype=object)):
                    f = int(np.argmin(load_)); asg[g] = f; load_[f] += sizes[g]
                df[f'block_b{int(B)}_r{r}'] = b; df[f'fold_b{int(B)}_r{r}'] = pd.Series(b).map(asg).values
        _S.update(df=df, Y=Y, NB=NB, FB=FeatureBuilder(df, Y, NB, climate_coords(df)))
    return _S


def specs():
    from common.data import ENV
    from common.features import CLIMATE_NAMES
    base = {'coordinates': False, 'kernel_rates': False, 'knn_rates': False, 'idw_rates': False, 'density_terms': False, 'cross_disease_rates': False,
            'bandwidths': (50.0, 150.0, 400.0), 'ks': (5, 15, 40), 'k': 15, 'prior': 1.0, 'idw_power': 2.0, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'complexity': 0}
    return {'FS1': {**base, 'climate': ENV}, 'FS5': {**base, 'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True,
                                                     'density_terms': True, 'cross_disease_rates': True}}


def job(cond, r, f):
    out = OUT / 'parts' / f'{cond}_r{r}_f{f}.csv'
    if out.exists(): return str(out)
    from common.data import SEED
    from common.fitness import auc_avg
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import average_precision_score
    S = setup(); df, Y, NB, FB = S['df'], S['Y'], S['NB'], S['FB']; loc = df['loc'].values
    if cond.startswith('blocks'):
        B = cond[6:]; fold = df[f'fold_b{B}_r{r}'].values; blk = df[f'block_b{B}_r{r}'].values
    else:
        fold = df[f'spatial_fold_r{r}'].values; blk = df[f'block_r{r}'].values
    te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
    te_locs = np.unique(loc[te]); dmin = NB.D[:, te_locs].min(1)              # each location's distance to the nearest test location
    removed = 0
    if cond.startswith('buffer') or cond.startswith('matched'):
        Bkm = float(cond.split('_')[1]); near = dmin[loc[tr]] <= Bkm; removed = int(near.sum())
        if cond.startswith('buffer'): tr = tr[~near]
        else:
            rng = np.random.default_rng(SEED_A6 + 1000 * r + 10 * f + int(Bkm)); far_locs = rng.permutation(np.unique(loc[tr][~near])); drop, cnt = [], 0
            for l in far_locs:
                if cnt >= removed: break
                drop.append(l); cnt += int((loc[tr] == l).sum())
            tr = tr[~np.isin(loc[tr], drop)]; removed = cnt
    tr_locs = np.unique(loc[tr]); nearest = NB.D[np.ix_(te_locs, tr_locs)].min(1)
    rows = []
    for d, dn in enumerate(['CBPP', 'FMD', 'LSD']):
        rec = {'condition': cond, 'rep': r, 'fold': f, 'disease': dn, 'train_households': len(tr), 'train_locations': len(tr_locs), 'removed_households': removed,
               'train_positives': int(Y[tr, d].sum()), 'test_households': len(te), 'test_positives': int(Y[te, d].sum()),
               'median_nearest_training_location_km': float(np.median(nearest)), 'test_prevalence': Y[te, d].mean()}
        for name, spec in specs().items():
            Xf, Xp, _ = FB.build(spec, tr, te, blk, d)
            p = ExtraTreesClassifier(300, min_samples_leaf=10, max_features='sqrt', n_jobs=1, random_state=SEED).fit(Xf, Y[tr, d]).predict_proba(Xp)[:, 1]
            rec[f'{name}_ROC_AUC'] = auc_avg(Y[te, d], p); rec[f'{name}_PR_AUC'] = average_precision_score(Y[te, d], p)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out, index=False); return str(out)


if __name__ == '__main__':
    t0 = time.time()
    conds = ['base81'] + [f'{k}_{b}' for b in (25, 50, 100) for k in ('buffer', 'matched')] + ['blocks120', 'blocks160']
    jobs = [(c, r, f) for c in conds for r in (1, 2, 3) for f in range(5)]
    for k, _ in enumerate(Parallel(n_jobs=2, return_as='generator_unordered')(delayed(job)(*j) for j in jobs), 1):
        if k % 15 == 0: print(f'{k}/{len(jobs)} done, {(time.time() - t0) / 60:.1f} min', flush=True)
    A = pd.concat([pd.read_csv(p) for p in sorted((OUT / 'parts').glob('*.csv'))], ignore_index=True)
    A['gain_ROC_AUC'] = A.FS5_ROC_AUC - A.FS1_ROC_AUC; A['gain_PR_AUC'] = A.FS5_PR_AUC - A.FS1_PR_AUC; A.to_csv(OUT / 'A6_fold_results_v1.csv', index=False)
    S = A.groupby(['condition', 'disease']).agg(train_households=('train_households', 'mean'), train_locations=('train_locations', 'mean'), removed_households=('removed_households', 'mean'),
                                                train_positives=('train_positives', 'mean'), min_test_positives=('test_positives', 'min'),
                                                median_nearest_training_km=('median_nearest_training_location_km', 'mean'),
                                                FS1_ROC_AUC=('FS1_ROC_AUC', 'mean'), FS5_ROC_AUC=('FS5_ROC_AUC', 'mean'), gain_ROC_AUC=('gain_ROC_AUC', 'mean'),
                                                gain_SE=('gain_ROC_AUC', lambda s: s.std(ddof=1) / np.sqrt(len(s))), folds_gain_positive=('gain_ROC_AUC', lambda s: f'{(s > 0).sum()}/{len(s)}'),
                                                FS1_PR_AUC=('FS1_PR_AUC', 'mean'), FS5_PR_AUC=('FS5_PR_AUC', 'mean'), no_skill_PR_AUC=('test_prevalence', 'mean')).reset_index()
    order = {c: i for i, c in enumerate(conds)}; S = S.sort_values(['disease', 'condition'], key=lambda s: s.map(order) if s.name == 'condition' else s)
    S.to_csv(OUT / 'A6_summary_v1.csv', index=False)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, d in zip(axs, ['CBPP', 'FMD', 'LSD']):
        s = S[S.disease == d].set_index('condition').reindex(conds); x = np.arange(len(conds))
        col = ['#0d3f86' if c == 'base81' else '#2a78d6' if c.startswith('buffer') else '#a9a8a0' if c.startswith('matched') else '#1baf7a' for c in conds]
        ax.bar(x, s.gain_ROC_AUC, yerr=1.96 * s.gain_SE, color=col, width=0.7, capsize=3); ax.axhline(0, color='#555', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([c.replace('_', ' ') + ' km' if '_' in c else c for c in conds], rotation=45, ha='right', fontsize=8); ax.set_title(d, loc='left')
    axs[0].set_ylabel('FS5 - FS1 ROC-AUC (mean of 15 folds, 95% CI)')
    fig.suptitle('A6: engineered-feature gain when nearby training data are removed (blue), size-matched random removal (grey), larger blocks (green)', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'A6_proximity_blocksize_v1.png', dpi=200, bbox_inches='tight')
    pd.set_option('display.width', 250)
    print(S[['condition', 'disease', 'train_households', 'removed_households', 'median_nearest_training_km', 'FS1_ROC_AUC', 'FS5_ROC_AUC', 'gain_ROC_AUC', 'gain_SE', 'folds_gain_positive', 'min_test_positives']].round(3).to_string(index=False))
    print(f'A6 done in {(time.time() - t0) / 60:.1f} min')
