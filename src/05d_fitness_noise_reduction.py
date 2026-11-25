"""Stage 5d (exploratory, added 2026-11-21 after the confirmatory test) — can the NOISE in the search fitness be reduced?

Stage 5 showed that the optimisers' inner fitness does not predict outer test ROC-AUC (mean within-fold Spearman -0.03),
so better search cannot give better prediction. The dataset itself is clean (no duplicates, missing or impossible
values); the noise is (i) real within-location outcome variation and (ii) sampling noise of AUC on few spatial blocks.
This stage re-scores the same candidate configurations under alternative fitness definitions and asks which (if any)
ranks candidates in the same order as their outer test ROC-AUC.

Candidates per (disease, outer fold), development repeat 1 only:
  - the 21 best configurations found by the 7 Stage 5 optimisers (3 seeds each), outer AUC taken from Stage 5;
  - the fixed FS5 configuration;
  - 18 random configurations (wider range of quality), outer AUC computed here with the same final learner.
Fitness variants (all use inner spatial-block CV inside the outer training set; no outer test labels are used):
  V0  as Stage 5: 1 x 5 inner folds, row-level ExtraTrees-100, mean AUC - 0.5 SD - 0.005 complexity
  V1  3 x 5 inner folds, same formula                                  (more averaging)
  V2  3 x 5 inner folds, pooled out-of-fold AUC per repeat, averaged      (one AUC per repeat, less small-fold noise)
  V3  3 x 5 inner folds, pooled out-of-fold Brier skill score             (smooth proper score instead of a rank score)
  V4  3 x 5 inner folds, location-level targets (village infection rate), mean AUC - 0.5 SD - 0.005 complexity
Also: does training the FINAL model on location-level targets (label smoothing) improve outer AUC? (FS5, 3 x 5 dev folds)
Resumable: each (disease, fold) is checkpointed. Run from the project root: python src/05d_fitness_noise_reduction.py"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC)); os.environ['PYTHONPATH'] = str(SRC)
warnings.filterwarnings('ignore')
ROOT = SRC.parent; OUT = ROOT / 'results/stage5d_noise_v1'; (OUT / 'parts').mkdir(parents=True, exist_ok=True); (OUT / 'figures').mkdir(exist_ok=True)
LOG = open(ROOT / 'logs/stage5d_noise_v1.log', 'a')
def log(m): LOG.write(f'{time.strftime("%H:%M:%S")} {m}\n'); LOG.flush(); print(m, flush=True)
DIS3 = ['CBPP', 'FMD', 'LSD']; N_RANDOM = 18
VARIANTS = {'V0': 'Stage 5 fitness (1x5, mean-0.5SD AUC)', 'V1': '3x5 repeats, mean-0.5SD AUC', 'V2': '3x5, pooled AUC',
            'V3': '3x5, pooled Brier skill', 'V4': '3x5, village-level targets, mean-0.5SD AUC'}

_S = {}
def _setup():
    if 'FB' not in _S:
        from common.data import load, labels
        from common.features import climate_coords, Neighbourhood
        from common.fitness import FeatureBuilder
        df = load(ROOT / 'data/raw'); Y = labels(df)
        _S.update(df=df, Y=Y, FB=FeatureBuilder(df, Y, Neighbourhood(df, Y), climate_coords(df)))
    return _S


def fs5_spec():
    from common.features import CLIMATE_NAMES
    return {'climate': CLIMATE_NAMES, 'coordinates': True, 'kernel_rates': True, 'knn_rates': True, 'idw_rates': False, 'density_terms': True,
            'cross_disease_rates': True, 'bandwidths': (50.0, 150.0, 400.0), 'ks': (5, 15, 40), 'k': 15, 'prior': 1.0, 'idw_power': 2.0,
            'min_samples_leaf': 10, 'max_features': 'sqrt', 'complexity': 16 / 17}


def score_fold(di, f):
    out = OUT / 'parts' / f'd{di}_f{f}.csv'
    if out.exists(): return str(out)
    from common.data import SEED
    from common.space import decode, random_positions
    from common.fitness import fit_predict, auc_avg, LAMBDA_SD, MU_COMPLEXITY
    from common.folds import inner_spatial_folds
    S = _setup(); df, Y, FB = S['df'], S['Y'], S['FB']
    blk = df.block_r1.values; te = np.where(df.spatial_fold_r1.values == f)[0]; tr = np.where(df.spatial_fold_r1.values != f)[0]
    pairs = inner_spatial_folds(tr, blk, SEED + 100 * f + di, k=5, repeats=3)       # first 5 pairs == Stage 5 inner folds
    cands = []
    for p in sorted((ROOT / 'results/stage5_v1/runs').glob(f'*_d{di}_f{f}_s*.json')):
        r = json.load(open(p))
        cands.append({'source': 'optimiser best', 'label': f"{r['optimiser']} s{r['seed']}", 'spec': decode(np.array(r['best_x'])),
                      'outer_ROC_AUC': r['outer_ROC_AUC'], 'stage5_inner_fitness': r['best_inner_fitness']})
    cands.append({'source': 'FS5 fixed', 'label': 'FS5', 'spec': fs5_spec(), 'outer_ROC_AUC': None, 'stage5_inner_fitness': np.nan})
    for k, x in enumerate(random_positions(N_RANDOM, np.random.default_rng(SEED + 555 + 10 * f + di))):
        cands.append({'source': 'random', 'label': f'random {k}', 'spec': decode(x), 'outer_ROC_AUC': None, 'stage5_inner_fitness': np.nan})
    rows = []
    for c in cands:
        spec = c['spec']; y = Y[:, di]
        if c['outer_ROC_AUC'] is None:                                                  # same final learner as Stage 5
            Xf, Xp, _ = FB.build(spec, tr, te, blk, di); c['outer_ROC_AUC'] = auc_avg(y[te], fit_predict('full', spec, Xf, y[tr], Xp, FB.loc[tr]))
        a_row, a_loc = [], []; oof = {r: (np.zeros(len(df)), np.zeros(len(df), bool)) for r in range(3)}; brier_ref = {r: [] for r in range(3)}
        for j, (fit, val) in enumerate(pairs):
            Xf, Xp, _ = FB.build(spec, fit, val, blk, di)
            p = fit_predict('row', spec, Xf, y[fit], Xp, FB.loc[fit]); a_row.append(auc_avg(y[val], p))
            q = fit_predict('loc', spec, Xf, y[fit], Xp, FB.loc[fit]); a_loc.append(auc_avg(y[val], q))
            r = j // 5; oof[r][0][val] = p; oof[r][1][val] = True; brier_ref[r].append(np.full(len(val), y[fit].mean()))
        a_row, a_loc = np.array(a_row), np.array(a_loc); pen = MU_COMPLEXITY * spec['complexity']
        rob = lambda a: np.nanmean(a) - LAMBDA_SD * np.nanstd(a) - pen
        pooled_auc, bss = [], []
        for r in range(3):
            pr, m = oof[r]; idx = np.where(m)[0]; pooled_auc.append(auc_avg(y[idx], pr[idx]))
            ref = np.zeros(len(df)); ref[np.concatenate([v for _, v in pairs[5 * r:5 * r + 5]])] = np.concatenate(brier_ref[r])
            bss.append(1 - np.mean((pr[idx] - y[idx]) ** 2) / np.mean((ref[idx] - y[idx]) ** 2))
        rows.append({'disease': DIS3[di], 'fold': f, 'source': c['source'], 'label': c['label'], 'n_groups_on': round(spec['complexity'] * 17),
                     'outer_ROC_AUC': c['outer_ROC_AUC'], 'stage5_inner_fitness': c['stage5_inner_fitness'],
                     'V0': rob(a_row[:5]), 'V1': rob(a_row), 'V2': np.mean(pooled_auc) - pen, 'V3': np.mean(bss) - pen, 'V4': rob(a_loc),
                     'inner_auc_sd_1x5': np.nanstd(a_row[:5]), 'inner_auc_sd_3x5': np.nanstd(a_row)})
    pd.DataFrame(rows).to_csv(out, index=False); return str(out)


def final_label_smoothing(rep, f):
    from common.fitness import fit_predict, auc_avg
    S = _setup(); df, Y, FB = S['df'], S['Y'], S['FB']; spec = fs5_spec()
    blk = df[f'block_r{rep}'].values; fo = df[f'spatial_fold_r{rep}'].values; te = np.where(fo == f)[0]; tr = np.where(fo != f)[0]; res = []
    for di, d in enumerate(DIS3):
        Xf, Xp, _ = FB.build(spec, tr, te, blk, di); y = Y[:, di]
        res.append({'rep': rep, 'fold': f, 'disease': d,
                    'household_labels (ExtraTrees-300 classifier)': auc_avg(y[te], fit_predict('full', spec, Xf, y[tr], Xp, FB.loc[tr])),
                    'village infection rate (ExtraTrees-300 regressor)': auc_avg(y[te], fit_predict('loc', spec, Xf, y[tr], Xp, FB.loc[tr], n_trees=300))})
    return res


if __name__ == '__main__':
    t0 = time.time()
    jobs = [(di, f) for f in range(5) for di in range(3)]
    log(f'Stage 5d: {len(jobs)} (disease, fold) jobs, {sum(not (OUT / "parts" / f"d{d}_f{f}.csv").exists() for d, f in jobs)} to do')
    for k, p in enumerate(Parallel(n_jobs=2, return_as='generator_unordered')(delayed(score_fold)(di, f) for di, f in jobs), 1):
        log(f'  {k}/{len(jobs)} done ({(time.time() - t0) / 60:.1f} min): {Path(p).name}')
    C = pd.concat([pd.read_csv(p) for p in sorted((OUT / 'parts').glob('*.csv'))], ignore_index=True); C.to_csv(OUT / 'candidate_scores.csv', index=False)

    # sanity: V0 must reproduce the Stage 5 inner fitness of the optimisers' best configurations
    ob = C[C.source == 'optimiser best']; dev = np.abs(ob.V0 - ob.stage5_inner_fitness).max(); log(f'max |V0 - Stage 5 inner fitness| = {dev:.2e}')

    rows = []
    for v in VARIANTS:
        for subset, cond in (('all 40 candidates', C.source.notna()), ('21 optimiser best only', C.source == 'optimiser best')):
            for (d, f), g in C[cond].groupby(['disease', 'fold']):
                top = g.sort_values(v, ascending=False)
                rows.append({'variant': v, 'subset': subset, 'disease': d, 'fold': f, 'spearman': stats.spearmanr(g[v], g.outer_ROC_AUC).correlation,
                             'outer_AUC_of_top1': top.outer_ROC_AUC.iloc[0], 'outer_AUC_of_top5_mean': top.outer_ROC_AUC.iloc[:5].mean(),
                             'outer_AUC_mean_all': g.outer_ROC_AUC.mean(), 'outer_AUC_best_possible': g.outer_ROC_AUC.max(),
                             'FS5_outer_AUC': C[(C.disease == d) & (C.fold == f) & (C.source == 'FS5 fixed')].outer_ROC_AUC.iloc[0]})
    W = pd.DataFrame(rows); W.to_csv(OUT / 'fitness_variant_fold_results.csv', index=False)
    S = W.groupby(['subset', 'variant']).agg(mean_spearman=('spearman', 'mean'), folds_positive=('spearman', lambda s: f'{(s > 0).sum()}/{len(s)}'),
                                             outer_AUC_top1=('outer_AUC_of_top1', 'mean'), outer_AUC_top5=('outer_AUC_of_top5_mean', 'mean'),
                                             outer_AUC_mean_all=('outer_AUC_mean_all', 'mean'), FS5_outer_AUC=('FS5_outer_AUC', 'mean')).reset_index()
    # one-sample Wilcoxon of within-fold Spearman against 0 and paired against V0
    for i, r in S.iterrows():
        a = W[(W.subset == r.subset) & (W.variant == r.variant)].sort_values(['disease', 'fold']).spearman.values
        b = W[(W.subset == r.subset) & (W.variant == 'V0')].sort_values(['disease', 'fold']).spearman.values
        S.loc[i, 'p_spearman_gt_0'] = stats.wilcoxon(a, alternative='greater').pvalue
        S.loc[i, 'p_vs_V0'] = stats.wilcoxon(a - b).pvalue if r.variant != 'V0' else np.nan
    S['description'] = S.variant.map(VARIANTS); S.to_csv(OUT / 'fitness_variant_summary.csv', index=False)
    per_d = W[W.subset == 'all 40 candidates'].groupby(['variant', 'disease']).spearman.mean().unstack(); per_d.to_csv(OUT / 'spearman_by_disease.csv')
    sd = C.groupby('disease')[['inner_auc_sd_1x5', 'inner_auc_sd_3x5']].mean(); sd.to_csv(OUT / 'inner_fold_auc_sd.csv')

    L = pd.DataFrame([r for res in Parallel(n_jobs=2)(delayed(final_label_smoothing)(rep, f) for rep in (1, 2, 3) for f in range(5)) for r in res])
    L.to_csv(OUT / 'final_model_label_smoothing_folds.csv', index=False)
    a_, b_ = 'household_labels (ExtraTrees-300 classifier)', 'village infection rate (ExtraTrees-300 regressor)'
    LS = L.groupby('disease')[[a_, b_]].mean(); LS['difference'] = LS[b_] - LS[a_]
    LS['folds_better'] = L.assign(w=L[b_] > L[a_]).groupby('disease').w.sum().astype(str) + '/15'
    LS['p_wilcoxon'] = [stats.wilcoxon(L[L.disease == d][b_] - L[L.disease == d][a_]).pvalue for d in LS.index]
    LS.to_csv(OUT / 'final_model_label_smoothing_summary.csv')

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); sub = W[W.subset == 'all 40 candidates']
    data = [sub[sub.variant == v].spearman.values for v in VARIANTS]
    ax.boxplot(data, widths=0.5, medianprops={'color': '#0d3f86', 'lw': 2}); ax.axhline(0, color='#888', lw=1, ls='--')
    for i, a in enumerate(data, 1): ax.scatter(np.full(len(a), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(a)), a, s=14, color='#2a78d6', alpha=0.7, zorder=3)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels([f'{v}\n{VARIANTS[v]}'.replace(', ', ',\n', 1) for v in VARIANTS], fontsize=7.5)
    ax.set_ylabel('Spearman: inner fitness vs outer ROC-AUC\n(within each disease x fold, 40 candidates)')
    ax.set_title('Does a less noisy fitness rank configurations correctly? (above 0 = yes)', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'figures' / 'fitness_noise_reduction_v1.png', dpi=200, bbox_inches='tight')

    pd.set_option('display.width', 250)
    log('\n' + S.round(4).to_string(index=False)); log('\n' + per_d.round(3).to_string()); log('\n' + sd.round(4).to_string()); log('\n' + LS.round(4).to_string())
    log(f'Stage 5d done in {(time.time() - t0) / 60:.1f} min')
