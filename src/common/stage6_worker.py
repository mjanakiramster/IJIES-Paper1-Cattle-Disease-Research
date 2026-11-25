"""One SR-SGO run on one development outer fold and disease (Stage 6). Checkpointed: JSON record + NPZ predictions."""
import json, time, warnings
from pathlib import Path
import numpy as np

_S = {}; SEED = 20261119; DIS = ['CBPP', 'FMD', 'LSD']


def _setup(root):
    if 'FB' not in _S:
        from common.data import load, labels
        from common.features import climate_coords, Neighbourhood
        from common.fitness import FeatureBuilder
        df = load(Path(root) / 'data/raw'); Y = labels(df)
        _S.update(df=df, Y=Y, FB=FeatureBuilder(df, Y, Neighbourhood(df, Y), climate_coords(df)))
    return _S


def _key(spec):
    return (spec['kernel_rates'], spec['knn_rates'], spec['idw_rates'], spec['density_terms'], spec['cross_disease_rates'], spec['bandwidths'], spec['ks'],
            round(spec['prior'], 2), round(spec['idw_power'], 2) if spec['idw_rates'] else None, spec['min_samples_leaf'], round(spec['max_features'], 2))


def run_one(root, out_dir, d, rep, f, variant, pop=12, max_evals=240, top_k=5, kind='row'):
    warnings.filterwarnings('ignore')
    out = Path(out_dir) / f'd{d}_r{rep}_f{f}.json'
    if out.exists(): return str(out)
    from common.sr_sgo import sr_sgo
    from common.space_v2 import decode, anchor_fs5, DIM
    from common.fitness import robust_fitness, fit_predict, auc_avg
    from common.folds import inner_spatial_folds
    S = _setup(root); df, Y, FB = S['df'], S['Y'], S['FB']
    blk = df[f'block_r{rep}'].values; te = np.where(df[f'spatial_fold_r{rep}'].values == f)[0]; tr = np.where(df[f'spatial_fold_r{rep}'].values != f)[0]
    pairs = inner_spatial_folds(tr, blk, SEED + 1000 * rep + 100 * f + d, k=5, repeats=2)
    cache = {}                                                         # spec key -> (fitness, spec)
    def fitness(X):
        vals = []
        for x in X:
            s = decode(x); k = _key(s)
            if k not in cache: cache[k] = (robust_fitness(s, FB, pairs, blk, d, kind=kind), s)
            vals.append(cache[k][0])
        return np.array(vals)
    X0 = np.random.default_rng(SEED + 5000 + 1000 * rep + 100 * f + d).uniform(-1, 1, (pop, DIM)); X0[0] = anchor_fs5()
    rng = np.random.default_rng(SEED + 9000 + 1000 * rep + 100 * f + d); t0 = time.time()
    if variant == 'original':                       # original SGO equations (Satapathy & Naik 2016)
        from common.optimisers import sgo
        bx, bf, hist = sgo(fitness, X0, max_evals, rng)
    else:
        bx, bf, hist = sr_sgo(fitness, X0, max_evals, rng, variant=variant)
    secs = time.time() - t0
    top = sorted(cache.values(), key=lambda t: -t[0])[:top_k]
    preds = []
    for F, s in top:
        Xf, Xp, _ = FB.build(s, tr, te, blk, d); preds.append(fit_predict('full', s, Xf, Y[tr, d], Xp, FB.loc[tr]))
    P = np.array(preds); p_bag, p_best = P.mean(0), P[0]
    anchor_F = cache[_key(decode(anchor_fs5()))][0]
    rec = {'disease': DIS[d], 'rep': rep, 'fold': f, 'variant': variant, 'evaluations': len(hist), 'distinct_configurations': len(cache), 'seconds': secs,
           'best_inner_fitness': bf, 'anchor_inner_fitness': anchor_F, 'top_inner_fitness': [float(t[0]) for t in top],
           'outer_ROC_AUC_bagged_top5': auc_avg(Y[te, d], p_bag), 'outer_ROC_AUC_single_best': auc_avg(Y[te, d], p_best),
           'top_specs': [{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()} for _, s in top], 'history': [float(v) for v in hist]}
    np.savez_compressed(Path(out_dir) / f'd{d}_r{rep}_f{f}_preds.npz', idx=te, p_bag=p_bag, p_best=p_best, P_top=P)
    json.dump(rec, open(out, 'w'))
    return str(out)
