"""One optimiser run on the real SR-SGO search problem (development repeat 1). Loads data lazily once per worker process.
Writes a JSON checkpoint; the caller skips runs whose checkpoint exists."""
import json, time, warnings
from pathlib import Path
import numpy as np

_S = {}
SEED = 20261119; OPT_ORDER = ['SGO', 'MSGO', 'PSO', 'GA', 'DE', 'GWO', 'RANDOM', 'HGWOPSO']


def _setup(root):
    if 'FB' not in _S:
        from common.data import load, labels
        from common.features import climate_coords, Neighbourhood
        from common.fitness import FeatureBuilder
        df = load(Path(root) / 'data/raw'); Y = labels(df)
        _S.update(df=df, Y=Y, FB=FeatureBuilder(df, Y, Neighbourhood(df, Y), climate_coords(df)))
    return _S


def run_one(root, out_dir, opt_name, d, f, seed, pop=10, max_evals=150, kind='row'):
    warnings.filterwarnings('ignore')
    out = Path(out_dir) / f'{opt_name}_d{d}_f{f}_s{seed}.json'
    if out.exists(): return str(out)
    from common.optimisers import OPTIMISERS
    from common.space import decode, DIM
    from common.fitness import robust_fitness, fit_predict, auc_avg
    from common.folds import inner_spatial_folds
    from sklearn.metrics import average_precision_score
    S = _setup(root); df, Y, FB = S['df'], S['Y'], S['FB']
    blk = df.block_r1.values; te = np.where(df.spatial_fold_r1.values == f)[0]; tr = np.where(df.spatial_fold_r1.values != f)[0]
    pairs = inner_spatial_folds(tr, blk, SEED + 100 * f + d, k=5, repeats=1)
    cache = {}
    def fitness(X):
        vals = []
        for x in X:
            key = np.round(x, 12).tobytes()
            if key not in cache: cache[key] = robust_fitness(decode(x), FB, pairs, blk, d, kind=kind)
            vals.append(cache[key])
        return np.array(vals)
    X0 = np.random.default_rng(SEED + 1000 * seed + 10 * f + d).uniform(-1, 1, (pop, DIM))          # same start for every optimiser
    rng = np.random.default_rng(SEED + 7 * seed + 13 * f + 17 * d + 101 * OPT_ORDER.index(opt_name))
    t0 = time.time(); bx, bf, hist = OPTIMISERS[opt_name](fitness, X0, max_evals, rng); secs = time.time() - t0
    spec = decode(bx); Xf, Xp, names = FB.build(spec, tr, te, blk, d)
    p = fit_predict('full', spec, Xf, Y[tr, d], Xp, FB.loc[tr])
    rec = {'optimiser': opt_name, 'disease': ['CBPP', 'FMD', 'LSD'][d], 'fold': f, 'seed': seed, 'best_inner_fitness': bf, 'evaluations': len(hist),
           'unique_candidates': len(cache), 'seconds': secs, 'outer_ROC_AUC': auc_avg(Y[te, d], p), 'outer_PR_AUC': average_precision_score(Y[te, d], p),
           'n_features': Xf.shape[1], 'best_x': [float(v) for v in bx], 'spec': {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
           'history': [float(v) for v in hist]}
    json.dump(rec, open(out, 'w'))
    return str(out)
