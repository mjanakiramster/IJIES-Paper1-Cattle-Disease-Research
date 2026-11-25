"""Fold helpers shared by all stages."""
import numpy as np
import pandas as pd


def balanced_assign(groups, sizes, k, rng):
    """Blocks in random order; each goes to the fold with the fewest households so far (rule used since run 1)."""
    load_ = np.zeros(k); out = {}
    for g in rng.permutation(np.array(groups, dtype=object)):
        f = int(np.argmin(load_)); out[g] = f; load_[f] += sizes[g]
    return out


def inner_spatial_folds(train_idx, row_blocks, seed, k=5, repeats=2):
    """Repeated inner spatial-block CV inside an outer training set. Returns a list of (fit_idx, val_idx) pairs."""
    b = np.asarray(row_blocks).astype(str)[train_idx]; sizes = pd.Series(b).value_counts().to_dict(); pairs = []
    for r in range(repeats):
        asg = balanced_assign(list(sizes), sizes, k, np.random.default_rng(seed + 7919 * r))
        f = np.array([asg[x] for x in b])
        for j in range(k):
            pairs.append((train_idx[f != j], train_idx[f == j]))
    return pairs
