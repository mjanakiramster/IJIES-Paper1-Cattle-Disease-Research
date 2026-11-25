"""Negative-control table with 1,000 permutations per scheme (rev5, approved 2026-11-22).

Same aggregation as analysis.table10 (fold means over development repeat 1, gain = FS5 - FS1 ROC-AUC, permutation
p = (1 + null gains >= observed) / (1 + permutations), Holm over the six tests), applied to the original 100
permutations (results/paper1_v1/E5_permutations) plus the 900 added ones (rev5_perm_extension_v1.E5X).
The original Table10_negative_controls_v1.csv is not touched.

table() computes the table; if the output file exists it must be identical (otherwise an error), so the notebook
re-checks the stored table instead of overwriting it. Run: PYTHONPATH=src python -m paper1.rev5_perm_tables_v1"""
import pandas as pd
from .rev5_perm_extension_v1 import load_all, holm, TABD

OUT = TABD / 'Table10_negative_controls_1000_v1.csv'
N_EXPECTED = 1000


def compute():
    A = load_all()
    M = A.groupby(['scheme', 'perm', 'disease'])[['FS1_ROC_AUC', 'FS5_ROC_AUC', 'FS1_PR_AUC', 'FS5_PR_AUC', 'test_prevalence']].mean().reset_index()
    M['gain'] = M.FS5_ROC_AUC - M.FS1_ROC_AUC; obs = M[M.perm == -1].set_index('disease'); rows = []
    for (s, dn), g in M[M.perm >= 0].groupby(['scheme', 'disease']):
        o = obs.loc[dn]; n = len(g); k = int((g.gain >= o.gain).sum()); g100 = g[g.perm < 100]; k100 = int((g100.gain >= o.gain).sum())
        rows.append({'Scheme': s, 'Disease': dn, 'Permutations': n,
                     'Null FS1 ROC-AUC mean [2.5-97.5%]': f'{g.FS1_ROC_AUC.mean():.3f} [{g.FS1_ROC_AUC.quantile(.025):.3f}, {g.FS1_ROC_AUC.quantile(.975):.3f}]',
                     'Null FS5 ROC-AUC mean [2.5-97.5%]': f'{g.FS5_ROC_AUC.mean():.3f} [{g.FS5_ROC_AUC.quantile(.025):.3f}, {g.FS5_ROC_AUC.quantile(.975):.3f}]',
                     'Null gain mean [2.5-97.5%]': f'{g.gain.mean():+.3f} [{g.gain.quantile(.025):+.3f}, {g.gain.quantile(.975):+.3f}]',
                     'Observed gain': f'{o.gain:+.3f}', 'Permutation p (resolution)': f'{(1 + k) / (1 + n):.4f} ({1 / (1 + n):.4f})',
                     'Null FS5 PR-AUC': f'{g.FS5_PR_AUC.mean():.3f}', 'No-skill PR-AUC': f'{g.test_prevalence.mean():.3f}',
                     'Null gains >= observed': k, 'Max null gain': f'{g.gain.max():+.3f}', 'Null gain 99th percentile': f'{g.gain.quantile(.99):+.3f}',
                     'p with first 100 permutations': f'{(1 + k100) / (1 + len(g100)):.4f}', '_p': (1 + k) / (1 + n)})
    T = pd.DataFrame(rows); T['Holm p (6 tests)'] = [f'{v:.4f}' for v in holm(T._p.values)]
    return T.drop(columns='_p'), M


def table():
    T, M = compute()
    assert (T.Permutations == N_EXPECTED).all(), f'expected {N_EXPECTED} permutations per test, found {T.Permutations.tolist()}'
    if OUT.exists():
        old = pd.read_csv(OUT, dtype=str); new = T.astype(str)
        assert old.equals(new.reset_index(drop=True)), f'{OUT.name} differs from the recomputed table'
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True); T.to_csv(OUT, index=False)
    return T, M


if __name__ == '__main__':
    T, _ = table(); print(T.to_string())
