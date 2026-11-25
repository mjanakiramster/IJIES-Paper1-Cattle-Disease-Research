"""Permutation negative controls extended from 100 to 1,000 per scheme (approved by the author 2026-11-22).

New experiment: permutation indices 100-999 for the household-level and village-level schemes, development repeat 1,
computed with exactly the same procedure as experiments.e5_part (same label permutation, same seeds formula, same
feature construction and ExtraTrees). Results are written to a NEW folder, so the original 100 permutations
(results/paper1_v1/E5_permutations) and every table built from them stay unchanged.

  results/paper1_rev5_v1/E5_permutations_ext_v1/{scheme}_{perm:03d}.csv   (perm 100-999)
  results/paper1_rev5_v1/tables/Table10_negative_controls_1000_v1.csv     (all 1,000 per scheme + the observed run)

Usage:
  python -m paper1.rev5_perm_extension_v1 run household 100 999     # one worker per scheme
  python -m paper1.rev5_perm_extension_v1 check                      # perm 0 recomputed here must equal the original file
  python -m paper1.rev5_perm_extension_v1 table                      # build the 1,000-permutation table
Run from the project root with src on the path (PYTHONPATH=src)."""
import sys, time
import numpy as np, pandas as pd
from . import experiments as X
from .core import setup, dev_splits, extratrees, FEATURE_SETS, DIS, auc, pr_auc, ROOT

E5X = ROOT / 'results/paper1_rev5_v1/E5_permutations_ext_v1'
TABD = ROOT / 'results/paper1_rev5_v1/tables'


def part(scheme, perm, outdir=E5X):
    """Same computation as experiments.e5_part, written to outdir. Skips files that already exist (resumable)."""
    out = outdir / f'{scheme}_{perm:03d}.csv'
    if out.exists(): return str(out)
    df, Y, _, _ = X._env()
    Yp = Y.copy() if perm < 0 else X.permute_labels(Y, df, scheme, np.random.default_rng(20261121 + (0 if scheme == 'household' else 50_000) + perm))
    _, _, _, FB = setup(Y_override=Yp); rows = []
    for r, f, tr, te, bc in dev_splits(df, reps=(1,)):
        for d, dn in enumerate(DIS):
            rec = {'scheme': scheme, 'perm': perm, 'fold': f, 'disease': dn, 'test_prevalence': Yp[te, d].mean()}
            for name, spec in FEATURE_SETS.items():
                Xf, Xp, _ = FB.build(spec, tr, te, df[bc].values, d); p = extratrees().fit(Xf, Yp[tr, d]).predict_proba(Xp)[:, 1]
                rec[f'{name}_ROC_AUC'] = auc(Yp[te, d], p); rec[f'{name}_PR_AUC'] = pr_auc(Yp[te, d], p)
            rows.append(rec)
    outdir.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix('.tmp'); pd.DataFrame(rows).to_csv(tmp, index=False); tmp.rename(out)
    return str(out)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def load_all():
    files = sorted(X.E5.glob('*.csv')) + sorted(E5X.glob('*.csv'))
    A = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert not A.duplicated(['scheme', 'perm', 'fold', 'disease']).any(), 'duplicate permutation rows'
    return A


def table():
    """Same aggregation as analysis.table10, over all available permutations; also reports the 100-permutation subset."""
    A = load_all()
    M = A.groupby(['scheme', 'perm', 'disease'])[['FS1_ROC_AUC', 'FS5_ROC_AUC', 'FS1_PR_AUC', 'FS5_PR_AUC', 'test_prevalence']].mean().reset_index()
    M['gain'] = M.FS5_ROC_AUC - M.FS1_ROC_AUC; obs = M[M.perm == -1].set_index('disease'); rows = []
    for (s, dn), g in M[M.perm >= 0].groupby(['scheme', 'disease']):
        o = obs.loc[dn]; n = len(g); k = int((g.gain >= o.gain).sum()); g100 = g[g.perm < 100]; k100 = int((g100.gain >= o.gain).sum())
        rows.append({'Scheme': s, 'Disease': dn, 'Permutations': n,
                     'Null FS1 ROC-AUC mean [2.5-97.5%]': f'{g.FS1_ROC_AUC.mean():.3f} [{g.FS1_ROC_AUC.quantile(.025):.3f}, {g.FS1_ROC_AUC.quantile(.975):.3f}]',
                     'Null FS5 ROC-AUC mean [2.5-97.5%]': f'{g.FS5_ROC_AUC.mean():.3f} [{g.FS5_ROC_AUC.quantile(.025):.3f}, {g.FS5_ROC_AUC.quantile(.975):.3f}]',
                     'Null gain mean [2.5-97.5%]': f'{g.gain.mean():+.3f} [{g.gain.quantile(.025):+.3f}, {g.gain.quantile(.975):+.3f}]',
                     'Null gain 99th percentile': f'{g.gain.quantile(.99):+.3f}', 'Max null gain': f'{g.gain.max():+.3f}',
                     'Observed gain': f'{o.gain:+.3f}', 'Null gains >= observed': k,
                     'Permutation p (resolution)': f'{(1 + k) / (1 + n):.4f} ({1 / (1 + n):.4f})',
                     'p with first 100 permutations': f'{(1 + k100) / (1 + len(g100)):.4f}',
                     'Null FS5 PR-AUC': f'{g.FS5_PR_AUC.mean():.3f}', 'No-skill PR-AUC': f'{g.test_prevalence.mean():.3f}', '_p': (1 + k) / (1 + n)})
    T = pd.DataFrame(rows); T['Holm p (6 tests)'] = [f'{v:.4f}' for v in holm(T._p.values)]
    TABD.mkdir(parents=True, exist_ok=True); out = TABD / 'Table10_negative_controls_1000_v1.csv'
    assert not out.exists() or '--allow-rebuild' in sys.argv, f'{out} exists; refuse to overwrite'
    T.drop(columns='_p').to_csv(out, index=False); return T, M


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        scheme, a, b = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); t0 = time.time()
        for perm in range(a, b + 1):
            part(scheme, perm); print(f'{(time.time() - t0) / 60:7.1f} min  {scheme} {perm}', flush=True)
        print('done', flush=True)
    elif cmd == 'check':
        import tempfile, pathlib
        d = pathlib.Path(tempfile.mkdtemp()); t0 = time.time(); f = part('household', 0, d)
        new, old = pd.read_csv(f), pd.read_csv(X.E5 / 'household_000.csv')
        same = new.shape == old.shape and np.allclose(new.select_dtypes('number').values, old.select_dtypes('number').values, rtol=0, atol=0)
        print('perm 0 identical to original:', same, f'({time.time() - t0:.1f} s)')
    elif cmd == 'table':
        T, _ = table(); print(T.drop(columns='_p').to_string())
