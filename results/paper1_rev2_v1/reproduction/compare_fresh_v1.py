"""Phase 3: compare every output of the from-scratch run (/home/claude/fresh/proj/results) with the main project.
CSV/JSON/TXT: byte-identical, else numeric comparison (max absolute difference over numeric cells, string cells must match).
NPZ: every array compared (max absolute difference; NaNs must coincide). PKL: data frames compared like CSV.
Images and PDFs: only checked to exist (renderers embed timestamps). Writes compare_fresh_v1.csv and prints a summary.
Usage: python compare_fresh_v1.py [subdir ...]"""
import sys, json, hashlib, io
from pathlib import Path
import numpy as np, pandas as pd
FRESH = Path('/home/claude/fresh/proj/results'); MAIN = Path('/home/claude/cattle_sgo_research/results')
DIRS = sys.argv[1:] or ['paper1_v1', 'paper1_rev_v1', 'paper1_rev2_v1']
def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()
def num_diff(a, b):
    if a.shape != b.shape: return np.inf, 'shape differs'
    if list(a.columns) != list(b.columns): return np.inf, 'columns differ'
    mx = 0.0; note = ''
    for c in a.columns:
        x, y = a[c], b[c]
        xn, yn = pd.to_numeric(x, errors='coerce'), pd.to_numeric(y, errors='coerce')
        if xn.notna().sum() == x.notna().sum() and yn.notna().sum() == y.notna().sum() and x.notna().any():
            if (xn.isna() != yn.isna()).any(): return np.inf, f'NaN pattern differs in {c}'
            m = np.nanmax(np.abs(xn.values.astype(float) - yn.values.astype(float))) if xn.notna().any() else 0.0; mx = max(mx, m)
        elif not (x.astype(str).values == y.astype(str).values).all():
            k = int((x.astype(str).values != y.astype(str).values).sum()); note += f'{c}: {k} text cells differ; '; mx = max(mx, np.nan if np.isnan(mx) else mx)
    return mx, note.strip()
rows = []
for d in DIRS:
    for f in sorted((FRESH / d).rglob('*')):
        if f.is_dir(): continue
        rel = f.relative_to(FRESH); m = MAIN / rel; r = {'file': str(rel), 'type': f.suffix}
        if not m.exists(): r.update(result='only in fresh run'); rows.append(r); continue
        if f.suffix in ('.png', '.pdf', '.xlsx', '.ipynb'): r.update(result='exists (not compared)'); rows.append(r); continue
        if md5(f) == md5(m): r.update(result='identical', max_abs_diff=0.0); rows.append(r); continue
        try:
            if f.suffix == '.csv':
                mx, note = num_diff(pd.read_csv(f), pd.read_csv(m))
            elif f.suffix == '.npz':
                A, Bz = np.load(f, allow_pickle=True), np.load(m, allow_pickle=True); mx, note = 0.0, ''
                if set(A.files) != set(Bz.files): mx, note = np.inf, 'keys differ'
                for k in A.files:
                    if k not in Bz.files: continue
                    a, b = A[k], Bz[k]
                    if a.shape != b.shape: mx, note = np.inf, f'{k} shape'; break
                    if a.dtype.kind in 'fc':
                        if (np.isnan(a) != np.isnan(b)).any(): mx, note = np.inf, f'{k} NaN pattern'; break
                        mx = max(mx, float(np.nanmax(np.abs(a - b))) if a.size and np.isfinite(a).any() else 0.0)
                    elif a.dtype.kind in 'iub': mx = max(mx, float(np.abs(a.astype(float) - b.astype(float)).max()) if a.size else 0.0)
                    elif not (a == b).all(): note += f'{k} differs; '
            elif f.suffix == '.json':
                ja, jb = json.load(open(f)), json.load(open(m))
                fa, fb = pd.json_normalize(ja if isinstance(ja, dict) else {'x': ja}), pd.json_normalize(jb if isinstance(jb, dict) else {'x': jb})
                mx, note = (0.0, '') if ja == jb else (np.nan, 'content differs')
                if ja != jb and list(fa.columns) == list(fb.columns):
                    try: mx, note = num_diff(fa, fb)
                    except Exception: pass
            elif f.suffix == '.pkl':
                a, b = pd.read_pickle(f), pd.read_pickle(m); mx, note = 0.0, ''
                if isinstance(a, pd.DataFrame) and a.shape == b.shape:
                    for c in a.columns:
                        va, vb = a[c].values, b[c].values
                        for x, y in zip(va, vb):
                            if isinstance(x, np.ndarray):
                                if x.shape != np.shape(y): mx = np.inf; break
                                if x.dtype.kind in 'fc': mx = max(mx, float(np.nanmax(np.abs(x - y))) if x.size else 0.0)
                                elif not (x == y).all(): note = f'{c} differs'
                            elif isinstance(x, (float, int, np.floating, np.integer)) and not isinstance(x, bool):
                                mx = max(mx, abs(float(x) - float(y)) if np.isfinite(float(x)) or np.isfinite(float(y)) else 0.0)
                            elif x != y and not (pd.isna(x) and pd.isna(y)): note = f'{c} differs'
                else: mx, note = np.inf, 'shape differs'
            else:
                mx, note = np.nan, 'bytes differ'
            r.update(result='numerically equal' if (mx == 0 and not note) else ('within 1e-9' if mx <= 1e-9 and not note else 'DIFFERS'), max_abs_diff=mx, note=note)
        except Exception as e:
            r.update(result='compare error', note=repr(e)[:200])
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv('/home/claude/fresh/compare_fresh_v1.csv', index=False)
print(R.groupby('result').size().to_string())
print(R[R.result.isin(['DIFFERS', 'compare error'])].head(40).to_string())
