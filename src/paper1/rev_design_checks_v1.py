"""Design checks for the IJIES Paper 1 revision (development data; the confirmatory fold file is read for its label-free
block structure only). Writes results/paper1_rev_v1/design_checks/*.
D1 block and fold structure; D2 inner folds; D3 test-label isolation of features, model, calibration and threshold;
D4 origin of the 81 km block size and label-free alternatives (variograms); D5 survey-campaign descriptors and variance shares.
Run from the project root: python src/paper1/rev_design_checks_v1.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import optimize
from paper1.core import setup, extratrees, FEATURE_SETS, DIS, SEED, platt, youden_threshold, auc
from common.folds import inner_spatial_folds
from common.data import BLOCK_KM
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'results' / 'paper1_rev_v1' / 'design_checks'; OUT.mkdir(parents=True, exist_ok=True)
df, Y, NB, FB = setup(); loc = df['loc'].values; res = {}

# ---------------- D1 block and fold structure
rows = []
for r in (1, 2, 3):
    blk = df[f'block_r{r}'].values; fo = df[f'spatial_fold_r{r}'].values
    one_block_per_loc = int(pd.Series(blk).groupby(loc).nunique().max()) == 1
    one_fold_per_block = int(pd.Series(fo).groupby(blk).nunique().max()) == 1
    ext = df.groupby(blk).agg(dx=('x_km', lambda v: v.max() - v.min()), dy=('y_km', lambda v: v.max() - v.min()))
    for f in range(5):
        te = fo == f; tl = np.unique(loc[te]); trl = np.unique(loc[~te])
        near = NB.D[np.ix_(tl, trl)].min(1)
        rows.append({'repeat': r, 'fold': f, 'test_households': int(te.sum()), 'test_locations': len(tl), 'test_blocks': len(np.unique(blk[te])),
                     'locations_in_both_train_and_test': len(np.intersect1d(tl, trl)), 'blocks_in_both': len(np.intersect1d(np.unique(blk[te]), np.unique(blk[~te]))),
                     'min_test_to_train_km': round(float(near.min()), 1), 'median_test_to_train_km': round(float(np.median(near)), 1),
                     'one_block_per_location': one_block_per_loc, 'one_fold_per_block': one_fold_per_block,
                     'max_block_extent_km': round(float(max(ext.dx.max(), ext.dy.max())), 1)})
D1 = pd.DataFrame(rows); D1.to_csv(OUT / 'D1_fold_structure_v1.csv', index=False)
res['D1'] = {'locations_shared_between_train_and_test': int(D1.locations_in_both_train_and_test.sum()), 'blocks_shared': int(D1.blocks_in_both.sum()),
             'all_blocks_within_block_size': bool((D1.max_block_extent_km < BLOCK_KM).all()), 'min_test_to_train_km': float(D1.min_test_to_train_km.min()),
             'median_test_to_train_km_range': [float(D1.median_test_to_train_km.min()), float(D1.median_test_to_train_km.max())],
             'fold_households_range': [int(D1.test_households.min()), int(D1.test_households.max())], 'block_km': BLOCK_KM}

# ---------------- D2 inner folds (as used for FS1/FS5 calibration and thresholds, E2)
bad = 0; n = 0
for r in (1, 2, 3):
    blk = df[f'block_r{r}'].values; fo = df[f'spatial_fold_r{r}'].values
    for f in range(5):
        tr = np.where(fo != f)[0]
        for fit, val in inner_spatial_folds(tr, blk, SEED + 100 * f + r, k=5, repeats=1):
            n += 1; bad += len(np.intersect1d(np.unique(blk[fit]), np.unique(blk[val]))) + len(np.intersect1d(fit, np.where(fo == f)[0])) + len(np.intersect1d(val, np.where(fo == f)[0]))
res['D2'] = {'inner_splits_checked': n, 'shared_blocks_or_outer_test_rows': bad}

# ---------------- D3 test-label isolation (repeat 1, fold 0, FS5, all diseases): flip all outer-test labels and rerun
r, f = 1, 0; blk = df['block_r1'].values; fo = df['spatial_fold_r1'].values; tr = np.where(fo != f)[0]; te = np.where(fo == f)[0]
Yf = Y.copy(); Yf[te] = 1 - Yf[te]; _, _, _, FBf = setup(Y_override=Yf); Z = np.load(ROOT / 'results/paper1_v1/E2_featuresets/r1_f0.npz')
pairs = inner_spatial_folds(tr, blk, SEED + 100 * f + r, k=5, repeats=1); pos = {i: q for q, i in enumerate(tr)}; d3 = []
for d, dn in enumerate(DIS):
    spec = FEATURE_SETS['FS5']; Xf, Xp, _ = FBf.build(spec, tr, te, blk, d); p_test = extratrees().fit(Xf, Yf[tr, d]).predict_proba(Xp)[:, 1]
    oof = np.full(len(tr), np.nan)
    for fit, val in pairs:
        A, Bv, _ = FBf.build(spec, fit, val, blk, d); oof[[pos[i] for i in val]] = extratrees().fit(A, Yf[fit, d]).predict_proba(Bv)[:, 1]
    p_ref, o_ref = Z['test'][:, d, 1], Z['oof'][:, d, 1]
    cal_new = platt(oof, Y[tr, d], p_test); cal_ref = platt(o_ref, Y[tr, d], p_ref)
    thr_new = youden_threshold(Y[tr, d], platt(oof, Y[tr, d], oof)); thr_ref = youden_threshold(Y[tr, d], platt(o_ref, Y[tr, d], o_ref))
    d3.append({'disease': dn, 'max_diff_test_scores': float(np.abs(p_test - p_ref).max()), 'max_diff_inner_oof': float(np.abs(oof - o_ref).max()),
               'max_diff_calibrated': float(np.abs(cal_new - cal_ref).max()), 'threshold_diff': abs(thr_new - thr_ref)})
res['D3'] = d3

# ---------------- D4 variograms (method of the original run-1 notebook: location-level, 20 km bins to 600 km, exponential model)
L = df.groupby('loc').agg(lat=('Latitude', 'first'), lon=('Longitude', 'first'), n=('ID', 'size'), year=('Year', lambda v: v.mode().iloc[0])).sort_index()
iu = np.triu_indices(len(L), 1); dist = NB.D[iu]; bins = np.arange(0, 620, 20); hc = (bins[:-1] + bins[1:]) / 2; which = np.digitize(dist, bins) - 1
def fit_range(z, w):
    sq = 0.5 * (z[iu[0]] - z[iu[1]]) ** 2; wt = w[iu[0]] * w[iu[1]]
    gam = np.array([np.average(sq[which == b], weights=wt[which == b]) if (which == b).sum() > 30 else np.nan for b in range(len(hc))]); ok = ~np.isnan(gam)
    p, _ = optimize.curve_fit(lambda h, a0, s, a: a0 + s * (1 - np.exp(-3 * h / a)), hc[ok], gam[ok], p0=[gam[ok][0], gam[ok].max() - gam[ok][0], 100], bounds=([0, 0, 5], [10, 10, 2000]), maxfev=20000)
    return float(p[2]), bool(p[2] > 600)
vrows = []
for d, dn in enumerate(DIS):
    z = pd.Series(Y[:, d]).groupby(loc).mean().sort_index().values; rng_, cap = fit_range(z, L.n.values)
    vrows.append({'variable': f'{dn} prevalence (all years)', 'uses_outcome_labels': True, 'practical_range_km': rng_, 'beyond_600km_fit_window': cap})
    yr_mean = pd.Series(Y[:, d]).groupby(df.Year.values).transform('mean').values
    z2 = pd.Series(Y[:, d] - yr_mean).groupby(loc).mean().sort_index().values; rng2, cap2 = fit_range(z2, L.n.values)
    vrows.append({'variable': f'{dn} prevalence minus survey-year mean', 'uses_outcome_labels': True, 'practical_range_km': rng2, 'beyond_600km_fit_window': cap2})
for v in ['Altitude', 'Temperature', 'Precipitation', 'Relative_humidity', 'Wind_speed']:
    z = df.groupby('loc')[v].mean().sort_index().values; z = (z - z.mean()) / z.std(); rng_, cap = fit_range(z, np.ones(len(z)))
    vrows.append({'variable': f'{v} (standardised)', 'uses_outcome_labels': False, 'practical_range_km': rng_, 'beyond_600km_fit_window': cap})
D4 = pd.DataFrame(vrows); D4.to_csv(OUT / 'D4_variogram_ranges_v1.csv', index=False)
res['D4'] = {'block_km_used': BLOCK_KM, 'rule': 'median of the three outcome practical ranges (run 1); equals the FMD range',
             'label_free_median_range_km': float(D4[~D4.uses_outcome_labels].practical_range_km.median())}

# ---------------- D5 campaign descriptors
NBm = NB.D; crow = []
for t in sorted(df.Year.unique()):
    m = df.Year.values == t; tl = np.unique(loc[m]); others = np.unique(loc[~m])
    other_near = NBm[np.ix_(tl, others)].min(1) if len(others) else np.array([np.inf])
    rec = {'Survey year': int(t), 'Households': int(m.sum()), 'Locations': len(tl), 'Regions': ', '.join(sorted(df.Region[m].unique())),
           'Latitude range': f'{df.Latitude[m].min():.1f} to {df.Latitude[m].max():.1f}', 'Longitude range': f'{df.Longitude[m].min():.1f} to {df.Longitude[m].max():.1f}',
           'Locations with another campaign within 81 km (%)': round(100 * float((other_near <= BLOCK_KM).mean()), 1),
           'Median distance to nearest other-campaign location (km)': round(float(np.median(other_near)), 1)}
    for d, dn in enumerate(DIS):
        lp = pd.Series(Y[m, d]).groupby(loc[m]).agg(['mean', 'size']); lp = lp[lp['size'] >= 5]['mean']
        rec[f'{dn} prevalence (%)'] = round(100 * Y[m, d].mean(), 1)
        rec[f'{dn} location prevalence IQR (locations with >= 5 households)'] = f'{lp.quantile(0.25):.2f} to {lp.quantile(0.75):.2f}'
    crow.append(rec)
D5 = pd.DataFrame(crow); D5.to_csv(OUT / 'D5_campaign_descriptors_v1.csv', index=False)
# share of household-level outcome variance explained by survey year and by location (one-way, eta squared)
vs = []
for d, dn in enumerate(DIS):
    yv = Y[:, d].astype(float); tot = ((yv - yv.mean()) ** 2).sum()
    def eta(g):
        gm = pd.Series(yv).groupby(g).transform('mean').values; return float(((gm - yv.mean()) ** 2).sum() / tot)
    vs.append({'disease': dn, 'eta2_survey_year': eta(df.Year.values), 'eta2_location': eta(loc), 'eta2_region': eta(df.Region.values),
               'eta2_location_within_year': float(1 - ((yv - pd.Series(yv).groupby(loc).transform('mean').values) ** 2).sum() / ((yv - pd.Series(yv).groupby(df.Year.values).transform('mean').values) ** 2).sum())})
D5b = pd.DataFrame(vs); D5b.to_csv(OUT / 'D5_variance_shares_v1.csv', index=False)
res['D5'] = D5b.round(3).to_dict('records')
json.dump(res, open(OUT / 'design_checks_summary_v1.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)); print(D4.round(1).to_string()); print(D5.to_string())
