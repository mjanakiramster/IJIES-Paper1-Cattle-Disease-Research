"""Stage 2 — spatial validation engine, confirmatory partition (generated, hashed, NOT evaluated) and decision log.
Run from the project root:  python src/02_spatial_validation.py"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load, labels, file_hashes, DIS, BLOCK_KM, SEED
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

CFG = yaml.safe_load(open('configs/experiment_config.yaml'))
OUT = Path('results/stage2_v1'); (OUT / 'figures').mkdir(parents=True, exist_ok=False)
df = load('data/raw'); Y = labels(df); K = CFG['k_folds']


def balanced_assign(groups, sizes, k, rng):
    """Same rule as run 1: blocks in random order, each to the fold with the fewest households so far."""
    load_ = np.zeros(k); out = {}
    for g in rng.permutation(np.array(groups, dtype=object)):
        f = int(np.argmin(load_)); out[g] = f; load_[f] += sizes[g]
    return out


def grid_blocks(ox, oy, block_km=BLOCK_KM):
    return (np.floor((df.x_km + ox) / block_km).astype(int).astype(str) + '_' + np.floor((df.y_km + oy) / block_km).astype(int).astype(str)).values


# ---------- 1. development partition: verify it is exactly reproducible from its recipe ----------
DEV_SEED = 20261118   # seed used by run 1 when it created stage3_folds_v1.csv
rng = np.random.default_rng(DEV_SEED); replay = {}
for r in (1, 2, 3):
    ox, oy = (0.0, 0.0) if r == 1 else rng.uniform(0, BLOCK_KM, 2)
    b = grid_blocks(ox, oy); sizes = pd.Series(b).value_counts().to_dict()
    asg = balanced_assign(list(sizes), sizes, K, rng)
    replay[r] = {'offset_km': (float(ox), float(oy)), 'blocks_match': bool((b == df[f'block_r{r}'].to_numpy(dtype=str)).all()),
                 'folds_match': bool((pd.Series(b).map(asg).values == df[f'spatial_fold_r{r}'].values).all())}
print('Development partition replay:', replay)
dev = df[['ID', 'Location_id', 'Region'] + [c for r in (1, 2, 3) for c in (f'block_r{r}', f'spatial_fold_r{r}')] + ['random_fold']].copy()
dev.to_csv(OUT / 'spatial_folds.csv', index=False)

# ---------- 2. confirmatory partition (generated now; evaluated only in Stage 9) ----------
def torus_dist(a, b, B=BLOCK_KM):
    d = np.abs(np.asarray(a) - np.asarray(b)) % B; d = np.minimum(d, B - d); return float(np.hypot(*d))
taken = [replay[r]['offset_km'] for r in (1, 2, 3)]; cand = [(float(x), float(y)) for x in np.arange(0, BLOCK_KM, 1.0) for y in np.arange(0, BLOCK_KM, 1.0)]
CONF_OFFSETS = []
for _ in range(CFG['confirmatory']['repeats']):
    best = max(cand, key=lambda c: min(torus_dist(c, t) for t in taken)); CONF_OFFSETS.append(best); taken.append(best)
OFFSET_DIST = [{'confirmatory': f'c{a + 1}', 'offset_km': CONF_OFFSETS[a], **{f'distance_km_to_r{r}': torus_dist(CONF_OFFSETS[a], replay[r]['offset_km']) for r in (1, 2, 3)}}
               for a in range(len(CONF_OFFSETS))]
print('Confirmatory grid origins (km):', OFFSET_DIST)
cseed = CFG['confirmatory_seed']; attempts = []
while True:
    rngc = np.random.default_rng(cseed); conf = df[['ID', 'Location_id']].copy()
    for a, (ox_, oy_) in enumerate(CONF_OFFSETS, start=1):
        b = grid_blocks(ox_, oy_); sizes = pd.Series(b).value_counts().to_dict()
        asg = balanced_assign(list(sizes), sizes, K, rngc)
        conf[f'cblock_c{a}'] = b; conf[f'cfold_c{a}'] = pd.Series(b).map(asg).values
    # redraw rule (pre-registered): every fold needs >= 10 positives and >= 10 negatives per disease
    ok = all(10 <= Y[conf[f'cfold_c{a}'].values == f, di].sum() <= (conf[f'cfold_c{a}'].values == f).sum() - 10
             for a in (1, 2) for f in range(K) for di in range(3))
    attempts.append({'seed': cseed, 'passes_redraw_rule': ok})
    if ok: break
    cseed += 1
conf.to_csv(OUT / 'confirmatory_folds.csv', index=False)
conf_hash = file_hashes(OUT / 'confirmatory_folds.csv')['sha256']

# ---------- 3. blocks table, LORO, fold report ----------
blk = []
for name, cols in [('development', [(f'block_r{r}', f'spatial_fold_r{r}', f'r{r}') for r in (1, 2, 3)]),
                   ('confirmatory', [(f'cblock_c{a}', f'cfold_c{a}', f'c{a}') for a in (1, 2)])]:
    src = dev if name == 'development' else conf
    for bc, fc, tag in cols:
        g = pd.DataFrame({'block': src[bc].values, 'fold': src[fc].values, 'loc': df['loc'].values, 'x': df.Longitude.values, 'y': df.Latitude.values})
        t = g.groupby('block').agg(fold=('fold', 'first'), households=('loc', 'size'), locations=('loc', 'nunique'), lon=('x', 'mean'), lat=('y', 'mean')).reset_index()
        blk.append(t.assign(partition=name, repeat=tag))
pd.concat(blk).to_csv(OUT / 'spatial_blocks.csv', index=False)

elig = []
for reg in sorted(df.Region.unique()):
    m = df.Region.values == reg
    for di, d in enumerate(DIS):
        pos = int(Y[m, di].sum()); elig.append({'region': reg, 'disease': d, 'households': int(m.sum()), 'positives': pos,
                                                 'negatives': int(m.sum()) - pos, 'eligible_for_LORO': pos >= 10 and int(m.sum()) - pos >= 10})
LORO = pd.DataFrame(elig); LORO.to_csv(OUT / 'loro_eligibility.csv', index=False)

L = df.groupby('loc')[['Latitude', 'Longitude']].first().sort_index()
la, lo = np.radians(L.Latitude.values), np.radians(L.Longitude.values)
h = np.sin((la[None] - la[:, None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[None] - lo[:, None]) / 2) ** 2
DL = 2 * 6371 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
rep_rows = []
def fold_rows(partition, tag, fold_vec):
    for f in range(K):
        te = fold_vec == f; tl = np.unique(df['loc'].values[te]); trl = np.unique(df['loc'].values[~te])
        rep_rows.append({'partition': partition, 'repeat': tag, 'fold': f, 'households': int(te.sum()), 'locations': len(tl),
                         'locations_also_in_training': len(np.intersect1d(tl, trl)),
                         'median_km_test_location_to_nearest_training_location': float(np.median(DL[np.ix_(tl, trl)].min(1))),
                         **{f'positives_{d}': int(Y[te, di].sum()) for di, d in enumerate(DIS)}})
for r in (1, 2, 3): fold_rows('development', f'r{r}', dev[f'spatial_fold_r{r}'].values)
for a in (1, 2): fold_rows('confirmatory', f'c{a}', conf[f'cfold_c{a}'].values)
fold_rows('random (reference)', 'r1', dev.random_fold.values)
REP = pd.DataFrame(rep_rows); REP.to_csv(OUT / 'fold_validation_report.csv', index=False)

# how different is the confirmatory partition from the development one? (location level, label-free)
locf = lambda v: pd.Series(v).groupby(df['loc'].values).first().values
diff = [{'confirmatory': f'c{a}', 'development': f'r{r}',
         'adjusted_rand_index_of_blocks': adjusted_rand_score(locf(conf[f'cblock_c{a}'].values), locf(dev[f'block_r{r}'].values)),
         'adjusted_rand_index_of_folds': adjusted_rand_score(locf(conf[f'cfold_c{a}'].values), locf(dev[f'spatial_fold_r{r}'].values))}
        for a in (1, 2) for r in (1, 2, 3)]
DIFF = pd.DataFrame(diff); DIFF.to_csv(OUT / 'confirmatory_vs_development_overlap.csv', index=False)

# ---------- 4. decision log (pre-registration) ----------
DECISION = {'written_in': 'Stage 2, before any new model was fitted', 'config': CFG,
            'development_partition': {'file': 'data/raw/stage3_folds_v1.csv', 'sha256': file_hashes('data/raw/stage3_folds_v1.csv')['sha256'], 'replay_check': replay},
            'confirmatory_partition': {'file': str(OUT / 'confirmatory_folds.csv'), 'sha256': conf_hash, 'seed_used': cseed, 'redraw_attempts': attempts,
                                       'offsets_km': CONF_OFFSETS, 'offset_distances': OFFSET_DIST,
                                       'status': 'SEALED - no model may be evaluated on it before Stage 9'},
            'loro_regions_eligible': LORO[LORO.eligible_for_LORO].groupby('region').disease.apply(list).to_dict(),
            'excluded_variables': ['Weight', 'Year_region', 'Year', 'Region'],
            'deviations': [{'stage': 2, 'what': 'confirmatory grid origins changed from fixed fractions to the maximin rule',
                            'why': 'c2 nearly coincided with development repeat 3 (block ARI 0.77)', 'when': 'before any model evaluation; no labels used except the pre-registered class-count check'}]}
json.dump(DECISION, open('configs/decision_log.json', 'w'), indent=1, default=str)
json.dump(DECISION, open(OUT / 'decision_log_snapshot.json', 'w'), indent=1, default=str)

# ---------- 5. figure: development repeat 1 vs confirmatory c1 ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
pal = ['#2a78d6', '#eb6834', '#1baf7a', '#9b59b6', '#c3a000']
for ax, (title, v) in zip(axs, [('Development repeat 1 (runs 1-3)', dev.spatial_fold_r1.values), ('Confirmatory c1 (sealed)', conf.cfold_c1.values),
                                 ('Confirmatory c2 (sealed)', conf.cfold_c2.values)]):
    Lf = pd.DataFrame({'lon': df.Longitude, 'lat': df.Latitude, 'f': v}).groupby(['lon', 'lat']).f.first().reset_index()
    for f in range(K): ax.scatter(Lf.lon[Lf.f == f], Lf.lat[Lf.f == f], s=12, color=pal[f], label=f'Fold {f + 1}')
    ax.set_title(title, loc='left', fontsize=10); ax.set_xlabel('Longitude')
axs[0].set_ylabel('Latitude'); axs[2].legend(frameon=False, fontsize=8, loc='lower right')
fig.suptitle(f'Spatial-block folds ({BLOCK_KM:.0f} km blocks); colours show which locations are held out together', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'development_vs_confirmatory_folds_v1.png', dpi=200, bbox_inches='tight')

pd.set_option('display.width', 220)
print('Confirmatory seed used:', cseed, '| sha256:', conf_hash)
print(REP.groupby(['partition', 'repeat'])[['households', 'locations_also_in_training', 'median_km_test_location_to_nearest_training_location'] + [f'positives_{d}' for d in DIS]].agg(['min', 'max']).round(1).to_string())
print(DIFF.round(3).to_string(index=False)); print(LORO[LORO.eligible_for_LORO].to_string(index=False))
