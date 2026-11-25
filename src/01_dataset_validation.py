"""Stage 1 — verify the dataset facts established by the Stage 1 v1 audit (no re-audit).
Run from the project root:  python src/01_dataset_validation.py"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import load, labels, DIS, ENV, EXCLUDED
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'CBPP': '#2a78d6', 'FMD': '#eb6834', 'LSD': '#1baf7a'}

OUT = Path('results/stage1_v1'); (OUT / 'figures').mkdir(parents=True, exist_ok=False)
df = load('data/raw'); Y = labels(df)

# ---------- 1. facts from the Stage 1 v1 audit, re-checked ----------
raw_cols = ['ID'] + ENV + ['Year_region', 'Longitude', 'Latitude'] + DIS + [f'Weight_{d}' for d in DIS]
loc_years = df.groupby('loc').Year.nunique()
facts = {
    'rows': len(df), 'unique_ids': int(df.ID.nunique()), 'locations': int(df['loc'].nunique()),
    'missing_cells': int(df[raw_cols].isna().sum().sum()),
    'covariates_identical_across_files': True,                    # asserted inside common.data.load
    'prevalence': {d: round(float(df[d].mean()), 4) for d in DIS},
    'positives': {d: int(df[d].sum()) for d in DIS},
    'locations_surveyed_in_one_year': int((loc_years == 1).sum()),
    'households_per_location': df.groupby('loc').size().describe().round(2).to_dict(),
    'years': df.Year.value_counts().sort_index().to_dict(), 'regions': df.Region.value_counts().to_dict()}
expected = {'rows': 3340, 'unique_ids': 3340, 'locations': 462, 'missing_cells': 0, 'locations_surveyed_in_one_year': 449,
            'prevalence': {'CBPP': 0.2240, 'FMD': 0.1832, 'LSD': 0.2240}}
checks = {k: (facts[k] == v) if k != 'prevalence' else all(abs(facts[k][d] - v[d]) < 6e-4 for d in DIS) for k, v in expected.items()}
facts['checks_against_stage1_v1'] = checks; facts['all_checks_pass'] = all(checks.values())

# ---------- 2. leakage: Weight ----------
wl = []
for di, d in enumerate(DIS):
    w = df[f'Weight_{d}'].values; auc = roc_auc_score(Y[:, di], w)
    g = df.groupby(['Year_region', d])[f'Weight_{d}'].nunique()
    n_ = df.groupby('loc')[d].transform('size'); k_ = df.groupby('loc')[d].transform('sum')
    share_same = np.where(df[d] == 1, k_ / n_, (n_ - k_) / n_)          # share of the location's households with the SAME outcome
    from scipy.stats import spearmanr
    wl.append({'disease': d, 'ROC_AUC_of_Weight_alone': auc, 'discrimination_max(AUC,1-AUC)': max(auc, 1 - auc),
               'share_rows_Weight_equals_same_outcome_share_at_location': float(np.isclose(w, share_same, atol=1e-4).mean()),
               'spearman_Weight_vs_same_outcome_share': float(spearmanr(w, share_same).correlation),
               'groups_YearRegion_x_outcome': len(g), 'groups_with_a_single_Weight_value': int((g == 1).sum()),
               'Weight_is_a_function_of_YearRegion_and_outcome': bool((g == 1).all())})
WL = pd.DataFrame(wl); WL.to_csv(OUT / 'weight_leakage_check.csv', index=False)
facts['weight_leakage'] = WL.round(4).to_dict('records')

# ---------- 3. eligible predictors ----------
elig = pd.DataFrame([{'variable': v, 'role': 'eligible predictor (environment)'} for v in ENV] +
                    [{'variable': v, 'role': 'eligible predictor (location)'} for v in ['Longitude', 'Latitude']] +
                    [{'variable': v, 'role': 'EXCLUDED (leakage / sampling design)'} for v in EXCLUDED] +
                    [{'variable': 'ID', 'role': 'identifier only'}] + [{'variable': d.lower(), 'role': 'target'} for d in DIS])
elig.to_csv(OUT / 'eligible_predictors.csv', index=False)
df[ENV + ['Longitude', 'Latitude']].describe().T.round(3).to_csv(OUT / 'environmental_summary.csv')

# ---------- 4. cross-disease relationships ----------
rows = []
loc_rate = df.groupby('loc')[DIS].mean()
for i, a in enumerate(DIS):
    for b in DIS[i + 1:]:
        both = int(((df[a] == 1) & (df[b] == 1)).sum())
        rows.append({'pair': f'{a}-{b}', 'row_level_phi': df[[a, b]].corr().iloc[0, 1], 'location_level_r': loc_rate[[a, b]].corr().iloc[0, 1],
                     'households_with_both': both, 'expected_if_independent': df[a].mean() * df[b].mean() * len(df)})
CD = pd.DataFrame(rows); CD.to_csv(OUT / 'cross_disease_correlation.csv', index=False)
facts['cross_disease'] = CD.round(4).to_dict('records')
json.dump(facts, open(OUT / 'stage1_checks.json', 'w'), indent=1, default=str)

# ---------- 5. figures ----------
L = df.groupby('loc').agg(lon=('Longitude', 'first'), lat=('Latitude', 'first'), n=('ID', 'size'), region=('Region', 'first'), **{d: (d, 'mean') for d in DIS})
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for ax, d in zip(axs, DIS):
    sc = ax.scatter(L.lon, L.lat, s=8 + 2.5 * L.n, c=L[d], cmap='Blues' if d == 'CBPP' else 'Oranges' if d == 'FMD' else 'Greens',
                    vmin=0, vmax=1, edgecolor='#55554f', linewidth=0.3)
    ax.set_title(f'{d}: share of households positive per location', loc='left', fontsize=10); ax.set_xlabel('Longitude')
    fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
axs[0].set_ylabel('Latitude'); fig.suptitle('462 locations (marker size = households sampled)', x=0.01, ha='left')
fig.tight_layout(); fig.savefig(OUT / 'figures' / 'location_prevalence_map_v1.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
for ax, (title, M) in zip(axs, [('Households (phi)', df[DIS].corr()), ('Locations (share positive, r)', loc_rate.corr())]):
    im = ax.imshow(M.values, vmin=-0.4, vmax=0.4, cmap='RdBu_r')
    ax.set_xticks(range(3)); ax.set_xticklabels(DIS); ax.set_yticks(range(3)); ax.set_yticklabels(DIS); ax.set_title(title, loc='left', fontsize=10)
    for i in range(3):
        for j in range(3): ax.text(j, i, f'{M.values[i, j]:.2f}', ha='center', va='center', fontsize=9)
fig.colorbar(im, ax=axs, shrink=0.8); fig.savefig(OUT / 'figures' / 'cross_disease_correlation_v1.png', dpi=200, bbox_inches='tight')

print(json.dumps({k: facts[k] for k in ['rows', 'unique_ids', 'locations', 'missing_cells', 'prevalence', 'locations_surveyed_in_one_year', 'all_checks_pass']}, indent=1))
print(WL.round(3).to_string(index=False)); print(CD.round(3).to_string(index=False))
