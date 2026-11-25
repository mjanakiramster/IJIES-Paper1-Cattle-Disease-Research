"""Numerical consistency check (Deliverable 3), v2 (an omitted result is labelled as needing clarification, not as incorrect): every major result reported in the manuscript compared with the output
printed by the executed notebook (IJIES_Paper1_Experiments_22sections_executed_v5.ipynb).
For each check the notebook value is read from the notebook's own displayed tables or printed text (not from result files),
the manuscript value is the string printed in the manuscript, and the page is found in the manuscript PDF.
Usage (project root): python src/paper1/rev4_numerical_check_v1.py <manuscript spec json> <manuscript pdf> <out csv>"""
import sys, json, re, io, subprocess
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
SPEC, PDF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NB = json.load(open(ROOT / 'IJIES_Paper1_Experiments_22sections_executed_v5.ipynb'))

# ------------------------------------------------------------------ notebook outputs by section
TABS, TEXT = {}, {}
sec = '0'
for c in NB['cells']:
    src = ''.join(c['source'])
    if c['cell_type'] == 'markdown':
        m = re.match(r'##\s+(\d+|Appendix)', src); sec = m.group(1) if m else sec; continue
    for o in c.get('outputs', []):
        d = o.get('data', {})
        if 'text/html' in d:
            for t in pd.read_html(io.StringIO(''.join(d['text/html']))): TABS.setdefault(sec, []).append(t)
        elif o.get('output_type') == 'stream':
            TEXT[sec] = TEXT.get(sec, '') + (''.join(o['text']) if isinstance(o['text'], list) else o['text'])
def T(s, col, n=0):
    hits = [t for t in TABS.get(s, []) if col in t.columns]; return hits[n]
def row(t, **kw):
    q = t
    for k, v in kw.items(): q = q[q[k].astype(str) == str(v)]
    assert len(q) == 1, (kw, len(q)); return q.iloc[0]
def first(s): return float(str(s).split(' ')[0].replace('+', ''))
def txt(s, pat, g=1): m = re.search(pat, TEXT[s]); assert m, pat; return m.group(g)

# ------------------------------------------------------------------ manuscript text and pages
B = json.load(open(SPEC))['blocks']
MS = '\n'.join([b.get('text', '') for b in B] + [str(c) for b in B if b['type'] == 'table' for r in b['rows'] for c in r] + [b.get('note') or '' for b in B if b['type'] == 'table'])
pages = []
n_pages = int(re.search(r'Pages:\s+(\d+)', subprocess.run(['pdfinfo', PDF], capture_output=True, text=True).stdout).group(1))
for p in range(1, n_pages + 1):
    pages.append(re.sub(r'\s+', ' ', subprocess.run(['pdftotext', '-f', str(p), '-l', str(p), '-layout', PDF, '-'], capture_output=True, text=True).stdout))
def page_of(s, hint=None):
    s2 = re.sub(r'\s+', ' ', s)
    for i, p in enumerate(pages, 1):
        if s2 in p and (hint is None or hint in p): return i
    for i, p in enumerate(pages, 1):
        if s2 in p: return i
    return None

ROWS = []
def chk(cid, where, quantity, kind, ms, nb, tol=None, status=None, note='', nbsec='', hint=None):
    """ms: string as printed in the manuscript; nb: notebook value (string or number)."""
    in_ms = ms in MS
    if tol is None: ok = str(nb) == ms
    else:
        nums_ms = [float(x) for x in re.findall(r'[-+]?\d+\.?\d*', ms.replace(',', ''))]; nums_nb = [float(x) for x in re.findall(r'[-+]?\d+\.?\d*', str(nb).replace(',', ''))]
        ok = len(nums_ms) == len(nums_nb) and all(abs(a - b) <= tol + 1e-12 for a, b in zip(nums_ms, nums_nb))
    st = status or ('Correct' if ok and in_ms else ('Incorrect' if not ok else 'Not found in manuscript'))
    if status and not ok and status.startswith('Correct') and not ms.startswith('not reported'): st = 'Incorrect'
    ROWS.append({'ID': cid, 'Manuscript location': where, 'Page': page_of(ms, hint), 'Quantity': quantity, 'Estimate type': kind, 'Manuscript value': ms, 'Notebook value': str(nb),
                 'Notebook section': nbsec, 'Status': st, 'Note': note})

# ------------------------------------------------------------------ data (Sections 2-5)
t1 = T('4', 'Survey year'); al = row(t1, **{'Survey year': 'All'})
for d in ('CBPP', 'FMD', 'LSD'):
    v = al[f'{d} positive, n (%)'].split('(')[1].rstrip(')')
    chk(f'D-{d}', '3.1 text; Table 1', f'{d} overall prevalence (%)', 'count / share', f'{float(v):.1f}% for {d}' if d != 'FMD' else f'{float(v):.1f}% for FMD', v, tol=0.05, nbsec='4 (Table 1)')
chk('D-hh', '3.1 text', 'Households and locations', 'count', '3,340 households at 462 locations', txt('2', r'(3,340 households at 462 locations)'), nbsec='2 (check)')
dq = T('3', 'count')
chk('D-multi', '3.1 text; Table 1 note', 'Locations surveyed in more than one year', 'count', '13 locations were surveyed in more than one year', f"{int(row(dq, **{'Unnamed: 0': 'locations surveyed in >1 year'})['count'])} locations were surveyed in more than one year", nbsec='3')
chk('D-afar', '3.2 text', '2012 households labelled Afar south of 8 deg N', 'count', 'Ten 2012 households', 'Ten 2012 households' if int(row(dq, **{'Unnamed: 0': '2012 Afar households south of 8 deg N (flag)'})['count']) == 10 else 'x', nbsec='3')
y = {int(r['Survey year']): r for _, r in t1[t1['Survey year'] != 'All'].iterrows()}
chk('D-cbpp14', '4.6 text', 'CBPP prevalence 2014', 'share', '43.4% of households in 2014', f"{y[2014]['CBPP positive, n (%)'].split('(')[1].rstrip(')')}% of households in 2014", nbsec='4 (Table 1)')
chk('D-lsd19', '4.6 text', 'LSD prevalence 2018 to 2019', 'share', 'from 7.5% in 2018 to 61.1% in 2019', f"from {y[2018]['LSD positive, n (%)'].split('(')[1].rstrip(')')}% in 2018 to {y[2019]['LSD positive, n (%)'].split('(')[1].rstrip(')')}% in 2019", nbsec='4 (Table 1)')
wl = T('5', 'spearman_Weight_vs_same_outcome_share')
chk('D-weight', '3.2 text', 'Weight variable: Spearman with same-outcome share', 'correlation', 'Spearman correlation 0.84-0.86', f"Spearman correlation {wl['spearman_Weight_vs_same_outcome_share'].min():.2f}-{wl['spearman_Weight_vs_same_outcome_share'].max():.2f}", nbsec='5')
chk('D-weightauc', '3.2 text', 'Weight variable: ROC-AUC alone', 'ROC-AUC', 'ROC-AUC of 0.85-0.93', f"ROC-AUC of {wl['discrimination_max(AUC,1-AUC)'].min():.2f}-{wl['discrimination_max(AUC,1-AUC)'].max():.2f}", nbsec='5')
eta = T('4', 'eta2_survey_year')
chk('D-eta', 'Abstract; 4.6; 6', 'Outcome variance explained by survey year', 'eta squared', '12.5-18.7%', f"{100 * eta.eta2_survey_year.min():.1f}-{100 * eta.eta2_survey_year.max():.1f}%", nbsec='4')
chk('D-etaloc', '4.6 text', 'Variance explained by location within year', 'eta squared', '28-30%', f"{100 * eta.eta2_location_within_year.min():.0f}-{100 * eta.eta2_location_within_year.max():.0f}%", nbsec='4')
d5 = T('4', 'Locations with another campaign within 81 km (%)')
chk('D-overlap', '4.6 text', 'Campaign locations with another campaign within 81 km', 'share', 'between 65% and 100%', f"between {d5['Locations with another campaign within 81 km (%)'].min():.0f}% and {d5['Locations with another campaign within 81 km (%)'].max():.0f}%",
    status='Correct but requires clarification', note='The number is right, but the sentence introduced it as evidence that campaigns "covered different areas"; a high share means campaigns overlap. Reworded in rev4 to say the campaigns overlapped only partly and surveyed different regions.', nbsec='4 (campaign table)')

# ------------------------------------------------------------------ design (Section 4, 7)
D1 = json.loads(re.search(r'\{\s*"D1".*?\n\}', TEXT['4'], re.S).group(0))
chk('V-fold', '3.5 text', 'Test households per fold', 'count range', '630-712 test households per fold', f"{D1['D1']['fold_households_range'][0]}-{D1['D1']['fold_households_range'][1]} test households per fold", nbsec='4 (design checks D1)')
vg = T('7', 'practical_range_km')
out = vg[vg.uses_outcome_labels & vg.variable.str.contains('all years')].practical_range_km.round(0).astype(int).tolist()
chk('V-vario', '3.5 text', 'Outcome variogram practical ranges', 'km', '647, 81 and 67 km', f'{out[0]}, {out[1]} and {out[2]} km', nbsec='7 (variogram table)')
chk('V-free', '3.5 text', 'Label-free variogram ranges (minimum)', 'km', '364 km to beyond 600 km', f"{vg[~vg.uses_outcome_labels].practical_range_km.min():.0f} km to beyond 600 km", nbsec='7 (variogram table)')
chk('V-inner', '4.3 bullet', 'Inner splits checked', 'count', 'All 75 inner splits', f"All {D1['D2']['inner_splits_checked']} inner splits", nbsec='4 (design checks D2)')

# ------------------------------------------------------------------ random vs spatial (Section 11-12)
rt1 = T('11', 'Random minus spatial'); ra = rt1[rt1.Model == '(a) Rank average']
chk('S-gap', 'Abstract; 4.1; 6', 'Random minus spatial ROC-AUC, rank average', 'fold mean difference', '0.10-0.21', f"{ra['Random minus spatial'].min():.2f}-{ra['Random minus spatial'].max():.2f}", nbsec='11',
    status='Correct but requires clarification', note='Difference of fold means (random 5-fold mean minus spatial 15-fold mean). Stated as fold means in rev4.')
chk('S-gapall', '4.1; 4.8', 'Random minus spatial ROC-AUC, all models', 'fold mean difference', '0.10-0.26', f"{rt1['Random minus spatial'].min():.2f}-{rt1['Random minus spatial'].max():.2f}", nbsec='11')
for d in ('CBPP', 'FMD', 'LSD'):
    chk(f'S-gap-{d}', '4.1 text', f'Random minus spatial, rank average, {d}', 'fold mean difference', f"{row(rt1, Disease=d, Model='(a) Rank average')['Random minus spatial']:.3f} ({d})", f"{row(rt1, Disease=d, Model='(a) Rank average')['Random minus spatial']:.3f} ({d})", nbsec='11')
rv = rt1['Random CV ROC-AUC, mean (SD) of 5 folds'].map(first)
chk('S-rand', '4.1; 4.8', 'Random CV ROC-AUC range', 'fold mean', '0.79-0.83', f'{rv.min():.2f}-{rv.max():.2f}', nbsec='11')
sp = rt1['Spatial CV ROC-AUC, mean (SD) of 15 folds'].map(first)
chk('S-spat', '4.1 text', 'Spatial CV ROC-AUC range', 'fold mean', '0.56-0.70', f'{sp.min():.2f}-{sp.max():.2f}', nbsec='11')
chk('S-wyr', '4.1 text', 'Within-year ROC-AUC, spatial to random', 'fold mean', 'from 0.49-0.59 to 0.67-0.74', f"from {rt1['Spatial CV within-year AUC'].min():.2f}-{rt1['Spatial CV within-year AUC'].max():.2f} to {rt1['Random CV within-year AUC'].min():.2f}-{rt1['Random CV within-year AUC'].max():.2f}", nbsec='11')
dist = T('21', 'Same location in training (%)')
chk('S-same', 'Abstract; 4.1', 'Random-CV test households sharing a location with training', 'share', '96.7%', f"{row(dist, scheme='Random 5-fold CV')['Same location in training (%)']:.1f}%", nbsec='21 (distance table)')
chk('S-dist81', '4.1 text', 'Median test-to-training distance, 81 km blocks', 'median over all test households', '35 km', f"{row(dist, scheme='Spatial 81 km')['Median (km)']:.0f} km", nbsec='21 (distance table)',
    status='Correct but requires clarification', note='Median over all test households of all folds (Table S22). Table 9 reports a different aggregation (mean over folds of the median over test locations: 38 km). Both are now labelled.')
t4 = T('12', 'MCC')
for d, m, v in (('CBPP', '(a) Rank average', 'ROC-AUC fold mean (SD)'), ('FMD', 'FS1', 'ROC-AUC fold mean (SD)'), ('FMD', 'FS5', 'ROC-AUC fold mean (SD)'), ('LSD', '(a) Rank average', 'ROC-AUC fold mean (SD)')):
    chk(f'S-t4-{d}-{m[:3]}', 'Table 4', f'{d} {m} spatial ROC-AUC', 'fold mean (SD)', row(t4, Disease=d, Model=m)[v], row(t4, Disease=d, Model=m)[v], nbsec='12 (Table 4)')
for d, m in (('CBPP', '(a) Rank average'), ('FMD', 'FS5'), ('LSD', '(a) Rank average'), ('FMD', 'FS1')):
    chk(f'S-pool-{d}-{m[:3]}', 'Table 4', f'{d} {m} pooled ROC-AUC [bootstrap CI]', 'pooled OOF, location-cluster bootstrap CI', row(t4, Disease=d, Model=m)['ROC-AUC pooled [95% CI]'], row(t4, Disease=d, Model=m)['ROC-AUC pooled [95% CI]'], nbsec='12 (Table 4)')
chk('S-fmdfs1', 'Prompt example', 'FMD FS1: fold mean vs pooled ROC-AUC', 'fold mean / pooled', '0.563 (0.101) / 0.572 [0.518, 0.625]', f"{row(t4, Disease='FMD', Model='FS1')['ROC-AUC fold mean (SD)']} / {row(t4, Disease='FMD', Model='FS1')['ROC-AUC pooled [95% CI]']}", status='Correct', nbsec='12 (Table 4)',
    note='Both values are correct; 0.563 is the mean of 15 fold AUCs, 0.572 the AUC of pooled out-of-fold predictions. The manuscript labels each.')
chk('S-pr', '4.1 text', 'FMD FS1 PR-AUC vs prevalence', 'fold mean', '0.243 against a prevalence of 0.183', f"{row(t4, Disease='FMD', Model='FS1')['PR-AUC (no-skill)'].replace(' (', ' against a prevalence of ').rstrip(')')}", nbsec='12 (Table 4)')
for d, m in (('CBPP', '(a) Rank average'), ('FMD', 'FS5'), ('LSD', 'FS1')):
    chk(f'S-mcc-{d}', 'Table 4', f'{d} {m} MCC', 'fold mean', f"{row(t4, Disease=d, Model=m)['MCC']:.2f}", f"{row(t4, Disease=d, Model=m)['MCC']:.2f}", tol=0.005, nbsec='12 (Table 4)')
lib = T('14', 'Model [feature set]')
for d, txt_ in (('CBPP', '0.703 for CBPP (GAM [B])'), ('FMD', '0.668 for FMD (LightGBM [ELK])'), ('LSD', '0.609 for LSD (ExtraTrees [ELK])')):
    v = lib[f'{d} ROC-AUC mean (SD)'].map(first); i = v.idxmax()
    chk(f'S-lib-{d}', '4.1 text', f'Best library model, {d}', 'fold mean', txt_, f"{v[i]:.3f} for {d} ({lib.loc[i, 'Model [feature set]']})", nbsec='14 (library table)')

# ------------------------------------------------------------------ calibration (Section 17)
cal = T('17', 'Calibration slope: fold median (IQR)')
def cal_rng(d, col, f=lambda s: float(s.split(' ')[0])): v = cal[cal.Disease == d][col].map(f); return v.min(), v.max()
a, b = cal_rng('FMD', 'Calibration slope: fold median (IQR)'); chk('C-fold-FMD', '4.1 text; Table 5', 'FMD calibration slope, fold median range', 'fold median', '1.17-1.84 for FMD', f'{a:.2f}-{b:.2f} for FMD', nbsec='17')
a, b = cal_rng('LSD', 'Calibration slope: fold median (IQR)'); chk('C-fold-LSD', '4.1 text; Table 5', 'LSD calibration slope, fold median range', 'fold median', '1.01-1.76 for LSD', f'{a:.2f}-{b:.2f} for LSD', nbsec='17')
a, b = cal_rng('FMD', 'Calibration slope: pooled [95% CI]'); chk('C-pool-FMD', '4.1 text; Table 5', 'FMD calibration slope, pooled range', 'pooled OOF', '-0.35 to 0.97 for FMD', f'{a:.2f} to {b:.2f} for FMD', nbsec='17')
a, b = cal_rng('LSD', 'Calibration slope: pooled [95% CI]'); chk('C-pool-LSD', '4.1 text; Table 5', 'LSD calibration slope, pooled range', 'pooled OOF', '0.11 to 0.70 for LSD', f'{a:.2f} to {b:.2f} for LSD', nbsec='17')
for d in ('CBPP', 'FMD', 'LSD'):
    a, b = cal_rng(d, 'Brier skill score [95% CI]', lambda s: float(s.split(' ')[0]))
    f2 = lambda x: f'{x:+.2f}'.replace('-0.00', '0.00').replace('+0.00', '0.00')
    chk(f'C-bss-{d}', '4.1 text; Table 5', f'{d} Brier skill score range', 'pooled OOF', f'{f2(a)} to {f2(b)} for {d}', f'{f2(a)} to {f2(b)} for {d}', nbsec='17')
chk('C-lsd-brier', 'Table 5', 'LSD FS5 Brier [CI] and reference', 'pooled OOF, bootstrap CI', f"{row(cal, Disease='LSD', Model='FS5')['Brier: pooled [95% CI]']}", row(cal, Disease='LSD', Model='FS5')['Brier: pooled [95% CI]'], nbsec='17')

# ------------------------------------------------------------------ ablation (Section 13)
ab = T('13', 'Feature set'); inc = T('13', 'Step')
def ia(step, d, col='Fold mean [corrected-t 95% CI]'): return row(inc, Step=step, Disease=d)[col]
chk('A-fs1', 'Abstract; 4.2', 'FMD FS1 to FS5 ROC-AUC', 'fold mean', 'from 0.563 to 0.663', f"from {first(row(ab, Disease='FMD', **{'Feature set': 'FS1'})['ROC-AUC fold mean (SD)']):.3f} to {first(row(ab, Disease='FMD', **{'Feature set': 'FS5'})['ROC-AUC fold mean (SD)']):.3f}", nbsec='13')
for step, lab in (('FS2 - FS1', 'climate'), ('FS3 - FS2', 'coordinates'), ('FS4 - FS3', 'rates'), ('FS5 - FS4', 'density')):
    chk(f'A-{lab}', '4.2 text; Table 6', f'FMD increment: {lab}', 'fold mean, corrected-t CI', ia(step, 'FMD'), ia(step, 'FMD'), nbsec='13 (increments)')
for step, lab in (('FS2 - FS1', 'climate'), ('FS3 - FS2', 'coordinates'), ('FS4 - FS3', 'rates')):
    chk(f'A-pool-{lab}', '4.2 text', f'FMD increment: {lab}', 'pooled, bootstrap CI', ia(step, 'FMD', 'Pooled [bootstrap 95% CI]').split(' [')[0], ia(step, 'FMD', 'Pooled [bootstrap 95% CI]').split(' [')[0], nbsec='13 (increments)')
chk('A-holm', 'Table 6 note', 'Holm p over 12 steps', 'Holm-adjusted p', 'all above 0.05', 'all above 0.05' if (inc['Holm p (12 tests)'].astype(float) > 0.05).all() else 'x', nbsec='13 (increments)')
chk('A-lsdwy', '4.2 text', 'LSD within-year ROC-AUC FS3 to FS4', 'fold mean', 'from 0.540 to 0.497', f"from {float(row(ab, Disease='LSD', **{'Feature set': 'FS3'})['Within-year AUC fold mean']):.3f} to {float(row(ab, Disease='LSD', **{'Feature set': 'FS4'})['Within-year AUC fold mean']):.3f}", nbsec='13')
for d in ('CBPP', 'FMD', 'LSD'):
    for s in ('FS1', 'FS5'):
        chk(f'A-t6-{d}-{s}', 'Table 6', f'{d} {s} pooled ROC-AUC [CI]', 'pooled OOF, bootstrap CI', row(ab, Disease=d, **{'Feature set': s})['ROC-AUC pooled [95% CI]'], row(ab, Disease=d, **{'Feature set': s})['ROC-AUC pooled [95% CI]'], nbsec='13')

# ------------------------------------------------------------------ paired comparisons (Section 14/20)
t7 = T('20', 'Overall: Holm p')
fm = row(t7, Comparison='FS5 minus FS1', Disease='FMD')
chk('P-fmd', '4.2 text; Table 7', 'FMD FS5 - FS1', 'fold mean, corrected-t CI', fm['Overall: fold mean [corrected-t 95% CI]'], fm['Overall: fold mean [corrected-t 95% CI]'], nbsec='20 (Table 7)')
chk('P-fmdholm', '4.2 text; Table 7', 'FMD FS5 - FS1 Holm p (3 diseases)', 'Holm-adjusted p, corrected t', f"Holm p = {fm['Overall: Holm p']:.3f}", f"Holm p = {fm['Overall: Holm p']:.3f}", nbsec='20 (Table 7)')
chk('P-fmdpool', '4.2 text; Table 7', 'FMD FS5 - FS1', 'pooled, bootstrap CI', fm['Overall: pooled [bootstrap 95% CI]'], fm['Overall: pooled [bootstrap 95% CI]'], nbsec='20 (Table 7)')
chk('P-fmdwy', '4.6 text; Table 7', 'FMD FS5 - FS1 within-year', 'pooled, bootstrap CI', fm['Within-year: pooled [bootstrap 95% CI]'], fm['Within-year: pooled [bootstrap 95% CI]'], nbsec='20 (Table 7)')
chk('P-fmdwyt', '4.6 text; Table 7', 'FMD FS5 - FS1 within-year', 'fold mean, corrected-t CI', fm['Within-year: fold mean [corrected-t 95% CI]'], fm['Within-year: fold mean [corrected-t 95% CI]'], nbsec='20 (Table 7)')
for d in ('CBPP', 'LSD'):
    r_ = row(t7, Comparison='FS5 minus FS1', Disease=d)
    chk(f'P-{d}', '4.2 text; Table 7', f'{d} FS5 - FS1', 'fold mean, corrected-t CI', r_['Overall: fold mean [corrected-t 95% CI]'], r_['Overall: fold mean [corrected-t 95% CI]'], nbsec='20 (Table 7)')
r_ = row(t7, Comparison='FS5 minus FS1', Disease='LSD'); chk('P-lsdwy', '4.6 text', 'LSD FS5 - FS1 within-year', 'pooled, bootstrap CI', r_['Within-year: pooled [bootstrap 95% CI]'], r_['Within-year: pooled [bootstrap 95% CI]'], nbsec='20 (Table 7)')
r_ = row(t7, Comparison='(a) Rank average minus FS5', Disease='FMD'); chk('P-rafs5', '4.2 text', 'FMD rank average - FS5', 'pooled, bootstrap CI', r_['Overall: pooled [bootstrap 95% CI]'].split(' [')[0], r_['Overall: pooled [bootstrap 95% CI]'].split(' [')[0], nbsec='20 (Table 7)')
gb = T('14', 'Rank average, 23 models')
chk('P-gb', '3.4 text', 'Ensemble without group-balanced model: largest mean change', 'fold mean difference', 'at most 0.001', f"at most {gb['Difference [corrected-t 95% CI]'].map(lambda s: abs(first(s))).max():.3f}", nbsec='14')
t11 = T('14', 'Version')
chk('P-prior', '4.3 bullet; 5', 'Old-code prior: largest change in mean rank-average ROC-AUC', 'fold mean difference', '0.0027', f"{t11['Difference [corrected-t 95% CI]'].map(lambda s: abs(first(s))).max():.4f}", nbsec='14 (Table 11, shown in v5)')

# ------------------------------------------------------------------ negative controls (Section 20)
t10 = T('20', 'Observed gain'); hf = row(t10, Scheme='household', Disease='FMD'); vf = row(t10, Scheme='village', Disease='FMD')
chk('N-obs', '4.3 text', 'Observed FMD gain on development repeat 1', 'fold mean over the 5 folds of repeat 1', f"({hf['Observed gain']:.3f})", f"({hf['Observed gain']:.3f})", nbsec='20 (Table 8)',
    status='Correct but requires clarification', note='0.084 is the mean over the five folds of development repeat 1, the only repeat used for the permutations; the 0.100 in Section 4.2 is the mean over all 15 folds. rev4 states this.')
chk('N-p', '4.3 text; Table 8', 'FMD permutation p, household and village', 'permutation p (100 permutations)', 'p = 0.010 in both designs', f"p = {hf['Permutation p (resolution)'].split(' ')[0]} in both designs" if hf['Permutation p (resolution)'] == vf['Permutation p (resolution)'] else 'x', nbsec='20 (Table 8)')
chk('N-holm', '4.3 text; Table 8', 'FMD Holm-adjusted permutation p (6 tests)', 'Holm-adjusted p', 'p = 0.059', f"p = {hf['Holm p (6 tests)']:.3f}", nbsec='20 (Table 8)',
    status='Correct but requires clarification', note='Value correct. rev3 said the gain "passed the negative controls" in the conclusion; corrected: the gain exceeded every permutation in both null designs, but the Holm-adjusted p (0.059) is slightly above 0.05 because 100 permutations limit p to 0.0099. More permutations are recommended.')
chk('N-res', '5 Limitations', 'Smallest attainable permutation p', 'resolution', '0.010 or above', f"{float(txt('20', r'= (0\.0099);')):.3f} or above", nbsec='20 (printed in v5)')

# ------------------------------------------------------------------ explainability (Section 15)
sg = T('15', 'mean_abs_SHAP_sum')
for g, lab in (('Own-disease rates', 'own-disease'), ('Cross-disease rates', 'cross-disease'), ('Coordinates', 'coordinates'), ('Climate indices', 'climate indices')):
    chk(f'X-{lab}', '4.4 text', f'FMD summed mean |SHAP|: {g}', 'mean |SHAP| (repeat 1)', f"{lab} {row(sg, disease='FMD', group=g).mean_abs_SHAP_sum:.3f}", f"{lab} {row(sg, disease='FMD', group=g).mean_abs_SHAP_sum:.3f}", nbsec='15')
rho = re.search(r"\{'CBPP': np.float64\(([\d.]+)\), 'FMD': np.float64\(([\d.]+)\), 'LSD': np.float64\(([\d.]+)\)\}", TEXT['15']).groups()
chk('X-rho', '4.4 text', 'SHAP vs permutation group ranking', 'Spearman rho (6 groups)', '0.83-0.94', f'{min(map(float, rho)):.2f}-{max(map(float, rho)):.2f}', nbsec='15')

# ------------------------------------------------------------------ applicability (Section 16, 12, 21)
ao = T('16', 'Test households inside AOA (%)')
chk('AOA-in', '4.5 text', 'Test households inside the AOA', 'share over all folds', '98.6-99.5%', f"{ao['Test households inside AOA (%)'].min():.1f}-{ao['Test households inside AOA (%)'].max():.1f}%", nbsec='16')
chk('AOA-rho', '4.5 text', 'DI vs absolute error', 'Spearman rho', '-0.08 to +0.17', f"{ao['Spearman DI vs |error|'].min():+.2f} to {ao['Spearman DI vs |error|'].max():+.2f}", nbsec='16')
t9 = T('12', 'Median nearest training location (km)', 0)
for c, lab in (('buffer_25', '25 km buffer'), ('buffer_50', '50 km buffer'), ('matched_50', 'size-matched 50 km'), ('buffer_100', '100 km buffer')):
    chk(f'AOA-{c}', '4.5 text; Table 9', f'FMD FS5 - FS1 with {lab}', 'fold mean, corrected-t CI (condition ratio)', row(t9, Condition=c, Disease='FMD')['FS5 - FS1 [95% CI]'], row(t9, Condition=c, Disease='FMD')['FS5 - FS1 [95% CI]'], nbsec='12 (Table 9)')
chk('AOA-d100', '4.5 text', 'Median distance with 100 km buffer', 'see note', '125 km', f"{row(dist, scheme='Spatial 81 km + 100 km buffer')['Median (km)']:.0f} km", nbsec='21 (distance table)',
    status='Correct but requires clarification', note='125 km is the median over all test households (Table S22); Table 9 shows 126 km (mean over folds of the median over test locations). rev4 cites the Table 9 value next to Table 9 and names the aggregation.')
t8 = T('12', 'Median nearest training location (km)', 1)
for c in ('blocks120', 'blocks160'):
    chk(f'AOA-{c}', '4.5 text; Table 9', f'FMD FS5 - FS1, {c}', 'fold mean, corrected-t CI (condition ratio)', row(t8, Condition=c, Disease='FMD')['FS5 - FS1 [95% CI]'], row(t8, Condition=c, Disease='FMD')['FS5 - FS1 [95% CI]'], nbsec='12 (Table 8 of the notebook)')

# ------------------------------------------------------------------ campaigns (Section 18)
t5 = T('18', 'Within-year AUC pooled [95% CI]')
for d, m in (('CBPP', '(a) Rank average'), ('FMD', 'FS5'), ('LSD', '(a) Rank average')):
    v = row(t5, Disease=d, Model=m)['Within-year AUC pooled [95% CI]'].split(' ')[0]
    chk(f'W-{d}', '4.6 text; Table 4', f'{d} best-model within-year ROC-AUC', 'pooled OOF', v, v, nbsec='18')
cid = T('18', 'Macro ROC-AUC for survey year [corrected-t 95% CI]')
for f_ in ('FS1', 'Neighbourhood rates only', 'FS5'):
    v = row(cid, Features=f_)['Macro ROC-AUC for survey year [corrected-t 95% CI]'].split(' ')[0]
    chk(f'Y-{f_[:5]}', '4.6 text; Table 10', f'Survey-year identification, {f_}', 'fold mean macro ROC-AUC', v, v, nbsec='18')
dd = re.search(r'FS1, macro AUC for survey year \[corrected-t 95% CI\]: ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]', TEXT['18']).groups()
chk('Y-diff', '4.6 text', 'Rates only minus FS1 (year identification)', 'fold mean, corrected-t CI', f'difference {dd[0]}, 95% CI {dd[1]} to {dd[2]}', f'difference {dd[0]}, 95% CI {dd[1]} to {dd[2]}', nbsec='18')
rt2 = T('18', 'Campaign-specific minus pooled [corrected-t 95% CI]')
v = row(rt2, Disease='LSD', Features='FS5')['Campaign-specific minus pooled [corrected-t 95% CI]']; chk('Y-camp-LSD', '4.6 text; Table 11', 'LSD FS5 campaign-specific minus pooled', 'fold mean, corrected-t CI', v.split(' ')[0], v.split(' ')[0], nbsec='18')
ls = float(txt('18', r'all years\): ([\d.]+)%')); chk('Y-later', '5 Limitations', 'Share of 150 km kernel weight from later surveys', 'mean over test rows (earlier audit)', '31%', f'{ls:.0f}%', nbsec='18 (printed in v5; earlier audit file)')

# ------------------------------------------------------------------ temporal (Section 19)
t12 = T('19', 'Train prevalence'); tv = t12['ROC-AUC [95% CI]'].map(first)
chk('TM-range', 'Abstract; 4.7', 'Forward temporal ROC-AUC range', 'single train/test split', '0.35-0.44', f'{tv.min():.2f}-{tv.max():.2f}', nbsec='19')
chk('TM-below', '4.7 text', 'Cases with 95% CI entirely below 0.5', 'location-cluster bootstrap CI', '3 of the 6 cases', f"{int((t12['ROC-AUC [95% CI]'].map(lambda s: float(s.split('[')[1].split(',')[1].rstrip(']'))) < 0.5).sum())} of the 6 cases", nbsec='19')
for d in ('CBPP', 'FMD', 'LSD'):
    r_ = t12[t12.Disease == d].iloc[0]; chk(f'TM-prev-{d}', '4.7 text', f'{d} prevalence train to test', 'share', f"{100 * r_['Train prevalence']:.1f}%", f"{100 * r_['Train prevalence']:.1f}%", nbsec='19')

# ------------------------------------------------------------------ leakage audit (Section 8, 20)
chk('L-74', '4.3 text', 'Features traced in the audit', 'count (earlier audit)', '74 features', f"{txt('8', r'features traced in the audit: (\d+)')} features", nbsec='8 (printed in v5)')
m = re.search(r'differing before the tie rule: (\d+) of (\d+)', TEXT['8'])
chk('L-knn', '4.3 bullet', 'k-nearest values differing before the tie rule', 'count (earlier audit)', '1,001 of 570,915', f'{int(m.group(1)):,} of {int(m.group(2)):,}', nbsec='8 (printed in v5)')
P4 = eval(re.search(r"\{'tie_rule'.*?\}", TEXT['8'], re.S).group(0))
chk('L-p4', '4.3 bullet', 'Independent reimplementation: k-nearest values after the tie rule', 'count', f"{P4['new_code_values']:,} values with {P4['new_code_mismatches']} mismatches", f"{P4['new_code_values']:,} values with {P4['new_code_mismatches']} mismatches", nbsec='8')
sv = T('20', 'max_abs_difference')
chk('L-stat', '4.3 bullet', 'Independent recomputation of table statistics', 'max abs difference', 'within 0.0005', f"within {sv[sv.quantity.str.contains('Table')].max_abs_difference.max():.4f}", nbsec='20')

# ------------------------------------------------------------------ confirmatory (Appendix)
cf = T('Appendix', 'FMD').set_index('disease')
chk('CF-fmd', '4.2 text', 'Confirmatory FMD FS5 - FS1', 'pooled, bootstrap CI', f"+{float(cf.loc['H1 FS5 - FS1: pooled', 'FMD']):.3f} (95% CI +{float(cf.loc['H1 FS5 - FS1: CI_lo', 'FMD']):.3f} to +{float(cf.loc['H1 FS5 - FS1: CI_hi', 'FMD']):.3f}",
    f"+{float(cf.loc['H1 FS5 - FS1: pooled', 'FMD']):.3f} (95% CI +{float(cf.loc['H1 FS5 - FS1: CI_lo', 'FMD']):.3f} to +{float(cf.loc['H1 FS5 - FS1: CI_hi', 'FMD']):.3f}", nbsec='Appendix (shown in v5)',
    status='Correct but requires clarification', note=f"Bootstrap interval correct. The fold-level corrected-t p for the same comparison was {float(cf.loc['H1 FS5 - FS1: p_corrected', 'FMD']):.3f} (not reported in rev3); rev4 reports both, and that the hypotheses were fixed after the development results.")
chk('CF-lsd', '4.2 text', 'Confirmatory LSD FS5 - FS1', 'pooled, bootstrap CI', f"+{float(cf.loc['H1 FS5 - FS1: pooled', 'LSD']):.3f} (95% CI {float(cf.loc['H1 FS5 - FS1: CI_lo', 'LSD']):.3f} to +{float(cf.loc['H1 FS5 - FS1: CI_hi', 'LSD']):.3f}",
    f"+{float(cf.loc['H1 FS5 - FS1: pooled', 'LSD']):.3f} (95% CI {float(cf.loc['H1 FS5 - FS1: CI_lo', 'LSD']):.3f} to +{float(cf.loc['H1 FS5 - FS1: CI_hi', 'LSD']):.3f}", nbsec='Appendix (shown in v5)')
chk('CF-h2', '4.2 text (rev4)', 'Confirmatory FS5 minus rank average', 'pooled, bootstrap CI', 'not reported in rev3', f"CBPP {float(cf.loc['H2 FS5 - (a): pooled', 'CBPP']):+.3f}, FMD {float(cf.loc['H2 FS5 - (a): pooled', 'FMD']):+.3f}, LSD {float(cf.loc['H2 FS5 - (a): pooled', 'LSD']):+.3f}",
    status='Correct but requires clarification', note='The second confirmatory hypothesis was not reported in the main text of rev3. For FMD the development advantage of FS5 over the ensemble was not seen; for LSD the ensemble was better. Added in rev4.', nbsec='Appendix (shown in v5)')

# ------------------------------------------------------------------ software and external
vers = json.loads(re.search(r'\{.*?\}', TEXT['1'], re.S).group(0))
chk('E-ver', '3.6 text', 'Software versions', 'environment', 'scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10', f"scikit-learn {vers['sklearn']}, XGBoost {vers['xgboost']}, LightGBM {vers['lightgbm']}, CatBoost {vers['catboost']}", nbsec='1')
chk('E-gizaw', '4.8 text', 'Kriging cross-validation correlation in Gizaw et al.', 'external', 'correlations above 0.99', 'not in notebook', status='Not verifiable', note='Value from the cited preprint [14], not from this study; it cannot be checked against the notebook.', nbsec='-')

R = pd.DataFrame(ROWS); R.to_csv(OUT, index=False)
print(R.Status.value_counts().to_string()); print(R[~R.Status.isin(['Correct'])][['ID', 'Status', 'Manuscript value', 'Notebook value']].to_string())
