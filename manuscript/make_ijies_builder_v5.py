"""IJIES builder v5 (approved by the author 2026-11-22):
  1. negative controls with 1,000 permutations per scheme (rev5 table Table10_negative_controls_1000_v1.csv);
     every permutation statement is generated from that table, so the wording follows the result;
  2. simplified abstract (fewer numbers, one sentence on practical use);
  3. one sentence in 4.1 reporting the spatial baselines of the library (GP kriging), from Supplementary Table S4;
  4. practical-guidance sentences in the Conclusion; the limitation on permutation resolution replaced.
Writes new files only: manuscript/ijies_body_v5.py, manuscript/build_manuscript_ijies_v5.py
(-> manuscript_ijies_spec_v5.json, manuscript_ijies_meta_v5.json). Run from the manuscript folder."""

b = open('ijies_body_v4.py').read()


def rep(old, new, count=1):
    global b
    assert b.count(old) == count, (b.count(old), old[:80])
    b = b.replace(old, new)


rep('# Body v4 (= v3 + Egypt LSD reference Gouda2025 in Related work).',
    '# Body v5 (= v4 + 1,000-permutation negative controls, simplified abstract, spatial-baseline sentence, practical guidance). Body v4 (= v3 + Egypt LSD reference Gouda2025 in Related work).')

# ---- permutation quantities, all read from the 1,000-permutation table
rep("holm_fmd = float(t10('household', 'FMD')['Holm p (6 tests)'])\n",
    "holm_fmd = max(float(t10(s, 'FMD')['Holm p (6 tests)']) for s in ('household', 'village'))\n"
    "k_fmd = [int(t10(s, 'FMD')['Null gains >= observed']) for s in ('household', 'village')]\n"
    "def _holm(p):\n"
    "    o = sorted(range(len(p)), key=lambda i: p[i]); adj = [0.0] * len(p); run = 0.0\n"
    "    for k, i in enumerate(o): run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run\n"
    "    return adj\n"
    "_p100 = [float(v) for v in T10['p with first 100 permutations']]\n"
    "holm100_fmd = max(a for a, s, d in zip(_holm(_p100), T10.Scheme, T10.Disease) if d == 'FMD')\n"
    "perm_all_exceeded = k_fmd == [0, 0]\n"
    "holm_fmd_hv = [float(t10(s, 'FMD')['Holm p (6 tests)']) for s in ('household', 'village')]\n"
    "def _none(k, lab): return f'none of the {n_perm:,} {lab}' if k == 0 else f'{k} of the {n_perm:,} {lab}'\n"
    "def _sig(ps): return ('both below' if max(ps) < 0.05 else 'both above' if min(ps) >= 0.05 else 'one below and one above') if len(ps) == 2 else ''\n"
    "sig_txt = 'below' if holm_fmd < 0.05 else 'above'\n"
    "_lib = pd.read_csv(RT / 'RT3_library_models_spatial_v1.csv').set_index('Model [feature set]')\n"
    "gp = {d: float(_lib.loc['GP-kriging [E]', f'{d} ROC-AUC mean (SD)'].split(' ')[0]) for d in DIS}\n"
    "gp_gap = [min(fm(d, 'FS5'), fm(d, '(a) Rank average')) - gp[d] for d in DIS] + [max(fm(d, 'FS5'), fm(d, '(a) Rank average')) - gp[d] for d in DIS]\n")

# ---- 2. simplified abstract
a0 = b.index("add('para', '**Abstract:** ' + (")
a1 = b.index("'Such models suit retrospective mapping within the surveyed period, not forecasting.'))") + len("'Such models suit retrospective mapping within the surveyed period, not forecasting.'))")
b = b[:a0] + '''add('para', '**Abstract:** ' + (
    'Machine-learning maps of livestock disease are usually checked with random cross-validation, which can overstate accuracy when nearby observations are similar. '
    'We evaluated such maps for household-reported contagious bovine pleuropneumonia (CBPP), foot-and-mouth disease (FMD) and lumpy skin disease (LSD) in 3,340 Ethiopian households from five survey campaigns (2012-2019), using 81 km spatial block cross-validation, nested feature sets from five environmental variables to 39 features, ExtraTrees and a 23-model ensemble, a leakage audit of neighbourhood disease rates, label-permutation controls and campaign-aware tests. '
    f'Random cross-validation gave mean ensemble ROC-AUC {min(gap_a.values()):.2f}-{max(gap_a.values()):.2f} higher than spatial validation, because almost all of its test households had training households at the same location. '
    f'Under spatial validation, the engineered features improved FMD prediction (ROC-AUC {fm("FMD", "FS1"):.3f} to {fm("FMD", "FS5"):.3f}), mainly through climate indices and coordinates; the neighbourhood rates were leakage-safe but added little once location was known. '
    + (f'The FMD gain exceeded all {n_perm:,} label permutations in both permutation designs (Holm-adjusted p = {holm_fmd:.3f}). ' if perm_all_exceeded else
       f'The FMD gain was reached by {_none(k_fmd[0], "household-level")} and {_none(k_fmd[1], "village-level")} label permutations (Holm-adjusted p ≤ {holm_fmd:.3f}). ')
    + 'CBPP and LSD showed no consistent gain. '
    f'Much of the apparent skill reflected differences between survey campaigns: within single years ROC-AUC was only {min(wy_best.values()):.2f}-{max(wy_best.values()):.2f}, and models trained on 2012-2015 did not transfer to 2018-2019. '
    'Such maps can help rank surveyed areas within one campaign for follow-up surveillance, but should not be used to compare campaigns or to forecast.'))''' + b[a1:]

# ---- methods: number of permutations
rep("Negative controls rebuilt the whole pipeline on permuted labels for development repeat 1 ({n_perm} household-level and {n_perm} village-level permutations); the permutation p-value is (1 + null gains at least as large as the observed gain) / (1 + {n_perm}), so it cannot be smaller than {1 / (n_perm + 1):.4f}.",
    "Negative controls rebuilt the whole pipeline on permuted labels for development repeat 1 ({n_perm:,} household-level and {n_perm:,} village-level permutations; the first 100 per scheme come from the original analysis and 900 were added later with the same code and seed rule); the permutation p-value is (1 + null gains at least as large as the observed gain) / (1 + {n_perm:,}), so it cannot be smaller than {1 / (n_perm + 1):.4f}.")

# ---- 3. spatial-baseline sentence at the end of the first 4.1 paragraph
rep("in line with the selection-bias argument of Cawley and Talbot {{Cawley2010}}.')",
    "in line with the selection-bias argument of Cawley and Talbot {{Cawley2010}}. The library also contained a spatial baseline evaluated under the same blocks, a location-level Gaussian-process model on environmental, climate and coordinate inputs (a form of kriging with covariates): its fold-mean ROC-AUC was {gp[\"CBPP\"]:.3f} (CBPP), {gp[\"FMD\"]:.3f} (FMD) and {gp[\"LSD\"]:.3f} (LSD) (Supplementary Table S4), {min(gp_gap):.2f}-{max(gp_gap):.2f} below FS5 and the rank average, so the gain over this simpler spatial model is small.')")

# ---- 1. results 4.3 with 1,000 permutations
s0 = b.index("para(f'Under permuted labels, ROC-AUC stayed close to 0.5")
s1 = b.index("CBPP did not differ from either null.')", s0) + len("CBPP did not differ from either null.')")
b = b[:s0] + '''_lsd_v = float(vl["Permutation p (resolution)"].split(" ")[0]); _lsd_h = float(hl["Permutation p (resolution)"].split(" ")[0])
_cb = [float(t10(s, 'CBPP')["Permutation p (resolution)"].split(" ")[0]) for s in ('household', 'village')]
para(f'Under permuted labels, ROC-AUC stayed close to 0.5 and the FS5 gain close to zero ({T("null")}; Supplementary Figure S6). The permutations used development repeat 1, so values are means over its five folds; the observed FMD gain there was {float(hf["Observed gain"]):.3f} (against {fmd_gain:.3f} over all 15 folds). '
     + (f'It exceeded all {n_perm:,} household-level and all {n_perm:,} village-level permutations (p = {p_fmd[0]} in both designs, the smallest attainable value), and after Holm correction across the six tests p = {holm_fmd:.3f}, {sig_txt} 0.05. '
        if perm_all_exceeded else
        f'This gain was reached by {_none(k_fmd[0], "household-level")} and {_none(k_fmd[1], "village-level")} permutations (p = {p_fmd[0]} and {p_fmd[1]}); after Holm correction across the six tests p = {holm_fmd_hv[0]:.3f} and {holm_fmd_hv[1]:.3f}, {_sig(holm_fmd_hv)} 0.05. ')
     + f'With the first 100 permutations alone, as in the earlier analysis, the Holm-adjusted p was {holm100_fmd:.3f}, the smallest value that design allowed. '
     + (('The FMD gain is therefore unlikely to arise from the feature construction or from chance' + (', although a few village-level permutations reached it, so clustering within villages can occasionally produce a gain this large. ' if k_fmd[1] > 0 else '. ')) if holm_fmd < 0.05 else 'The FMD gain is therefore unlikely to arise from the feature construction or from chance alone, but this test does not establish it at the 5% level after correction. ')
     + f'The LSD gain was unusual under the household-level null (p = {hl["Permutation p (resolution)"].split(" ")[0]}; Holm p = {float(hl["Holm p (6 tests)"]):.3f})'
     + (f' but not under the village-level null (p = {vl["Permutation p (resolution)"].split(" ")[0]}), so it can be explained by clustering within villages' if _lsd_v >= 0.05 else f' and under the village-level null (p = {vl["Permutation p (resolution)"].split(" ")[0]}; Holm p = {float(vl["Holm p (6 tests)"]):.3f})')
     + ('; CBPP did not differ from either null.' if min(_cb) >= 0.05 else f'; for CBPP p = {_cb[0]:.4f} (household) and {_cb[1]:.4f} (village).'))''' + b[s1:]
rep("rows = [[r.Scheme, r.Disease, r['Null FS5 ROC-AUC mean [2.5-97.5%]'], r['Null gain mean [2.5-97.5%]'], r['Observed gain'], r['Permutation p (resolution)'], r['Holm p (6 tests)']] for _, r in T10.iterrows()]",
    "rows = [[r.Scheme, r.Disease, r['Null FS5 ROC-AUC mean [2.5-97.5%]'], r['Null gain mean [2.5-97.5%]'], r['Observed gain'], r['Permutation p (resolution)'], f\"{float(r['Holm p (6 tests)']):.3f}\"] for _, r in T10.iterrows()]")
rep("table('null', f'Negative controls: full pipeline rebuilt on permuted labels ({n_perm} permutations per scheme,",
    "table('null', f'Negative controls: full pipeline rebuilt on permuted labels ({n_perm:,} permutations per scheme,")
rep("note=f'Permutation p = (1 + null gains at least as large as the observed gain) / (1 + {n_perm}); smallest attainable value {1 / (n_perm + 1):.4f}. Holm correction across six tests (three diseases × two schemes).')",
    "note=f'Permutation p = (1 + null gains at least as large as the observed gain) / (1 + {n_perm:,}); smallest attainable value {1 / (n_perm + 1):.4f}. The first 100 permutations per scheme are those of the original analysis. Holm correction across six tests (three diseases × two schemes).')")

# ---- 4. limitations and conclusion
rep(" With {n_perm} permutations per scheme, the Holm-adjusted FMD permutation p-value cannot fall below {6 / (n_perm + 1):.3f}; more permutations would give a stronger test.',",
    " The permutation controls cover development repeat 1 only, and their number was raised from 100 to {n_perm:,} per scheme after the 100-permutation result was known; this sharpens the resolution of the p-value but does not change the test.',")
rep("the gain exceeded all label permutations (Holm-adjusted p = {holm_fmd:.3f}),",
    "the gain {'exceeded all ' + format(n_perm, ',') + ' label permutations in both designs' if perm_all_exceeded else 'was reached by ' + _none(k_fmd[0], 'household-level') + ' and ' + _none(k_fmd[1], 'village-level') + ' label permutations'} (Holm-adjusted p {'=' if perm_all_exceeded else '≤'} {holm_fmd:.3f}),")
rep("para('Within a single campaign, such maps can help rank surveyed areas for follow-up surveillance, but they should not be used to compare risk across campaigns or to forecast. Studies that pool campaign-based surveys should report within-campaign discrimination next to overall scores, test outcome-derived features for leakage and with an ablation, and use probability-based or repeated surveys if the aim is forecasting.')",
    "para(f'Within a single campaign, such maps can help rank surveyed areas for follow-up surveillance, but they should not be used to compare risk across campaigns or to forecast. Because within-campaign ROC-AUC was only {min(wy_best.values()):.2f}-{max(wy_best.values()):.2f}, animal-health teams should use them to decide which surveyed locations to revisit first, together with field reports, not as stand-alone risk estimates; the LSD probabilities in particular should not be read as risk levels. Studies that pool campaign-based surveys should report within-campaign discrimination next to overall scores, test outcome-derived features for leakage and with an ablation, check whether the features identify the survey campaign, and use probability-based or repeated surveys if the aim is forecasting.')")
open('ijies_body_v5.py', 'w').write(b)

# ---- assemble builder v5 (header and footer from rev4 v3, as for v2-v4)
s = open('build_manuscript_rev4_v3.py').read()
head = s[:s.index('# ------------------------------------------------------------------ abstract')]
foot = s[s.index('# ------------------------------------------------------------------ resolve table / figure labels'):]
head = head.replace('"""Revised IJIES Paper 1 manuscript, revision 4, builder v3',
                    '"""IJIES submission version (shortened), builder v5: as builder v4 plus 1,000-permutation negative controls\n'
                    '(rev5 table), simplified abstract, spatial-baseline sentence and practical guidance. Header from rev4 v3: Revised IJIES Paper 1 manuscript, revision 4, builder v3', 1)
old_t10 = "T10 = pd.read_csv(TAB / 'Table10_negative_controls_v1.csv')"
assert head.count(old_t10) == 1
head = head.replace(old_t10, "T10 = pd.read_csv(ROOT / 'results/paper1_rev5_v1/tables/Table10_negative_controls_1000_v1.csv', dtype=str)")
head = head.replace("REFS = pd.read_csv(P1 / 'references' / 'reference_verification_table_v4.csv')",
                    "REFS = pd.read_csv(P1 / 'references' / 'reference_verification_table_v6.csv')")
assert "reference_verification_table_v6.csv')" in head and 'Table10_negative_controls_1000_v1.csv' in head
foot = foot.replace('manuscript_rev4_spec_v3.json', 'manuscript_ijies_spec_v5.json').replace('manuscript_rev4_meta_v3.json', 'manuscript_ijies_meta_v5.json')
assert 'manuscript_ijies_spec_v5.json' in foot and 'manuscript_ijies_meta_v5.json' in foot
open('build_manuscript_ijies_v5.py', 'w').write(head + b + '\n' + foot)
print('wrote ijies_body_v5.py and build_manuscript_ijies_v5.py')
