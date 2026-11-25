"""Supplement builder v2 for the IJIES version (approved 2026-11-22): as v1, with Table S8 and Figure S6 from the
1,000-permutation negative controls (rev5) and the executed notebook v6 named. Writes a new file only:
manuscript/build_supplement_ijies_v2.py (-> supplement_ijies_spec_v2.json). Run from the manuscript folder."""
s = open('build_supplement_ijies_v1.py').read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('"""Supplementary Material for the shortened IJIES version, builder v1:',
    '"""Supplementary Material for the shortened IJIES version, builder v2: Table S8 and Figure S6 with 1,000 permutations per scheme\n'
    '(rev5 table and figure); executed notebook v6 named. Builder v1:')
rep('executed notebook IJIES_Paper1_Experiments_22sections_executed_v5 (22 sections plus an appendix)',
    'executed notebook IJIES_Paper1_Experiments_22sections_executed_v6 (22 sections plus an appendix)')
rep("T10 = pd.read_csv(TAB / 'Table10_negative_controls_v1.csv')",
    "T10 = pd.read_csv(ROOT / 'results/paper1_rev5_v1/tables/Table10_negative_controls_1000_v1.csv', dtype=str)\n"
    "T10 = T10[['Scheme', 'Disease', 'Permutations', 'Null FS1 ROC-AUC mean [2.5-97.5%]', 'Null FS5 ROC-AUC mean [2.5-97.5%]', 'Null gain mean [2.5-97.5%]', 'Observed gain',\n"
    "           'Permutation p (resolution)', 'Null FS5 PR-AUC', 'No-skill PR-AUC', 'Holm p (6 tests)']]")
rep("note='Permutation p = (1 + null gains at least as large as the observed gain) / (1 + 100); the smallest attainable value is 0.0099. Holm correction across the six tests (three diseases × two schemes).')",
    "note=f'Permutation p = (1 + null gains at least as large as the observed gain) / (1 + {int(T10.Permutations.min()):,}); the smallest attainable value is {1 / (1 + int(T10.Permutations.min())):.4f}. The first 100 permutations per scheme are those of the original analysis (Holm-adjusted FMD p = 0.059 with 100); the other 900 were added with the same code and seed rule. Holm correction across the six tests (three diseases × two schemes).')")
rep("fig('Fig5_permutation_null_v3', '**Figure S6.** Null distributions of the FS5 minus FS1 ROC-AUC under household-level and village-level label permutation; the orange line and the text in each panel give the observed value and its permutation p-value (development repeat 1).')",
    "fig('Fig5_permutation_null_1000_v2', '**Figure S6.** Null distributions of the FS5 minus FS1 ROC-AUC under household-level and village-level label permutation (1,000 permutations per scheme); the orange line and the text in each panel give the observed value and its permutation p-value (development repeat 1).', folder=ROOT / 'results/paper1_rev5_v1/figures')")
rep("open(ROOT / 'manuscript' / 'supplement_ijies_spec_v1.json', 'w')", "open(ROOT / 'manuscript' / 'supplement_ijies_spec_v2.json', 'w')")
open('build_supplement_ijies_v2.py', 'w').write(s)
print('wrote build_supplement_ijies_v2.py')
