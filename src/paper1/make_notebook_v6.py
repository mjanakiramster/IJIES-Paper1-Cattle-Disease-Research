"""Makes the notebook builder v6 from v5 (approved 2026-11-22): adds the 900 extra permutations per scheme
(rev5_perm_extension_v1, computed if missing, checkpointed), the 1,000-permutation table (rev5_perm_tables_v1, checked
against the stored file) and Supplementary Figure S6 v5. Everything else is unchanged; the 100-permutation table and
figure are still shown. Writes a new file only: src/paper1/build_notebook_rev3_v6.py. Run from the project root."""
from pathlib import Path
src = Path('src/paper1/build_notebook_rev3_v5.py').read_text()


def rep(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old[:90])
    src = src.replace(old, new)


rep('"""Builds the notebook in the 22 sections requested for the study, v5:',
    '"""Builds the notebook in the 22 sections requested for the study, v6: permutation negative controls extended to\n'
    '1,000 per scheme (900 added permutations computed if missing; new table checked against the stored file; Figure S6 v5).\n'
    'Earlier, v5:')
rep("    if p.exists(): display(Image(filename=str(p), width=width)); return",
    "    if p.exists(): display(Image(filename=str(p), width=width)); return\n"
    "    p = ROOT / 'results/paper1_rev5_v1/figures' / f'{name}.png'\n"
    "    if p.exists(): display(Image(filename=str(p), width=width)); return")
rep("print(counts); check(counts['E5_permutations'] >= 201 and counts['E11_ablation'] >= 15, 'all experiment checkpoints present')",
    "print(counts); check(counts['E5_permutations'] >= 201 and counts['E11_ablation'] >= 15, 'all experiment checkpoints present')\n"
    "from paper1 import rev5_perm_extension_v1 as PX   # rev5: 900 more permutations per scheme (about 4 hours on 2 cores if missing)\n"
    "for s_ in ('household', 'village'):\n"
    "    for k_ in range(100, 1000): PX.part(s_, k_)\n"
    "n_ext = len(list(PX.E5X.glob('*.csv'))); print('added permutations:', n_ext); check(n_ext == 1800, '900 added permutations per scheme present')")
rep("n_perm = 100; print(f'smallest attainable permutation p with {n_perm} permutations: 1/{n_perm + 1} = {1 / (n_perm + 1):.4f}; Holm over 6 tests multiplies the smallest p by 6: {6 / (n_perm + 1):.4f}')",
    "n_perm = 100; print(f'smallest attainable permutation p with {n_perm} permutations: 1/{n_perm + 1} = {1 / (n_perm + 1):.4f}; Holm over 6 tests multiplies the smallest p by 6: {6 / (n_perm + 1):.4f}')\n"
    "from paper1 import rev5_perm_tables_v1 as PT   # rev5: 1,000 permutations per scheme (manuscript Table 4, Supplementary Table S8)\n"
    "T1000, _ = PT.table(); display(T1000); check((T1000.Permutations == 1000).all(), '1,000 permutations per test; stored table reproduced')\n"
    "runpy.run_path('src/paper1/fig5_permutation_v5.py'); show('Fig5_permutation_null_1000_v2')\n"
    "n_perm = 1000; print(f'smallest attainable permutation p with {n_perm} permutations: 1/{n_perm + 1} = {1 / (n_perm + 1):.4f}; Holm over 6 tests multiplies the smallest p by at most 6: {6 / (n_perm + 1):.4f}')")
rep("out = ROOT / 'IJIES_Paper1_Experiments_22sections_unexecuted_v5.ipynb'", "out = ROOT / 'IJIES_Paper1_Experiments_22sections_unexecuted_v6.ipynb'")
Path('src/paper1/build_notebook_rev3_v6.py').write_text(src)
print('wrote src/paper1/build_notebook_rev3_v6.py')
