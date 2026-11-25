"""Builds the rev5 code-and-results package (new file; the rev4 zip is not changed).
Starts from the exact contents of the rev4 package (deliverables/IJIES_Paper1_v4/IJIES_Paper1_code_and_results.zip) and adds
the rev5 code, results, notebook v6, IJIES manuscript builders and README. Usage, from the project root:
  python src/paper1/rev5_package_v1.py <output zip>"""
import sys, zipfile, shutil, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUTZIP = Path(sys.argv[1]); TOP = 'IJIES_Paper1_code_and_results'
OLD = ROOT / 'deliverables/IJIES_Paper1_v4/IJIES_Paper1_code_and_results.zip'
ADD = ['src/paper1/rev5_perm_extension_v1.py', 'src/paper1/rev5_perm_tables_v1.py', 'src/paper1/fig5_permutation_v4.py', 'src/paper1/fig5_permutation_v5.py',
       'src/paper1/make_notebook_v6.py', 'src/paper1/build_notebook_rev3_v6.py', 'src/paper1/rev5_package_v1.py',
       'IJIES_Paper1_Experiments_22sections_unexecuted_v6.ipynb', 'IJIES_Paper1_Experiments_22sections_executed_v6.ipynb',
       'manuscript/build_manuscript_ijies_v5.py', 'manuscript/ijies_body_v5.py', 'manuscript/make_ijies_builder_v5.py',
       'manuscript/build_supplement_ijies_v2.py', 'manuscript/make_supplement_ijies_v2.py',
       'manuscript/manuscript_ijies_spec_v5.json', 'manuscript/manuscript_ijies_meta_v5.json', 'manuscript/supplement_ijies_spec_v2.json',
       'results/paper1_v1/references/reference_verification_table_v6.csv']
ADD_DIRS = ['results/paper1_rev5_v1']
tmp = Path(tempfile.mkdtemp()); zipfile.ZipFile(OLD).extractall(tmp); base = tmp / TOP
n_new = 0
for rel in ADD:
    src = ROOT / rel; assert src.exists(), f'missing {rel}'
    dst = base / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); n_new += 1
for rel in ADD_DIRS:
    for src in (ROOT / rel).rglob('*'):
        if src.is_file() and src.suffix != '.tmp':
            dst = base / src.relative_to(ROOT); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); n_new += 1
shutil.copy2(ROOT / 'README_paper1_rev5_v1.md', base / 'README.md')
OUTZIP.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUTZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(base.rglob('*')):
        if p.is_file(): z.write(p, p.relative_to(tmp))
print('package', OUTZIP, '| files added or replaced:', n_new + 1, '| total files:', sum(1 for p in base.rglob('*') if p.is_file()))
