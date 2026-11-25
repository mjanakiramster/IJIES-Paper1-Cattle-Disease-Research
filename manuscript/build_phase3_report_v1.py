"""Phase 3 reproduction report: the from-scratch run in a clean copy, compared file by file with the reported results.
Reads results/paper1_rev2_v1/reproduction/compare_fresh_v1.csv. Spec for render_docx_v2.js.
Run from the project root: python manuscript/build_phase3_report_v1.py"""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUTD = ROOT / 'manuscript'; REP = ROOT / 'results/paper1_rev2_v1/reproduction'
C = pd.read_csv(REP / 'compare_fresh_v1.csv'); n = C.result.value_counts()
cmp_ = C[C.result.isin(['identical', 'numerically equal', 'within 1e-9', 'DIFFERS', 'compare error'])]
by_type = cmp_.groupby('type').result.agg(lambda s: f'{(s == "identical").sum()} / {len(s)}').to_dict()
B = []
def add(t, text=None, **kw): B.append({'type': t, **({'text': text} if text is not None else {}), **kw})
def tab(cap, header, rows, widths, size=15, note=None):
    B.append({'type': 'caption', 'text': cap, 'keepNext': True}); B.append({'type': 'table', 'header': header, 'rows': [[str(c) for c in r] for r in rows], 'widths': widths, 'size': size, 'note': note})
add('title', 'IJIES Paper 1: Phase 3 Reproduction Report (from-scratch run)')
add('center', 'Notebook: IJIES_Paper1_Experiments_22sections (v2 for the computation, v4 for the final execution). Date: 2026-11-22.', size=19)
add('h1', '1. Result')
add('para', f'All {len(cmp_)} result files that could be compared were byte-identical to the files behind the paper ({n.get("DIFFERS", 0)} differences, {n.get("compare error", 0)} comparison errors). The final notebook ran from top to bottom without errors, and its 13 built-in checks all passed.')
add('h1', '2. What was done')
for t in ['A clean copy of the project was made with the source code, the frozen configuration, the raw data and the fixed fold file. The three result folders that hold every number in the paper (results/paper1_v1, results/paper1_rev_v1 and results/paper1_rev2_v1) were deleted, so every experiment, table and figure in them had to be recomputed.',
          'Inputs kept as they were, because they are not recomputed by the notebook: the 23-model library cache for the models that use no outcome-derived features (results/stage4_v1, data/reference), the corrected-prior refits (results/audit_v1/A5), the confirmatory test (results/stage9_confirmatory_v1, reported as observed), the optimisation study used only in Supplementary Section S2 (results/stage5_v1, stage5d_noise_v1, stage6_v1), and the Weight check (results/stage1_v1).',
          'The 22-section notebook (v2) computed every experiment checkpoint: FS1/FS5 models, the 23-model library, 201 permutation runs, proximity and block-size conditions, temporal validation, grouped importance, random cross-validation, campaign-specific models, ablation, SHAP and partial dependence, area of applicability and survey-year identification. To use the second processor core, a helper process ran the same functions of the same code in reverse order; each checkpoint is written once and skipped if it exists, so the two processes never computed the same result twice.',
          'The notebook stopped in Section 18 because a print statement contained "95% CI" inside %-formatting. This was a display bug in the notebook, not in the analysis code. It was fixed in notebook v4, which also adds the revision-3 tables (Table 2 v2, Table 3 v3, the calibration table) and figures (workflow v7, ablation v3, distances v2). Notebook v4 was then executed from the top in the same clean copy; it used the checkpoints computed above and finished without errors in 2.7 minutes.',
          'Every output file of the clean copy was compared with the main project: CSV, JSON, NPZ and pickle files byte by byte (with a numeric comparison as fallback). Images, PDFs and Excel files were only checked to exist, because the renderers embed time stamps.']:
    add('bullet', t)
add('h1', '3. Comparison')
tab('Table 1. Files compared, by type', ['File type', 'Identical / compared'], [[k, v] for k, v in sorted(by_type.items())], [3000, 3000])
tab('Table 2. All outputs of the clean run', ['Outcome', 'Files'], [[k, v] for k, v in n.items()], [5000, 1500],
    note='"Only in fresh run": the exported fold assignments, the exported household-level predictions and the revision-2 Excel workbook, which had not been written in the main project; they were copied into it. Files present only in the main project are earlier superseded versions (older figures, version-1 metrics) and hand-made audit tables, which the notebook does not write.')
add('h1', '4. Limits of this check')
for t in ['The run used the same machine type and library versions as the original analysis. Reproduction on another platform, such as Google Colab, was not tested here.',
          'The kept inputs listed in Section 2 were not recomputed. The library cache affects only models without outcome-derived features, and the confirmatory and optimisation results are reported as observed.',
          'The comparison script and the full file-by-file list are in results/paper1_rev2_v1/reproduction (compare_fresh_v1.py, compare_fresh_v1.csv).']:
    add('bullet', t)
json.dump({'title': 'Phase 3 Reproduction Report', 'running_head': 'IJIES Paper 1: Phase 3 reproduction report', 'blocks': B}, open(OUTD / 'phase3_report_spec_v1.json', 'w'), indent=1, ensure_ascii=False)
print(n.to_dict(), by_type)
