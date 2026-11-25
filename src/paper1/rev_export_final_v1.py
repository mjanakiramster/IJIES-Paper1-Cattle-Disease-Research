"""Exports the final tables and figures of the revised paper with manuscript numbering.
Tables are taken from the rendered manuscript and supplement specs, so each CSV is exactly the table printed in the paper;
all of them are also written to one Excel workbook. Figures are copied (PNG 300 dpi + PDF) under their manuscript numbers.
Run from the project root: python src/paper1/rev_export_final_v1.py <output folder>"""
import sys, json, re, shutil
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; OUT = Path(sys.argv[1]); (OUT / 'tables').mkdir(parents=True, exist_ok=True); (OUT / 'figures').mkdir(parents=True, exist_ok=True)
P1F = ROOT / 'results/paper1_v1/figures'; RF = ROOT / 'results/paper1_rev_v1/figures'
FIGS = {'Figure1_workflow': RF / 'Fig1_workflow_v6', 'Figure2_random_vs_spatial_cv': RF / 'Fig3_random_vs_spatial_cv_v1', 'Figure3_FS1_vs_FS5_paired_folds': RF / 'Fig4_FS1_vs_FS5_paired_folds_v2',
        'Figure4_survey_locations_and_prevalence': P1F / 'Fig2_survey_years_v2', 'Figure5_overall_vs_within_year_AUC': P1F / 'Fig3_overall_vs_within_year_AUC_v1',
        'Figure6_campaign_specific_vs_pooled': RF / 'Fig6_campaign_specific_vs_pooled_v3', 'Figure7_permutation_nulls': P1F / 'Fig5_permutation_null_v3',
        'Figure8_spatial_sensitivity': P1F / 'Fig4_spatial_sensitivity_v2', 'Figure9_forward_temporal': RF / 'Fig8_forward_temporal_v1',
        'FigureS1_ROC_PR_curves': P1F / 'FigS1_ROC_PR_curves_v1', 'FigureS2_calibration': P1F / 'FigS2_calibration_v2', 'FigureS3_confusion_matrices': P1F / 'FigS3_confusion_matrices_v1',
        'FigureS4_grouped_importance': P1F / 'FigS4_grouped_importance_v2'}
src_map = []
for new, old in FIGS.items():
    for ext in ('png', 'pdf'): shutil.copy(f'{old}.{ext}', OUT / 'figures' / f'{new}.{ext}')
    src_map.append({'manuscript figure': new, 'source file': str(old.relative_to(ROOT)) + '.png/.pdf'})
pd.DataFrame(src_map).to_csv(OUT / 'figures' / 'figure_source_map.csv', index=False)
sheets = {}
for spec in ('manuscript/manuscript_rev_spec_v2.json', 'manuscript/supplement_rev_spec_v2.json'):
    B = json.load(open(ROOT / spec))['blocks']; cap = None
    for b in B:
        if b['type'] == 'caption' and re.match(r'\*\*Table S?\d+\.\*\*', b['text']): cap = b['text']
        elif b['type'] == 'table' and cap:
            num = re.match(r'\*\*Table (S?\d+)\.\*\*', cap).group(1); title = re.sub(r'\*\*Table S?\d+\.\*\*\s*', '', cap)
            df = pd.DataFrame(b['rows'], columns=b['header']); name = f'Table{num}'
            df.to_csv(OUT / 'tables' / f'{name}.csv', index=False); sheets[name] = (title, df, b.get('note')); cap = None
with pd.ExcelWriter(OUT / 'tables' / 'IJIES_Paper1_all_tables.xlsx') as W:
    pd.DataFrame([{'Sheet': k, 'Caption': v[0], 'Note': v[2] or ''} for k, v in sheets.items()]).to_excel(W, sheet_name='Contents', index=False)
    for k, (title, df, note) in sheets.items(): df.to_excel(W, sheet_name=k, index=False)
print(len(FIGS), 'figures;', len(sheets), 'tables:', ', '.join(sheets))
