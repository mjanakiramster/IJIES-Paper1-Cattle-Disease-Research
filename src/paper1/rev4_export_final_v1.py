"""Revision 4 export (same procedure as rev3_export_final_v1.py, rev4 specs). Exports the tables and figures of revision 3 with manuscript numbering (Phase 7 package).
Figures and tables are taken from the manuscript and supplement specs, so each exported file is exactly what the paper shows:
each figure is copied (PNG 300 dpi + PDF) under its printed number, each table is written to CSV, and all tables go to one
Excel workbook. Run from the project root: python src/paper1/rev3_export_final_v1.py <output folder>"""
import sys, json, re, shutil
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; OUT = Path(sys.argv[1]); (OUT / 'tables').mkdir(parents=True, exist_ok=True); (OUT / 'figures').mkdir(parents=True, exist_ok=True)
SPECS = ('manuscript/manuscript_rev4_spec_v3.json', 'manuscript/supplement_rev4_spec_v2.json')
sheets, figs = {}, []
for spec in SPECS:
    B = json.load(open(ROOT / spec))['blocks']; cap = None; pending = None
    for b in B:
        if b['type'] == 'figure': pending = Path(b['path'])
        elif b['type'] == 'caption' and pending is not None and re.match(r'\*\*Figure S?\d+\.\*\*', b['text']):
            num = re.match(r'\*\*Figure (S?\d+)\.\*\*', b['text']).group(1); name = f'Figure{num}'
            for ext in ('png', 'pdf'):
                src = pending.with_suffix(f'.{ext}')
                if src.exists(): shutil.copy(src, OUT / 'figures' / f'{name}.{ext}')
            figs.append({'figure': name, 'source file': str(pending.relative_to(ROOT).with_suffix('')) + '.png/.pdf', 'caption': re.sub(r'\*\*Figure S?\d+\.\*\*\s*', '', b['text'])}); pending = None
        elif b['type'] == 'caption' and re.match(r'\*\*Table S?\d+\.\*\*', b['text']): cap = b['text']
        elif b['type'] == 'table' and cap:
            num = re.match(r'\*\*Table (S?\d+)\.\*\*', cap).group(1); title = re.sub(r'\*\*Table S?\d+\.\*\*\s*', '', cap)
            df = pd.DataFrame(b['rows'], columns=b['header']); name = f'Table{num}'
            df.to_csv(OUT / 'tables' / f'{name}.csv', index=False); sheets[name] = (title, df, b.get('note')); cap = None
pd.DataFrame(figs).to_csv(OUT / 'figures' / 'figure_source_map.csv', index=False)
with pd.ExcelWriter(OUT / 'tables' / 'IJIES_Paper1_all_tables_rev4.xlsx') as W:
    pd.DataFrame([{'Sheet': k, 'Caption': v[0], 'Note': v[2] or ''} for k, v in sheets.items()]).to_excel(W, sheet_name='Contents', index=False)
    for k, (title, df, note) in sheets.items(): df.to_excel(W, sheet_name=k, index=False)
print(len(figs), 'figures;', len(sheets), 'tables:', ', '.join(sheets))
