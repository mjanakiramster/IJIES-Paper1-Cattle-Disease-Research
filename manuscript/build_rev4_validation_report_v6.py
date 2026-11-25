"""Correction and validation report, v6 (composite values and capitalisation handled in the rev4 column). Earlier v5 (rev4 column checks the rev4 text directly). Earlier v4 (uses numerical check v3). Earlier v3 (table widths scaled to the page width; v2 let Table 3 run off the page). Earlier v2 (page numbers found from explicit search snippets for rows whose quoted text wraps in the PDF). Earlier v1: Correction and validation report for the final IJIES revision (rev4): Deliverable 1 (correction audit), Deliverable 3
(numerical consistency against the executed notebook v5), Deliverable 4 (additional experiments), Deliverable 5 (submission
checklist), plus reference and guideline checks. Writes a render spec and an Excel workbook with the full tables.
Run from the project root: python manuscript/build_rev4_validation_report_v1.py"""
import json, re, subprocess
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUTD = ROOT / 'manuscript'; RES = ROOT / 'results/paper1_rev4_v1'
OLD_PDF = OUTD / 'IJIES_Paper1_Manuscript_rev3_v3.pdf'
N3 = pd.read_csv(RES / 'numerical_check_rev3_v3.csv'); N4 = pd.read_csv(RES / 'numerical_check_rev4_v2.csv').set_index('ID')
REF = pd.read_csv(ROOT / 'results/paper1_v1/references/reference_verification_table_v4.csv').set_index('key')
M4 = json.load(open(OUTD / 'manuscript_rev4_meta_v3.json'))
npg = int(re.search(r'Pages:\s+(\d+)', subprocess.run(['pdfinfo', str(OLD_PDF)], capture_output=True, text=True).stdout).group(1))
PAGES = [re.sub(r'\s+', ' ', subprocess.run(['pdftotext', '-f', str(p), '-l', str(p), '-layout', str(OLD_PDF), '-'], capture_output=True, text=True).stdout) for p in range(1, npg + 1)]
def pg(snippet):
    s = re.sub(r'\s+', ' ', snippet)
    for i, p in enumerate(PAGES, 1):
        if s in p: return i
    return '-'
def tpage(loc):
    m = re.search(r'Table (\d+)', str(loc))
    return pg(f'Table {m.group(1)}.') if m else '-'
N3['Page'] = [int(p) if pd.notna(p) else tpage(l) for p, l in zip(N3.Page, N3['Manuscript location'])]
B4 = json.load(open(OUTD / 'manuscript_rev4_spec_v3.json'))['blocks']
MS4 = '\n'.join([b.get('text', '') for b in B4] + [str(c) for b in B4 if b['type'] == 'table' for rr in b['rows'] for c in rr] + [b.get('note') or '' for b in B4 if b['type'] == 'table'])
N3['In rev4'] = ['same text' if all(p.strip() in MS4 for p in str(v).split(' / ')) or str(v).lower() in MS4.lower() else ('not in rev3' if str(v).startswith('not reported') else 'reworded (Table 1)') for v in N3['Manuscript value']]

B = []
def add(t, text=None, **kw): B.append({'type': t, **({'text': text} if text is not None else {}), **kw})
def tab(cap, header, rows, widths, size=13, note=None):
    k = min(1.0, 9600 / sum(widths)); widths = [int(w * k) for w in widths]   # fit the A4 text width
    B.append({'type': 'caption', 'text': cap, 'keepNext': True}); B.append({'type': 'table', 'header': header, 'rows': [[str(c) for c in r] for r in rows], 'widths': widths, 'size': size, 'note': note})

add('title', 'IJIES Paper 1: Correction and Validation Report (revision 4)')
add('center', 'Checked: IJIES_Paper1_Manuscript_rev3_v3.pdf against IJIES_Paper1_Experiments_22sections_executed_v5.ipynb. Corrected manuscript: IJIES_Paper1_Manuscript_rev4_v3. Date: 2026-11-22.', size=19)
add('h1', 'Summary')
n = N3.Status.value_counts()
for t in [f'{len(N3)} reported results were compared with the values printed by the executed notebook. {n.get("Correct", 0)} were correct as written, {n.get("Correct but requires clarification", 0)} were correct but needed clearer wording, {n.get("Incorrect", 0)} were incorrect, and {n.get("Not verifiable", 0)} (a value quoted from another study) cannot be checked against the notebook. No number in the paper had to change.',
          'The corrections are about interpretation and labelling: the negative-control result (Holm-adjusted p = 0.059), the status of the confirmatory partition (hypotheses fixed after the development results; fold-level p = 0.069 for FMD; the second hypothesis), which estimate each number is (fold mean, pooled, within-year; bootstrap or corrected t interval), two distance aggregations, one misleading sentence about campaign overlap, and fuller statements on LSD and on the temporal test.',
          'Notebook v5 differs from v4 only by display cells that print values the paper quotes but v4 did not show (earlier-audit counts, Table 11, distances, per-disease confirmatory results, permutation resolution). It was executed from scratch in the clean copy; all 526 comparable result files are still byte-identical to those behind the paper.',
          f'References: {int((REF.loc[M4["references_cited"], "status"] == "Verified").sum())} of {len(M4["references_cited"])} cited references are fully verified. The dataset title, a DOI, an issue number and two author names were completed or corrected. One article number (reference [4]) could not be confirmed and was removed; please supply it.',
          'The IJIES template was not applied: the template file could not be read in this workspace. Formatting items are listed in the checklist as not verified.']:
    add('bullet', t)

# ---------------------------------------------------------------- Deliverable 1
add('h1', 'Deliverable 1. Correction audit')
A = [
 ('Abstract', 'Random cross-validation overstated ensemble ROC-AUC by 0.10-0.21, because 97%', 'Estimate type not named (difference of fold means); "because" stronger than the evidence.', 'Random cross-validation gave mean ensemble ROC-AUC 0.10-0.21 higher than spatial validation, mainly because 97% of its test households had training households at the same location.', '11, 21', 'Medium'),
 ('Abstract', 'FS5 raised FMD ROC-AUC from 0.563 to 0.663', 'Fold mean not named; a corrected-t CI was quoted without naming it.', 'FS5 raised mean FMD ROC-AUC from 0.563 to 0.663 (CI given in Section 4.2 with its type).', '13, 20', 'Medium'),
 ('Abstract', 'The leakage audit showed that these rates are safe, not that they are needed.', 'Permutation result missing; readers could infer full significance (item A).', 'Added: The FMD gain exceeded all label permutations, but the Holm-adjusted permutation p-value was 0.059.', '20', 'High'),
 ('Abstract', 'FS5 gave no consistent gain for CBPP or LSD.', 'LSD weakness under-reported (item D).', 'Added: ... and the LSD probabilities had no skill beyond the prevalence.', '17', 'Medium'),
 ('Abstract', 'models trained on 2012-2015 failed on 2018-2019', 'Wording (item E).', 'models trained on 2012-2015 showed no useful discrimination on 2018-2019 (ROC-AUC 0.35-0.44).', '19', 'Low'),
 ('3.3 Feature engineering', 'We call this construction leakage-safe because it follows three rules', 'The audit tests were described only in Results; leakage safety and usefulness not separated in Methods (item G).', 'New paragraph listing the five tests (reimplementation, test-label flip, training-label flip, inner-fold check, household- and village-level permutations) and stating that passing them shows safety, not usefulness.', '8, 20', 'Medium'),
 ('3.5 Spatial validation', 'The partition was then assigned, once and before opening, to two hypotheses about FS5', 'Hypotheses not stated; text could be read as an independent pre-registration (item B).', 'H1 and H2 stated; "recorded before the partition was opened but after the development results were known ... a held-out check ..., not an independent pre-registered confirmation".', 'Appendix', 'High'),
 ('3.6 Statistical evaluation', 'Holm correction [36] was applied across the diseases (and across the 12 ablation steps).', 'Holm family for permutation tests and p-value resolution missing.', 'Holm families named (3 diseases; 12 steps; 6 permutation tests); permutation p formula and minimum 0.0099 given; SHAP version added.', '1, 20', 'Medium'),
 ('4.1 Spatial generalisation', 'Random 5-fold cross-validation gave ROC-AUC of 0.79-0.83', 'Fold means not named.', '"mean fold ROC-AUC"; "raised the mean fold ROC-AUC by ...".', '11', 'Low'),
 ('4.1 Spatial generalisation', 'the median distance to the nearest training location was 35 km', 'Different from Table 9 (38 km) because the aggregation differs; not explained.', '"... 35 km (... over all test households; Table S22)"; Table 9 column renamed and defined as the fold mean of the median over test locations.', '12, 21', 'Medium'),
 ('Table 4', 'Spatial PR-AUC (no-skill)', 'PR-AUC and MCC are fold means; header did not say so.', 'Headers: "Spatial PR-AUC, fold mean (no-skill)", "MCC, fold mean".', '12', 'Medium'),
 ('4.2 Feature ablation', 'The fold-mean gain of FS5 over FS1 was +0.100 [+0.025, +0.175] (Holm p = 0.039)', 'Test behind the p-value not named.', '"+0.100 (corrected t 95% CI +0.025 to +0.175; Holm-adjusted p = 0.039 across the three diseases)".', '20', 'Low'),
 ('4.2 Feature ablation', 'The locked confirmatory partition showed the same pattern', 'Fold-level test (p = 0.069) omitted; "locked" and "registered" overstate independence (item B).', 'Pooled bootstrap CI and fold-level corrected-t p both reported; H1 not confirmed; FMD agrees in direction but its fold-level test was not significant; hypotheses fixed after development results.', 'Appendix', 'High'),
 ('4.2 Feature ablation', 'For FMD, the single FS5 ExtraTrees model was slightly better than the rank average', 'H2 (confirmatory) contradicts this and was not reported.', 'Added H2: no difference for FMD (+0.003, 95% CI -0.009 to +0.016); rank average better for LSD (-0.023, 95% CI -0.044 to -0.005).', 'Appendix', 'Medium'),
 ('4.3 Negative controls', 'this value is set by the number of permutations rather than by the size of the gain', 'Could be read as dismissing the Holm result; 0.084 vs 0.100 unexplained (item A).', 'States that the gain exceeded both nulls (p = 0.010 each), Holm-adjusted p = 0.059 is slightly above 0.05, the test does not establish the gain at 5% after correction, 1,000 permutations would allow p = 0.006; 0.084 is the repeat-1 mean (0.100 over 15 folds); LSD p-values given.', '20', 'High'),
 ('Table 8', 'Negative controls: full pipeline rebuilt on permuted labels', 'Aggregation and Holm family missing.', 'Caption: values are means over the five folds of repeat 1; note gives the p formula, minimum 0.0099 and the Holm family.', '20', 'Medium'),
 ('4.5 Spatial applicability', 'which moved the median distance to the nearest training location to 125 km', 'Table 9 shows 126 km (different aggregation).', '"... to 126 km (Table 9)"; CI type named at the start of the paragraph.', '12', 'Medium'),
 ('4.6 Survey-campaign confounding', 'The campaigns also covered different areas: between 65% and 100%', 'The statistic shows overlap, not separation.', '"The campaigns overlapped only partly in space ... but the regions surveyed differed: Amhara, Benishangul-Gumuz, Gambela and Tigray were surveyed only in 2018-2019."', '4', 'Medium'),
 ('4.6 Survey-campaign confounding', 'Location therefore carries information about the campaign', 'Risk of reading campaign signals as biology (item F).', 'Added: these signals reflect when and where the surveys were done, not biological relationships or risk factors.', '18', 'Medium'),
 ('4.7 Temporal generalisation', 'ranked households worse than chance', 'Wording; possible causes not discussed (item E).', '"did not show reliable discrimination"; new paragraph on simultaneous changes in prevalence, regions, sampling conditions and outbreak dynamics, and on what AUC below 0.5 means.', '19, 4', 'Medium'),
 ('4.7 Temporal generalisation', '(no LSD summary)', 'LSD evidence scattered; risk of over-reading LSD results (item D).', 'New paragraph: pooled 0.600, within-year 0.531, no within-year gain, gain not beyond village null, confirmatory +0.008, Brier skill -0.02 to +0.01, temporal 0.35.', '12, 17-20', 'Medium'),
 ('5 Limitations', 'The confirmatory hypotheses were written after the development results were known', 'Consequence not stated; permutation limit not linked to the Holm result.', 'Adds "held-out check, not an independent pre-registered confirmation" and that 100 permutations limit the Holm-adjusted FMD p to 0.059.', 'Appendix, 20', 'High'),
 ('6 Conclusion', 'The gain passed the negative controls, appeared in the same direction on a locked confirmatory partition', 'Overstates significance and independence (items A, B).', 'States the Holm-adjusted p (0.059) and that the held-out hypotheses were fixed after the development results.', '20, Appendix', 'High'),
 ('6 Conclusion', 'overstated ROC-AUC by 0.10-0.21 for the ensemble', 'Estimate type.', '"gave mean fold ROC-AUC 0.10-0.21 higher than spatial validation".', '11', 'Low'),
 ('Data availability', 'ILRI MELDATA (hdl:20.500.11766.1/FK2/8YFO3D, CC BY 4.0)', 'Dataset title missing.', 'Title "Transboundary cattle diseases" added (verified on the MELDATA listing).', '-', 'Low'),
 ('References [13]', '[dataset title to be inserted from the MELDATA landing page]', 'Placeholder.', 'S. Gizaw, M. Dione, B. Wieland, and H. Desta, "Transboundary cattle diseases", MELDATA, V1, 2026, hdl:20.500.11766.1/FK2/8YFO3D (CC BY 4.0).', '-', 'Medium'),
 ('References [4]', 'Vol. 54, No. 1, Art. 55, 2022', 'Article number not confirmed by any source found.', 'Article number removed; authors to supply it from the publisher page (DOI 10.1007/s11250-022-03073-2).', '-', 'Medium'),
 ('References [17], [18], [20]', 'C. Mila ...; Vol. 17, pp. 5897-5912; F. Regassa, et al.', 'Name spelling, missing issue, incomplete author list.', 'C. Milà; Vol. 17, No. 15, pp. 5897-5912; S. Tesfaye, F. Regassa, G. Beyene, S. Leta, and J. Paeshuyse.', '-', 'Low'),
]
SNIP = {'Table 4': 'Table 4.', 'Table 8': 'Table 8.', '4.6 Survey-campaign confounding': None, '4.7 Temporal generalisation': '4.7 Temporal generalisation', 'References [17], [18], [20]': '[17] C. Mila'}
def page_for(a, b):
    p = pg(b[:45])
    if p == '-' and 'covered different areas' in b: p = pg('between 65% and 100%')
    if p == '-' and SNIP.get(a): p = pg(SNIP[a])
    return p
rows = [[i + 1, a, page_for(a, b), b, c, d, e, f] for i, (a, b, c, d, e, f) in enumerate(A)]
tab('Table 1. Correction audit (page = page of the rev3 PDF; notebook = section of the executed notebook v5)', ['No.', 'Section', 'Page', 'Existing statement or result', 'Problem', 'Corrected statement', 'Notebook section', 'Priority'], rows,
    [450, 1300, 550, 2300, 1900, 2700, 900, 750], size=12)

# ---------------------------------------------------------------- Deliverable 2
add('h1', 'Deliverable 2. Corrected manuscript')
add('para', f'IJIES_Paper1_Manuscript_rev4_v3 (DOCX and PDF) is the complete revised manuscript: {M4["tables"]} tables, {M4["figures"]} figures, {M4["n_references"]} references and a {M4["abstract_words"]}-word abstract, 19 pages in the working layout. Only the statements in Table 1 were changed; all other text, tables and figures are kept from rev3. The supplement (IJIES_Paper1_Supplementary_Material_rev4_v2) changes only the notebook name, the reference table, and the notes of Tables S7 and S8.')

# ---------------------------------------------------------------- Deliverable 3
add('h1', 'Deliverable 3. Numerical consistency report')
add('para', 'Each value was read from the tables and printed text of the executed notebook (not from the result files), compared with the value printed in the manuscript, and located in the rev3 PDF. "Estimate type" names the aggregation so that fold means, pooled values, within-year values and the two kinds of interval are not mixed. The prompt\'s example is row S-fmdfs1: FMD FS1 has a fold-mean ROC-AUC of 0.563 and a pooled ROC-AUC of 0.572; both are correct and both are labelled in the paper.')
tab('Table 2. Status of the checked results', ['Status', 'Results'], [[k, v] for k, v in n.items()], [5000, 1500], size=14)
rows = [[r.ID, r['Manuscript location'], r.Page, r.Quantity, r['Estimate type'], r['Manuscript value'], r['Notebook value'], r['Notebook section'], r.Status, r['In rev4']] for _, r in N3.iterrows()]
tab('Table 3. Manuscript values (rev3) against the executed notebook (v5)', ['ID', 'Location', 'Page', 'Quantity', 'Estimate type', 'Manuscript', 'Notebook', 'NB section', 'Status', 'rev4'], rows,
    [900, 1100, 450, 1700, 1300, 1600, 1600, 900, 1050, 900], size=10)
rows = [[r.ID, r.Note] for _, r in N3[N3.Status != 'Correct'].iterrows()]
tab('Table 4. Notes for results that need clarification or cannot be verified', ['ID', 'Explanation and resolution'], rows, [1300, 8400], size=12)

# ---------------------------------------------------------------- references
add('h1', 'Reference verification')
rows = []
for i, k in enumerate(M4['references_cited'], 1):
    r = REF.loc[k]; rows.append([i, k, r.status, r.DOI_or_id, r.fields_verified])
tab('Table 5. Cited references (rev4 numbering) and what was verified', ['No.', 'Key', 'Status', 'DOI or identifier', 'Fields verified'], rows, [450, 1600, 1700, 2300, 3700], size=11,
    note='Checked against publisher pages, DOI or index records, or repository records. No reference or DOI was invented. Information needed from the authors: the article number of reference [4].')

# ---------------------------------------------------------------- Deliverable 4
add('h1', 'Deliverable 4. Additional experiments')
add('para', 'Essential corrections are the text changes in Table 1; they are implemented and need no new experiment. The experiments below would strengthen the paper. None has been run, because new experiments need your approval; each would be reported as a new, clearly labelled analysis.')
tab('Table 6. Recommended additional experiments', ['Priority', 'Experiment', 'Why it matters', 'Approximate cost'], [
    ['Recommended before submission', 'Raise the permutations to at least 1,000 per design (household and village) for the FMD and LSD gain.', 'The main FMD claim currently rests on p = 0.010 per design and a Holm-adjusted p of 0.059, which is limited by the number of permutations. With 1,000 permutations the smallest Holm-adjusted p is 0.006.', 'About 15 s per permutation run: 1,800 extra runs, about 7.5 CPU-hours (about 4 h on 2 cores); same code, checkpointed.'],
    ['Optional', 'Ablation in other orders: FS1 + rates without coordinates, and FS1 + coordinates without rates.', 'Shows whether rates and coordinates substitute for each other, which is the basis of the main interpretation.', 'About 10-20 min (two extra sets on 15 splits).'],
    ['Optional', 'SHAP and partial dependence on all three development repeats (now repeat 1 only).', 'Checks that the explanation ranking is stable across fold layouts.', 'About 20 min.'],
    ['Optional', 'Leave-one-campaign-out validation within the surveyed regions.', 'A second view of campaign transfer that does not mix time with new regions as strongly as the forward test.', 'About 30-60 min.'],
    ['Optional', 'Run the notebook on Google Colab from the code zip.', 'Confirms reproducibility on a second platform.', 'About 4 h of Colab time without checkpoints; minutes with them.'],
    ['Not recommended', 'Gridded area-of-applicability maps.', 'Needs predictor rasters that are not part of the dataset; adds data and assumptions outside the study.', '-']], [1500, 2600, 3400, 2200], size=12)

# ---------------------------------------------------------------- Deliverable 5
add('h1', 'Deliverable 5. Final submission checklist')
tab('Table 7. Submission checklist', ['Area', 'Item', 'Status'], [
    ['Scientific accuracy', 'Every major number matches the executed notebook', f'Done ({n.get("Correct", 0)} correct, {n.get("Correct but requires clarification", 0)} clarified, 0 incorrect)'],
    ['Scientific accuracy', 'No forecasting, causal, laboratory-confirmed or all-Ethiopia claims; no SGO or swarm method in the paper', 'Done (automatic text check and reading)'],
    ['Scientific accuracy', 'Weak results kept (temporal failure, LSD, CBPP, H1 not confirmed)', 'Done'],
    ['Reproducibility', 'Notebook v5 executed from scratch; outputs byte-identical (526 files)', 'Done'],
    ['Reproducibility', 'Second platform (Colab)', 'Not tested'],
    ['Statistics', 'Fold mean, pooled and within-year estimates named; bootstrap and corrected-t intervals named; Holm families stated', 'Done'],
    ['Statistics', 'Permutation test resolution', 'Stated; more permutations recommended (Deliverable 4)'],
    ['References', 'Cited references verified', f'{int((REF.loc[M4["references_cited"], "status"] == "Verified").sum())} of {len(M4["references_cited"])}; article number of [4] needed from authors'],
    ['Formatting', 'IJIES template applied', 'Not done: template file not readable in this workspace (please attach it)'],
    ['Formatting', 'Guidelines page (checked 2026-11-22): single-column from Vol. 20, No. 1 (2027); two-column papers converted by the editorial office', 'Noted; manuscript is single-column'],
    ['Formatting', 'Guidelines page: "Standard number of pages is 8"; papers must be 8 pages or more; APC depends on page count', 'Manuscript is 19 pages in the working layout; length is acceptable but affects the charge'],
    ['Formatting', 'Abstract length, keyword rules, heading and caption styles', 'Not verified: not stated on the guidelines page; template needed'],
    ['Formatting', 'Reference style: numbered, authors, "title", journal, Vol., No., pp., year', 'Matches the example shown in a copy of the IJIES format; to be rechecked against the template'],
    ['Formatting', 'Equations numbered (1), (2); tables and figures cited in order', 'Done (checked by the build script)'],
    ['Declarations', 'Conflicts of interest', 'Done'],
    ['Declarations', 'Author contributions, funding', 'Placeholders: authors to complete'],
    ['Declarations', 'Data and code availability', 'Dataset title added; repository link to be added'],
    ['Declarations', 'Declaration of AI use (IJIES requires transparency for substantial use)', 'Draft: authors to confirm'],
    ['Front matter', 'Co-author names, affiliations, corresponding e-mail', 'Placeholders: authors to complete']], [1500, 5200, 3000], size=12)

json.dump({'title': 'Correction and Validation Report', 'running_head': 'IJIES Paper 1: correction and validation report (rev4)', 'blocks': B}, open(OUTD / 'rev4_validation_report_spec_v6.json', 'w'), indent=1, ensure_ascii=False)
with pd.ExcelWriter(RES / 'IJIES_Paper1_rev4_validation_tables_v6.xlsx') as W:
    pd.DataFrame([dict(zip(['No.', 'Section', 'Page (rev3)', 'Existing statement', 'Problem', 'Corrected statement', 'Notebook section', 'Priority'], r)) for r in [[i + 1, a, page_for(a, b), b, c, d, e, f] for i, (a, b, c, d, e, f) in enumerate(A)]]).to_excel(W, sheet_name='1 Correction audit', index=False)
    N3.to_excel(W, sheet_name='3 Numerical consistency', index=False)
    REF.loc[M4['references_cited']].reset_index().to_excel(W, sheet_name='References', index=False)
print(n.to_dict(), 'audit rows', len(A))
