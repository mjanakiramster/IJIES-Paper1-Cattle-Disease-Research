// Renders manuscript JSON (built by build_manuscript_v1.py) into an editable Word file. Usage: node render_docx_v1.js in.json out.docx
const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType, HeadingLevel,
        BorderStyle, ShadingType, PageNumber, Footer, Header, LevelFormat, TabStopType } = require('docx');
const spec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const FONT = 'Times New Roman';
const PAGE_W = 11906, MARGIN = 1134;                 // A4, 2 cm margins
const TEXT_W = PAGE_W - 2 * MARGIN;

function runs(text, base = {}) {
  // supports **bold**, *italic*, ^{superscript} and _{subscript} inline markers (v2 adds super/subscript)
  const out = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^\{[^}]+\}|_\{[^}]+\})/g; let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith('^{')) out.push(new TextRun({ text: t.slice(2, -1), superScript: true, font: FONT, ...base }));
    else if (t.startsWith('_{')) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' };
function table(b) {
  const ncol = b.header.length; const widths = b.widths || Array(ncol).fill(Math.floor(TEXT_W / ncol));
  const total = widths.reduce((a, c) => a + c, 0);
  const mk = (cells, head) => new TableRow({ tableHeader: head, cantSplit: true, children: cells.map((c, i) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders: { top: border, bottom: border, left: border, right: border },
    shading: head ? { fill: 'E8EEF6', type: ShadingType.CLEAR, color: 'auto' } : undefined, margins: { top: 40, bottom: 40, left: 60, right: 60 },
    children: [new Paragraph({ spacing: { before: 0, after: 0 }, alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(c), { size: b.size || 16, bold: !!head }) })] })) });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [mk(b.header, true), ...b.rows.map(r => mk(r, false))] });
}
const children = [];
for (const b of spec.blocks) {
  if (b.type === 'title') children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: runs(b.text, { size: 30, bold: true }) }));
  else if (b.type === 'center') children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: runs(b.text, { size: b.size || 20 }) }));
  else if (b.type === 'h1') children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 100 }, keepNext: true, children: runs(b.text, { size: 24, bold: true }) }));
  else if (b.type === 'h2') children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 80 }, keepNext: true, children: runs(b.text, { size: 22, bold: true, italics: true }) }));
  else if (b.type === 'para') children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 264 }, children: runs(b.text, { size: 22 }) }));
  else if (b.type === 'small') children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 }, children: runs(b.text, { size: 18 }) }));
  else if (b.type === 'bullet') children.push(new Paragraph({ numbering: { reference: 'bullets', level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60 }, children: runs(b.text, { size: 22 }) }));
  else if (b.type === 'equation') children.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 60, after: 100 },
      tabStops: [{ type: TabStopType.CENTER, position: Math.floor(TEXT_W / 2) }, { type: TabStopType.RIGHT, position: TEXT_W }],
      children: [new TextRun({ text: '\t', font: FONT }), ...runs(b.text, { size: 22, italics: true }), new TextRun({ text: `\t(${b.number})`, font: FONT, size: 22 })] }));
  else if (b.type === 'caption') children.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 120, after: 80 }, keepNext: !!b.keepNext, children: runs(b.text, { size: 19 }) }));
  else if (b.type === 'table') { children.push(table(b)); if (b.note) children.push(new Paragraph({ spacing: { before: 40, after: 160 }, children: runs(b.note, { size: 16 }) })); else children.push(new Paragraph({ spacing: { after: 120 }, children: [] })); }
  else if (b.type === 'figure') {
    const img = fs.readFileSync(b.path); const w = b.width_px, h = Math.round(b.width_px * b.aspect);
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 }, children: [new ImageRun({ type: 'png', data: img, transformation: { width: w, height: h },
      altText: { title: b.alt || 'figure', description: b.alt || 'figure', name: b.alt || 'figure' } })] }));
  }
  else if (b.type === 'reference') children.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 50 }, indent: { left: 400, hanging: 400 }, children: runs(b.text, { size: 18 }) }));
  else if (b.type === 'pagebreak') children.push(new Paragraph({ pageBreakBefore: true, children: [] }));
}
const doc = new Document({
  creator: 'Paper 1 manuscript builder', title: spec.title,
  styles: { default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 22, bold: true, italics: true }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 1 } }] },
  numbering: { config: [{ reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] }] },
  sections: [{ properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: runs(spec.running_head || '', { size: 16, color: '666666' }) })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children }] });
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[3], buf); console.log('wrote', process.argv[3]); });
