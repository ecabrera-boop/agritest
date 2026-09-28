"""
Build an editable Word (.docx) copy of the user manual.

Run from the project root, after generate_user_manual.py has produced the
screenshots (docs/manual_screenshots/_shots.json):

    python docs/generate_user_manual_docx.py

The manual text lives only in generate_user_manual.build_pdf(). This script
re-executes that content section with Word-emitting helpers, so the PDF and
the DOCX always carry the same text and figures.
"""
from __future__ import annotations

import inspect
import json
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image as PILImage

DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR))
import generate_user_manual as gm  # noqa: E402

GREEN = RGBColor(0x19, 0x87, 0x54)
DARK_GREEN = RGBColor(0x1B, 0x43, 0x32)
GREY = RGBColor(0x6C, 0x75, 0x7D)
PAGE_W_CM = 21.0 - 3.4          # A4 minus 1.7 cm margins each side


# ═══════════════════════════════════════════════════════════════════════════
# python-docx helpers
# ═══════════════════════════════════════════════════════════════════════════

def shade(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def cell_border(cell, side: str, hex_color: str, size: int = 24) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    el = OxmlElement(f"w:{side}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(size))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), hex_color)
    borders.append(el)


def set_col_widths(table, widths_cm: list[float]) -> None:
    table.autofit = False
    for row in table.rows:
        for idx, width in enumerate(widths_cm):
            row.cells[idx].width = Cm(width)


def add_field(paragraph, instr: str, placeholder: str = "") -> None:
    """Insert a Word field (PAGE, TOC …) into a paragraph."""
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    run._r.append(begin)
    run._r.append(instr_el)
    run._r.append(sep)
    if placeholder:
        paragraph.add_run(placeholder)
    end_run = paragraph.add_run()
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    end_run._r.append(end)


_ENTITIES = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&nbsp;": " ", "&quot;": '"'}
_TOKEN = re.compile(r"(</?b>|</?i>|</?strong>|</?em>|<br\s*/?>)")


def add_markup(paragraph, text: str, size: float | None = None, color: RGBColor | None = None,
               font: str | None = None) -> None:
    """Render reportlab-style inline markup (<b>, <i>, <br/>, entities) as Word runs."""
    bold = italic = False
    for token in _TOKEN.split(text):
        if not token:
            continue
        low = token.lower()
        if low in ("<b>", "<strong>"):
            bold = True
        elif low in ("</b>", "</strong>"):
            bold = False
        elif low in ("<i>", "<em>"):
            italic = True
        elif low in ("</i>", "</em>"):
            italic = False
        elif low.startswith("<br"):
            paragraph.add_run().add_break(WD_BREAK.LINE)
        else:
            for ent, ch in _ENTITIES.items():
                token = token.replace(ent, ch)
            run = paragraph.add_run(token)
            run.bold = bold
            run.italic = italic
            if size:
                run.font.size = Pt(size)
            if color is not None:
                run.font.color.rgb = color
            if font:
                run.font.name = font
                run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def plain(text: str) -> str:
    text = _TOKEN.sub("", text)
    for ent, ch in _ENTITIES.items():
        text = text.replace(ent, ch)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Document skeleton
# ═══════════════════════════════════════════════════════════════════════════

def new_document(brand: str) -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.7)
    sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(2.0)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for name, size, color, before, after in (
        ("Heading 1", 20, GREEN, 6, 10),
        ("Heading 2", 14, DARK_GREEN, 12, 6),
        ("Heading 3", 11.5, DARK_GREEN, 8, 4),
    ):
        st = styles[name]
        st.font.name = "Arial"
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True

    cap = styles.add_style("Figure Caption", 1)
    cap.base_style = normal
    cap.font.size = Pt(9)
    cap.font.color.rgb = GREY
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after = Pt(10)

    code = styles.add_style("Code Block", 1)
    code.base_style = normal
    code.font.name = "Courier New"
    code.element.rPr.rFonts.set(qn("w:eastAsia"), "Courier New")
    code.font.size = Pt(9)
    code.paragraph_format.left_indent = Cm(0.4)

    # Footer: brand on the left, page number on the right
    footer = sec.footer
    fp = footer.paragraphs[0]
    # The Footer style already has centre + right tab stops; jump past the centre one.
    run = fp.add_run(f"{brand} — User Manual\t\t")
    run.font.size = Pt(8.5)
    run.font.color.rgb = GREY
    r2 = fp.add_run("Page ")
    r2.font.size = Pt(8.5)
    r2.font.color.rgb = GREY
    add_field(fp, "PAGE", "1")
    for r in fp.runs:
        r.font.size = Pt(8.5)
        r.font.color.rgb = GREY
    p_pr = fp._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    top = OxmlElement("w:top")
    top.set(qn("w:val"), "single")
    top.set(qn("w:sz"), "4")
    top.set(qn("w:space"), "4")
    top.set(qn("w:color"), "D0E3D7")
    bdr.append(top)
    p_pr.append(bdr)

    doc.core_properties.title = f"{brand} User Manual"
    doc.core_properties.author = f"{brand} — Rice Leaf Disease Detector"
    doc.core_properties.subject = f"How to use the {brand} rice leaf disease detection system"
    return doc


class NumberingFactory:
    """Creates a fresh numbered list (restarting at 1) for every steps() call."""

    def __init__(self, doc: Document):
        self.doc = doc
        self.numbering = doc.part.numbering_part.numbering_definitions._numbering
        style = doc.styles["List Number"]
        num_id = style.element.pPr.numPr.numId.val
        self.abstract_id = None
        for num in self.numbering.findall(qn("w:num")):
            if num.get(qn("w:numId")) == str(num_id):
                self.abstract_id = num.find(qn("w:abstractNumId")).get(qn("w:val"))
                break

    def new_list(self) -> int:
        existing = [int(n.get(qn("w:numId"))) for n in self.numbering.findall(qn("w:num"))]
        new_id = max(existing, default=0) + 1
        num = OxmlElement("w:num")
        num.set(qn("w:numId"), str(new_id))
        abs_el = OxmlElement("w:abstractNumId")
        abs_el.set(qn("w:val"), str(self.abstract_id))
        num.append(abs_el)
        override = OxmlElement("w:lvlOverride")
        override.set(qn("w:ilvl"), "0")
        start = OxmlElement("w:startOverride")
        start.set(qn("w:val"), "1")
        override.append(start)
        num.append(override)
        self.numbering.append(num)
        return new_id


# ═══════════════════════════════════════════════════════════════════════════
# Emitters that mirror build_pdf()'s helper API
# ═══════════════════════════════════════════════════════════════════════════

def build_docx(shots: dict, workflow: Path, photo_tips: Path, diseases: list[dict], out: Path) -> Path:
    brand = gm.BRAND
    doc = new_document(brand)
    numbering = NumberingFactory(doc)
    fig_no = [0]

    def B(text: str) -> str:
        return str(text).replace("RiceGuard", brand)

    # ── style tokens used by the content section ───────────────────────
    STYLE_MAP = {
        "H1": "Heading 1", "H2": "Heading 2", "H3": "Heading 3",
        "Body": "Normal", "Small": "Normal", "Meta": "Normal", "Sub": "Normal",
        "DocTitle": "Normal", "um": "Normal", "code": "Code Block", "Caption": "Figure Caption",
    }

    class Style(str):
        pass

    def ParagraphStyle(name, parent=None, **kw):
        return Style(name)

    body, small, h1, h2, h3 = Style("Body"), Style("Small"), Style("H1"), Style("H2"), Style("H3")
    caption, bullet, step_style = Style("Caption"), Style("Bullet"), Style("Step")
    cell, cell_b, title_style, subtitle, meta = Style("Cell"), Style("CellB"), Style("DocTitle"), Style("Sub"), Style("Meta")

    def add_par(text, style: str):
        text = B(text)
        name = STYLE_MAP.get(str(style), "Normal")
        p = doc.add_paragraph(style=name)
        s = str(style)
        if s == "DocTitle":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_markup(p, text, size=30, color=GREEN)
            for r in p.runs:
                r.bold = True
        elif s == "Sub":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_markup(p, text, size=15, color=DARK_GREEN)
            for r in p.runs:
                r.bold = True
        elif s == "um":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_markup(p, text, size=22, color=GREEN)
            for r in p.runs:
                r.bold = True
        elif s == "Meta":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_markup(p, text, color=GREY)
        elif s == "Small":
            add_markup(p, text, size=9)
        elif s == "code":
            add_markup(p, text)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:fill"), "F1F3F5")
            p._p.get_or_add_pPr().append(shd)
        elif s == "H1" and plain(text) == "Contents":
            # Not a real heading, otherwise the TOC would list itself.
            p.style = doc.styles["Normal"]
            add_markup(p, text, size=20, color=GREEN)
            for r in p.runs:
                r.bold = True
        elif s in ("H1", "H2", "H3"):
            p.add_run(plain(text))
        else:
            add_markup(p, text)
        return p

    class Paragraph:
        def __init__(self, text, style, **kw):
            self.text, self.style = text, style

    class Spacer:
        def __init__(self, w, h):
            self.h = h

    class PageBreak:
        pass

    class Image:
        def __init__(self, path):
            self.path = Path(path)
            with PILImage.open(self.path) as im:
                self.drawWidth, self.drawHeight = im.size
            self.hAlign = "LEFT"

    class TableOfContents:
        def __init__(self):
            self.levelStyles = []

    class Story:
        def append(self, item):
            if isinstance(item, PageBreak):
                doc.add_page_break()
            elif isinstance(item, Spacer):
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(max(item.h, 0) * 28.35 if item.h < 20 else item.h)
            elif isinstance(item, Paragraph):
                add_par(item.text, item.style)
            elif isinstance(item, Image):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(item.path), width=Cm(item.drawWidth))
            elif isinstance(item, TableOfContents):
                p = doc.add_paragraph()
                add_field(p, 'TOC \\o "1-2" \\h \\z \\u',
                          "Right-click here and choose Update Field to build the table of contents.")
            # KeepTogether and other layout-only flowables are ignored.

    story = Story()

    class colors:
        @staticmethod
        def HexColor(v):
            return v

    cm = 1.0                    # content section multiplies lengths by cm → values become cm
    PAGE_W = PAGE_W_CM

    def P(text, style=body):
        add_par(text, style)

    def bullets(items, style=None):
        for item in items:
            p = doc.add_paragraph(style="List Bullet")
            add_markup(p, B(item))

    def steps(items):
        num_id = numbering.new_list()
        for item in items:
            p = doc.add_paragraph(style="List Number")
            add_markup(p, B(item))
            num_pr = p._p.get_or_add_pPr().get_or_add_numPr()
            num_pr.get_or_add_ilvl().val = 0
            num_pr.get_or_add_numId().val = num_id

    def _picture(path: Path, max_w: float, max_h: float):
        jpg = gm.to_jpeg(Path(path))
        with PILImage.open(jpg) as im:
            w, h = im.size
        ratio = min(max_w / w, max_h / h)
        return jpg, w * ratio, h * ratio

    def _caption(text):
        fig_no[0] += 1
        p = doc.add_paragraph(style="Figure Caption")
        add_markup(p, B(f"Figure {fig_no[0]}. {text}"))

    def figure(path, caption_text, max_h=gm.FIG_MAX_H_CM, width_ratio=1.0):
        if not path or not Path(path).exists():
            return
        fig_no[0] += 1
        parts = gm.layout_figure(Path(path), PAGE_W * width_ratio, max_h, 2.54)
        for i, (jpg, w, _) in enumerate(parts, 1):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_after = Pt(0)
            p.add_run().add_picture(str(jpg), width=Cm(w))
            text = caption_text if len(parts) == 1 else f"{caption_text} (part {i} of {len(parts)})"
            cp = doc.add_paragraph(style="Figure Caption")
            add_markup(cp, B(f"Figure {fig_no[0]}. {text}"))

    def side_by_side(paths, caption_text, max_h=20.0):
        paths = [Path(p) for p in paths if p and Path(p).exists()]
        if not paths:
            return
        gap = 0.4
        col_w = (PAGE_W - gap * (len(paths) - 1)) / len(paths)
        t = doc.add_table(rows=1, cols=len(paths))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, pth in enumerate(paths):
            jpg, w, _ = _picture(pth, col_w - 0.3, max_h)
            c = t.rows[0].cells[i]
            c.width = Cm(col_w)
            para = c.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.add_run().add_picture(str(jpg), width=Cm(w))
        _caption(caption_text)

    def _styled_table(header: list[str], rows: list[list[str]], widths: list[float]):
        t = doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, text in enumerate(header):
            c = t.rows[0].cells[j]
            shade(c, "198754")
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            add_markup(c.paragraphs[0], f"<b>{text}</b>", size=9.5, color=RGBColor(0xFF, 0xFF, 0xFF))
        for i, row in enumerate(rows, 1):
            for j, text in enumerate(row):
                c = t.rows[i].cells[j]
                if i % 2 == 0:
                    shade(c, "F1F8F4")
                c.paragraphs[0].paragraph_format.space_after = Pt(0)
                add_markup(c.paragraphs[0], B(str(text)), size=9.5)
        set_col_widths(t, widths)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def legend(rows, col_widths=(1.2, 4.3, None)):
        widths = list(col_widths)
        if widths[2] is None:
            widths[2] = PAGE_W - widths[0] - widths[1]
        _styled_table(
            ["No.", "Element", "What it is / what it does"],
            [[f"<b>{n}</b>", f"<b>{name}</b>", desc] for n, name, desc in rows],
            widths,
        )

    def table(header, rows, widths):
        _styled_table(header, rows, widths)

    def note(text, kind="tip"):
        color = {"tip": "198754", "warn": "C77700", "info": "0D6EFD"}[kind]
        label = {"tip": "TIP", "warn": "IMPORTANT", "info": "NOTE"}[kind]
        t = doc.add_table(rows=1, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        c0, c1 = t.rows[0].cells
        for c in (c0, c1):
            shade(c, "F6F9F7")
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
        cell_border(c0, "left", color, size=24)
        add_markup(c0.paragraphs[0], f"<b>{label}</b>", size=8.5,
                   color=RGBColor.from_string(color))
        add_markup(c1.paragraphs[0], B(text), size=9.5)
        set_col_widths(t, [2.7, PAGE_W - 2.7])
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def badge_table():
        rows = [
            ("198754", "Healthy", "The leaf shows no sign of disease. The result includes upkeep advice for a healthy crop."),
            ("DC3545", "Unhealthy", "A disease was detected. The disease name is shown under the verdict, with a full treatment guide."),
            ("FFC107", "Not a Rice Leaf", "The photo does not look like a rice leaf, so no diagnosis was made. Retake the photo."),
            ("6C757D", "Unknown", "The AI could not make a confident decision, or processing failed. Try a clearer photo."),
        ]
        t = _styled_table(["Colour", "Verdict", "Meaning"],
                          [["", f"<b>{n}</b>", d] for _, n, d in rows],
                          [2.0, 3.6, PAGE_W - 5.6])
        for i, (color, _, _) in enumerate(rows, 1):
            c = t.rows[i].cells[0]
            c._tc.get_or_add_tcPr().remove(c._tc.get_or_add_tcPr().find(qn("w:shd"))) if c._tc.get_or_add_tcPr().find(qn("w:shd")) is not None else None
            shade(c, color)

    # ── Re-run the content section of build_pdf() with these emitters ──
    src = inspect.getsource(gm.build_pdf)
    start = src.rfind(chr(10), 0, src.index("# ── Cover")) + 1
    end = src.index("doc.multiBuild(")
    content = textwrap.dedent(src[start:end])
    namespace = {
        "story": story, "P": P, "bullets": bullets, "steps": steps, "figure": figure,
        "side_by_side": side_by_side, "legend": legend,
        "table": table, "note": note, "badge_table": badge_table,
        "Paragraph": Paragraph, "ParagraphStyle": ParagraphStyle, "Spacer": Spacer,
        "PageBreak": PageBreak, "Image": Image, "TableOfContents": TableOfContents,
        "KeepTogether": lambda items: None, "colors": colors, "cm": cm, "PAGE_W": PAGE_W,
        "body": body, "small": small, "h1": h1, "h2": h2, "h3": h3, "caption": caption,
        "bullet": bullet, "step_style": step_style, "cell": cell, "cell_b": cell_b,
        "title_style": title_style, "subtitle": subtitle, "meta": meta,
        "GREEN": "#198754", "DARK_GREEN": "#1b4332", "GREY": "#6c757d", "LIGHT": "#f1f8f4",
        "FONT": "Arial", "FONT_B": "Arial",
        "shots": shots, "diseases": diseases, "workflow": workflow, "photo_tips": photo_tips,
        "datetime": datetime, "Path": Path, "split_tall": gm.split_tall, "to_jpeg": gm.to_jpeg,
        "min": min,
    }
    exec(compile(content, "<manual-content>", "exec"), namespace)

    doc.save(str(out))
    return out


def main() -> None:
    gm.django_setup()
    from detection.models import SiteSettings

    gm.BRAND = (SiteSettings.load().site_brand or "RiceGuard").strip()
    safe = "".join(ch for ch in gm.BRAND if ch.isalnum() or ch in "-_") or "RiceGuard"
    out = DOCS_DIR / f"{safe}_User_Manual.docx"

    index = gm.SHOT_DIR / "_shots.json"
    if not index.exists():
        raise SystemExit("Run docs/generate_user_manual.py first to capture the screenshots.")
    shots = {k: (Path(v) if v else None) for k, v in json.loads(index.read_text()).items()}
    diseases = gm.disease_reference()
    workflow = gm.create_workflow_diagram()
    tips = gm.create_photo_tips_diagram()

    build_docx(shots, workflow, tips, diseases, out)
    print(f"Done: {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
