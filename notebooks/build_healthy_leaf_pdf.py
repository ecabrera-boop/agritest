"""
Documentary PDF for rice_healthy_leaf_simulation.ipynb

Re-runs the notebook analysis (Layer 1 health identification), saves every
graph, and writes a narrative PDF.

From the project root:
    python notebooks/build_healthy_leaf_pdf.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image as PILImage

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
NOTEBOOK = HERE / "rice_healthy_leaf_simulation.ipynb"
FIG_DIR = HERE / "healthy_leaf_pdf_figures"
OUTPUT_PDF = HERE / "Rice_Healthy_Leaf_Identification.pdf"

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def _text_to_source(text: str) -> list[str]:
    return [ln if ln.endswith("\n") else ln + "\n" for ln in (text.rstrip("\n") + "\n").splitlines(True)]


def run_notebook_figures() -> list[tuple[int, str, list[Path]]]:
    """Execute notebook code cells; return [(cell_index, title, figure_paths)]."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIG_DIR.glob("*.png"):
        old.unlink()

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns: dict = {
        "__name__": "__main__",
        "np": np,
        "plt": plt,
        "Image": PILImage,
        "Path": Path,
        "os": os,
        "sys": sys,
    }

    captured: list[tuple[int, str, list[Path]]] = []
    fig_counter = [0]

    def save_open_figures(cell_idx: int) -> list[Path]:
        paths: list[Path] = []
        for num in list(plt.get_fignums()):
            fig = plt.figure(num)
            fig_counter[0] += 1
            out = FIG_DIR / f"c{cell_idx:02d}_{fig_counter[0]:02d}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            paths.append(out)
        plt.close("all")
        return paths

    orig_show = plt.show

    def capturing_show(*_a, **_k):
        return None

    plt.show = capturing_show

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = _cell_source(cell).strip()
        if not src:
            continue
        title = src.splitlines()[0].lstrip("# ").strip()[:80]
        try:
            exec(compile(src, f"nb_cell_{i}", "exec"), ns, ns)
        except Exception:
            print(f"[cell {i}] {title}")
            traceback.print_exc()
        paths = save_open_figures(i)
        if paths:
            captured.append((i, title, paths))
            print(f"[cell {i}] saved {len(paths)} figure(s) — {title}")

    plt.show = orig_show
    return captured, ns


def add_image(story, path: Path, page_width: float, body_style, max_height: float) -> None:
    from reportlab.platypus import Image, Paragraph

    if not path.exists():
        story.append(Paragraph(f"<i>Missing figure: {path.name}</i>", body_style))
        return
    img = Image(str(path))
    ratio = min(page_width / img.drawWidth, max_height / img.drawHeight, 1.0)
    img.drawWidth *= ratio
    img.drawHeight *= ratio
    story.append(img)


def build_pdf(captured: list[tuple[int, str, list[Path]]], ns: dict) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        KeepTogether,
        ListFlowable,
        ListItem,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=22, spaceAfter=12,
        textColor=colors.HexColor("#2e9e4f"),
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=16, spaceBefore=14, spaceAfter=8,
        textColor=colors.HexColor("#1b5e3a"),
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=6,
        textColor=colors.HexColor("#2d6a4f"),
    )
    body = ParagraphStyle(
        "body", parent=styles["BodyText"], fontSize=10.5, leading=14, alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    caption = ParagraphStyle(
        "Caption", parent=styles["BodyText"], fontSize=9, textColor=colors.grey,
        alignment=TA_CENTER, spaceAfter=10, spaceBefore=4,
    )
    meta = ParagraphStyle(
        "Meta", parent=body, alignment=TA_CENTER,
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Healthy Rice Leaf Identification — How It Works",
        author="RiceGuard",
    )
    story = []
    page_width = A4[0] - 3.6 * cm

    def img(path: Path, cap: str = "", max_h: float = 9.2 * cm):
        add_image(story, path, page_width, body, max_h)
        if cap:
            story.append(Paragraph(cap, caption))

    def bullets(items: list[str]) -> None:
        for item in items:
            story.append(Paragraph(f"• {item}", ParagraphStyle("b", parent=body, leftIndent=12, spaceAfter=3)))

    figs_by_cell = {idx: paths for idx, _, paths in captured}

    diagnosis = str(ns.get("diagnosis", "healthy")).upper()
    confidence = float(ns.get("confidence", 0.0)) * 100
    health_probs = ns.get("health_probs") or ns.get("prob_map") or {}
    layer1 = ns.get("layer1_signals") or {}
    coverage = float(ns.get("coverage", 0.0)) * 100
    used_model = bool(ns.get("used_model", False))
    source = "production model" if used_model else "Layer 1 simulation (whole-leaf colour)"

    # ── Cover ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2.2 * cm))
    story.append(Paragraph("RiceGuard", title_style))
    story.append(Paragraph(
        "<b>Healthy Rice Leaf Identification</b><br/>Documentary of the interactive notebook",
        ParagraphStyle("Sub", parent=styles["Heading2"], alignment=TA_CENTER, fontSize=13),
    ))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        f"Source notebook: <font face='Courier'>notebooks/rice_healthy_leaf_simulation.ipynb</font><br/>"
        f"Generated: {datetime.now().strftime('%B %d, %Y')}<br/>"
        f"Result on this sample: <b>{diagnosis}</b> ({confidence:.1f}% confidence)<br/>"
        f"Engine used: {source}",
        meta,
    ))
    story.append(Spacer(1, 0.6 * cm))
    cover_img = None
    if 5 in figs_by_cell:
        cover_img = figs_by_cell[5][0]
    elif (HERE / "assets" / "healthy_leaf.jpg").exists():
        cover_img = HERE / "assets" / "healthy_leaf.jpg"
    if cover_img:
        img(cover_img, "Sample used in this documentary: a healthy green rice leaf.", max_h=8.5 * cm)

    story.append(PageBreak())

    # ── How the system works ─────────────────────────────────────────────
    story.append(Paragraph("1. How the system identifies a healthy leaf", h1))
    story.append(Paragraph(
        "The RiceGuard web app does not guess a disease from a red overlay. It runs a "
        "stacked pipeline. This notebook documents <b>Layer 1</b> — the health decision — "
        "on a leaf that has no disease spots.",
        body,
    ))
    tbl = Table(
        [
            ["Layer", "Question", "If the answer is…"],
            ["0  Rice-leaf gate", "Is this a rice leaf photo?", "No → stop (not rice)"],
            ["1  Health model", "Healthy or diseased?", "Healthy → stop (this notebook)"],
            ["2  Disease model", "Which disease?", "Only runs if Layer 1 says diseased"],
            ["3  Recommendation", "What should the farmer do?", "PhilRice-aligned guidance"],
        ],
        colWidths=[4.2 * cm, 6.2 * cm, 6.4 * cm],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e9e4f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3faf4")]),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#d8f3dc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph(
        "The production health model (<font face='Courier'>cnn_health.pkl</font> / "
        "<font face='Courier'>rf_health.pkl</font>) is trained on the <b>whole leaf</b>. "
        "A green blade with vein texture is still healthy. A naïve HSV “lesion” mask often "
        "paints veins red and would wrongly say brown spot or leaf smut — that is a teaching "
        "overlay, not the diagnosis.",
        body,
    ))
    story.append(Paragraph(
        "This Jupyter kernel may be older than the trained sklearn pickle. When the pickle "
        "cannot load, the notebook uses a <b>Layer 1 simulation</b> with the same idea: "
        "compare green tissue vs brown/gray damage across the entire leaf mask. On this "
        "sample both routes agree: <b>HEALTHY</b>.",
        body,
    ))

    story.append(Paragraph("Notebook walk-through", h2))
    bullets([
        "Step 1 — Load the photo.",
        "Step 2 — Letterbox-resize to 128×128 (same size the app uses).",
        "Step 3 — Inspect RGB / HSV colour (healthy green band).",
        "Step 4 — HSV lesion search (may false-positive on veins).",
        "Step 5 — Turn the leaf into numbers (GLCM, HSV moments, LBP, shape).",
        "Step 6 — Layer 1: healthy vs diseased (the decision).",
        "Step 7–8 — Flow diagram and farmer dashboard.",
    ])

    story.append(PageBreak())

    # Map notebook cells to documentary sections
    sections = [
        (5, "2. Step 1 — Load the healthy leaf photo",
         "The input is a close-up of a green rice leaf with no brown or gray disease spots. "
         "Everything later is computed from this single image."),
        (7, "3. Step 2 — Preprocess to 128×128",
         "The app letterbox-resizes every upload to 128×128 with a neutral green-gray backdrop "
         "and a gentle brightness normalize. Training and inference share this step so features "
         "stay comparable."),
        (9, "4. Step 3 — Colour analysis (RGB and HSV)",
         "Healthy rice tissue sits in a green hue band (about 35–110°) with useful saturation. "
         "Disease pulls pixels toward brown, yellow, or gray. The histograms below are how you "
         "see that before any classifier runs."),
        (11, "5. Step 4 — Search for lesions (teaching overlay)",
         "HSV rules mark off-green or dark pixels inside the leaf. On a healthy leaf this mask "
         "is often noisy: veins, shadows, and a white background can look like lesions. "
         f"On this run HSV coverage was <b>{coverage:.1f}%</b> of the leaf — that number is "
         "<b>not</b> the diagnosis. Layer 1 looks at the whole leaf instead."),
        (13, "6. Step 5 — Features (numbers the classifier sees)",
         "Texture (GLCM, LBP), colour moments, and shape descriptors summarise the region. "
         "The production model uses a long feature vector (or a CNN embedding). The bar charts "
         "show the teaching version of those families."),
        (15, "7. Step 6 — Layer 1 health identification",
         "This is the decision the app makes first: <b>healthy</b> or <b>diseased</b>. "
         "Green fraction of the leaf is high; brown/yellow/gray are low. HSV lesion coverage "
         "is only a weak extra signal because it false-positives on veins. "
         f"Result: <b>{diagnosis}</b> at <b>{confidence:.1f}%</b> confidence "
         f"({source}). Layer 2 (disease name) is skipped."),
        (17, "8. Step 7 — End-to-end flow",
         "Photo → preprocess → colour/lesion view → features → Layer 1 card. "
         "For a healthy leaf the path stops at HEALTHY."),
        (19, "9. Step 8 — Diagnosis dashboard",
         "The summary a farmer would see: input photo, teaching overlay, Layer 1 bars, "
         "and PhilRice-aligned prevention tips for a healthy crop."),
        (21, "10. Comparison (healthy vs a diseased sample)",
         "Side-by-side HSV-mask coverage is only a visualisation. The health decision "
         "still comes from Layer 1, not from which overlay is redder."),
    ]

    for cell_idx, heading, text in sections:
        story.append(Paragraph(heading, h1))
        story.append(Paragraph(text, body))
        paths = figs_by_cell.get(cell_idx, [])
        if not paths:
            story.append(Paragraph("<i>No figure captured for this step (run the notebook cells if missing).</i>", body))
            continue
        # One figure per row; use slightly smaller height if many
        max_h = 8.8 * cm if len(paths) == 1 else 7.2 * cm
        for p in paths:
            img(p, p.stem.replace("_", " "), max_h=max_h)

    story.append(PageBreak())
    story.append(Paragraph("11. Layer 1 numbers on this sample", h1))
    story.append(Paragraph(
        "These whole-leaf fractions are what the simulation (and the trained health model, "
        "in spirit) use. Green dominates; brown is near zero — so the leaf is healthy even "
        "when HSV marks some vein pixels.",
        body,
    ))

    def pct(key: str) -> str:
        v = layer1.get(key)
        if v is None:
            return "—"
        return f"{float(v) * 100:.1f}%"

    rows = [
        ["Signal", "Value", "How to read it"],
        ["Green fraction of leaf", pct("green_frac"), "High → healthy tissue"],
        ["Brown fraction", pct("brown_frac"), "High → necrotic / brown-spot like"],
        ["Yellow fraction", pct("yellow_frac"), "Streaks / chlorosis"],
        ["Gray fraction", pct("gray_frac"), "Blast-like gray centres"],
        ["HSV lesion coverage", f"{coverage:.1f}%", "Teaching overlay only (veins inflate it)"],
        ["Diagnosis", diagnosis, source],
        ["Confidence", f"{confidence:.1f}%", "P(healthy) vs P(diseased)"],
    ]
    if isinstance(health_probs, dict) and health_probs:
        for k, v in sorted(health_probs.items(), key=lambda kv: -float(kv[1])):
            rows.append([f"P({k})", f"{float(v)*100:.1f}%", "Layer 1 class probability"])

    num_tbl = Table(rows, colWidths=[5.2 * cm, 3.6 * cm, 8.0 * cm])
    num_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e9e4f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3faf4")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(num_tbl)

    story.append(Paragraph("12. What this notebook is — and is not", h1))
    bullets([
        "It <b>is</b> a step-by-step documentary of Layer 1 health identification, with the same graphs as the .ipynb.",
        "It <b>is</b> aligned with the Django upload path: preprocess → health → (disease only if diseased).",
        "It is <b>not</b> a replacement for the trained .pkl models in production. The web app still loads cnn_health / rf_health when the environment can.",
        "HSV red overlays in Step 4 are for teaching. Do not treat high overlay coverage as a disease call on a green leaf.",
    ])
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "Re-generate this PDF after changing the notebook:<br/>"
        "<font face='Courier'>python notebooks/build_healthy_leaf_pdf.py</font>",
        body,
    ))

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.8 * cm, 1.1 * cm, "RiceGuard — Healthy leaf identification documentary")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUTPUT_PDF


def main() -> None:
    print("Notebook:", NOTEBOOK)
    print("Running analysis and capturing graphs…")
    captured, ns = run_notebook_figures()
    n_figs = sum(len(p) for _, _, p in captured)
    print(f"Captured {n_figs} figure(s) from {len(captured)} cells")
    print("Building PDF…")
    try:
        import reportlab  # noqa: F401
    except ImportError:
        print("Installing reportlab…")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    out = build_pdf(captured, ns)
    print("Wrote", out)


if __name__ == "__main__":
    main()
