"""
Generate Layer 0 (not-rice gate) documentation PDF in the same style as
docs/RiceGuard_System_Features.pdf.

Runs the Layer 0 simulation notebook, captures every graph, and writes a
narrative A4 PDF.

From the project root:
    python docs/generate_layer0_pdf.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOK = PROJECT_ROOT / "notebooks" / "rice_layer0_not_rice_simulation.ipynb"
FIG_DIR = DOCS_DIR / "layer0_pdf_figures"
OUTPUT_PDF = DOCS_DIR / "RiceGuard_Layer0_Not_Rice.pdf"
SAMPLE = PROJECT_ROOT / "notebooks" / "assets" / "not_rice_barnyard_grass.png"

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def create_layer0_pipeline_diagram() -> Path:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.set_title(
        "RiceGuard Layer 0 — Rice vs Not-Rice Gate",
        fontsize=15, fontweight="bold", pad=12,
    )

    boxes = [
        (0.35, 2.35, 2.5, 1.9, "#264653", "Upload\nphoto"),
        (3.2, 2.35, 2.7, 1.9, "#e76f51", "Layer 0\nRice-leaf gate"),
        (6.3, 3.35, 2.7, 1.15, "#8a8a8a", "Layer 1 Health\n(skipped)"),
        (9.4, 3.35, 2.9, 1.15, "#8a8a8a", "Layer 2 Disease\n(skipped)"),
        (6.3, 0.55, 6.0, 1.45, "#e76f51", "STOP — Not a rice leaf\nNo diagnosis. Ask for a rice blade photo."),
    ]
    for x, y, w, h, fc, label in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor=fc, edgecolor="white", linewidth=2,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

    ax.add_patch(FancyArrowPatch(
        (2.9, 3.3), (3.15, 3.3),
        arrowstyle="-|>", mutation_scale=14, color="#1b4332", lw=2,
    ))
    ax.annotate(
        "", xy=(6.3, 3.85), xytext=(5.95, 3.5),
        arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.4, linestyle="--"),
    )
    ax.annotate(
        "", xy=(9.35, 3.9), xytext=(9.05, 3.9),
        arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.4, linestyle="--"),
    )
    ax.annotate(
        "", xy=(6.3, 1.25), xytext=(5.95, 2.45),
        arrowprops=dict(arrowstyle="-|>", color="#e76f51", lw=2.4),
    )
    ax.text(7.7, 4.75, "rice  →  continue", ha="center", fontsize=8.5, color="#6c757d")
    ax.text(7.7, 2.15, "not_rice / wrong_shape  →  hard stop", ha="center",
            fontsize=8.5, color="#9a3412")

    ax.text(
        6.5, 0.12,
        "Sample in this document: barnyard grass (Echinochloa crus-galli) — a rice-field weed.",
        ha="center", fontsize=9, color="#495057",
    )

    out = FIG_DIR / "00_layer0_pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close(fig)
    return out


def run_notebook_figures() -> tuple[list[tuple[int, str, list[Path]]], dict]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIG_DIR.glob("c*.png"):
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
    plt.show = lambda *_a, **_k: None

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


def build_pdf(
    captured: list[tuple[int, str, list[Path]]],
    ns: dict,
    pipeline_diag: Path,
) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    GREEN = colors.HexColor("#198754")
    DARK = colors.HexColor("#2d6a4f")
    STRIPE = colors.HexColor("#f8f9fa")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=24, spaceAfter=16,
        textColor=GREEN,
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=18, spaceBefore=16, spaceAfter=10,
        textColor=GREEN,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=14, spaceBefore=12, spaceAfter=6,
        textColor=DARK,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10.5, leading=14,
        alignment=TA_JUSTIFY, spaceAfter=6,
    )
    caption = ParagraphStyle(
        "Caption", parent=styles["BodyText"], fontSize=9, textColor=colors.grey,
        alignment=TA_CENTER, spaceAfter=12, spaceBefore=4,
    )
    bullet = ParagraphStyle(
        "Bullet", parent=body, leftIndent=14, spaceAfter=4,
    )
    meta = ParagraphStyle("Meta", parent=body, alignment=TA_CENTER)

    def add_image(path: Path | None, caption_text: str = "", max_height: float = 12 * cm) -> None:
        if path and path.exists():
            img = Image(str(path))
            ratio = min(page_width / img.drawWidth, max_height / img.drawHeight, 1.0)
            img.drawWidth *= ratio
            img.drawHeight *= ratio
            story.append(img)
            if caption_text:
                story.append(Paragraph(caption_text, caption))
        else:
            story.append(Paragraph(f"<i>[Figure missing: {caption_text}]</i>", body))

    def bullets(items: list[str]) -> None:
        for item in items:
            story.append(Paragraph(f"• {item}", bullet))

    def green_table(data: list[list[str]], col_widths: list[float]) -> Table:
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    page_width = A4[0] - 4 * cm
    figs_by_cell = {idx: paths for idx, _, paths in captured}

    verdict = str(ns.get("verdict", "not_rice"))
    p_not = float(ns.get("p_not", 0.0)) * 100
    source = str(ns.get("source", "teaching scorer"))
    plant_frac = float(ns.get("plant_frac", 0.0)) * 100
    aspect = float(ns.get("aspect", 0.0))
    fill = float(ns.get("fill", 0.0))
    n_large = int(ns.get("n_large", 0))

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="RiceGuard Layer 0 — Identify Not Rice",
        author="Rice Leaf Disease Detector",
    )
    story: list = []

    # ── Cover ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2.2 * cm))
    story.append(Paragraph("RiceGuard", title_style))
    story.append(Paragraph(
        "<b>Layer 0 — Identify Not Rice</b><br/>Rice-leaf gate documentation",
        ParagraphStyle("Sub", parent=styles["Heading2"], alignment=TA_CENTER, fontSize=14),
    ))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}<br/>"
        "Companion to <font face='Courier'>docs/RiceGuard_System_Features.pdf</font><br/>"
        "Notebook: <font face='Courier'>notebooks/rice_layer0_not_rice_simulation.ipynb</font><br/>"
        f"Sample verdict: <b>{verdict.replace('_', ' ').upper()}</b> "
        f"({p_not:.1f}% not-rice, {source})",
        meta,
    ))
    story.append(Spacer(1, 0.5 * cm))
    cover = SAMPLE if SAMPLE.exists() else (figs_by_cell.get(3) or [None])[0]
    add_image(cover, "Cover sample: barnyard grass (Echinochloa crus-galli) — not a rice leaf.", 9.5 * cm)
    story.append(PageBreak())

    # ── TOC ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Table of Contents", h1))
    bullets([
        "1. Purpose of Layer 0",
        "2. Place in the 4-layer pipeline",
        "3. Why this sample is not rice",
        "4. How the gate decides (colour, shape, classifier)",
        "5. Notebook walk-through (figures)",
        "6. Production models and API",
        "7. Farmer-facing result",
        "8. How to regenerate this PDF",
    ])
    story.append(PageBreak())

    # ── 1. Purpose ───────────────────────────────────────────────────────
    story.append(Paragraph("1. Purpose of Layer 0", h1))
    story.append(Paragraph(
        "RiceGuard is trained only on rice leaves. If a farmer uploads a face, a maize "
        "plant, soil, or a lookalike weed, Layers 1 and 2 would still output a confident "
        "but meaningless health or disease label. <b>Layer 0 is the rice-leaf gate</b>: "
        "it asks “is this a rice leaf?” and <b>stops the pipeline</b> when the answer is no.",
        body,
    ))
    story.append(Paragraph("What Layer 0 must reject", h2))
    bullets([
        "Non-plants: faces, indoor scenes, bare soil, tools.",
        "Other crops: maize, broadleaf weeds, mixed field scenes.",
        "Rice lookalikes: barnyard grass and similar grassy weeds with branched seed heads.",
        "Wrong framing: a whole clump of plants instead of one elongated rice leaf blade.",
    ])
    story.append(Paragraph(
        "Hard rejections use reasons <font face='Courier'>not_rice</font> and "
        "<font face='Courier'>wrong_shape</font>. A soft "
        "<font face='Courier'>low_confidence</font> flag can continue with a caution "
        "banner. This documentary focuses on a <b>hard not-rice</b> example.",
        body,
    ))
    story.append(PageBreak())

    # ── 2. Pipeline ──────────────────────────────────────────────────────
    story.append(Paragraph("2. Place in the 4-layer pipeline", h1))
    story.append(Paragraph(
        "Every upload hits Layer 0 first. On this barnyard-grass photo the gate rejects, "
        "so Layer 1 (healthy vs diseased) and Layer 2 (which disease) never run.",
        body,
    ))
    add_image(pipeline_diag, "Figure 1: Layer 0 hard-stop. Grey boxes are skipped.")
    story.append(green_table(
        [
            ["Layer", "Question", "This sample"],
            ["0  Rice-leaf gate", "Is this a rice leaf?", "No → STOP (not rice)"],
            ["1  Health model", "Healthy or diseased?", "Skipped"],
            ["2  Disease model", "Which disease?", "Skipped"],
            ["3  Recommendation", "What should the farmer do?", "Upload a rice leaf blade"],
        ],
        [4.2 * cm, 6.4 * cm, 5.4 * cm],
    ))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph(
        "Production code: <font face='Courier'>ml/layer0_validator.py</font> "
        "(Random Forest gate on raw colour/texture features) and "
        "<font face='Courier'>ml/cnn_pipeline.py</font> (CNN embedding + rice-database "
        "distance). The gate uses the <b>raw</b> upload — no luminance enhancement — "
        "so it cannot be fooled by a beautified image.",
        body,
    ))
    story.append(PageBreak())

    # ── 3. Sample ────────────────────────────────────────────────────────
    story.append(Paragraph("3. Why this sample is not rice", h1))
    story.append(Paragraph(
        "The notebook uses <font face='Courier'>notebooks/assets/not_rice_barnyard_grass.png</font>. "
        "The plant is <b>barnyard grass</b> (<i>Echinochloa crus-galli</i>), a common weed "
        "in Philippine rice paddies. It is green and grassy, so a naïve “is it green?” "
        "check would pass it. Layer 0 must use shape and a rice-vs-not-rice classifier.",
        body,
    ))
    story.append(green_table(
        [
            ["Cue", "Rice leaf (expected)", "This photo"],
            ["Subject", "One long thin blade", "Clump of many stems"],
            ["Inflorescence", "Rice panicle (if present)", "Branched bristly spikes"],
            ["Background", "Plain / one leaf filling frame", "Maize and field clutter"],
            ["Colour", "Green plant pixels", "Also green — not enough to accept"],
        ],
        [3.4 * cm, 6.3 * cm, 6.3 * cm],
    ))
    story.append(Spacer(1, 0.3 * cm))
    for p in figs_by_cell.get(3, [])[:1]:
        add_image(p, "Figure 2: Input photo loaded by the simulation notebook.", 9 * cm)
    for p in figs_by_cell.get(5, [])[:1]:
        add_image(p, "Figure 3: Teaching callouts — seed heads, stem clump, other crop.", 9 * cm)
    story.append(PageBreak())

    # ── 4. Decision ──────────────────────────────────────────────────────
    story.append(Paragraph("4. How the gate decides", h1))
    story.append(Paragraph("Colour (plant mask)", h2))
    story.append(Paragraph(
        "Layer 0 first measures the fraction of plant-like HSV pixels (green and "
        "yellow-brown leaf tones). Faces and indoor scenes score near zero and are "
        f"hard-rejected below 5%. This weed scored <b>{plant_frac:.1f}%</b> plant colour, "
        "so colour alone does <b>not</b> reject it.",
        body,
    ))
    for p in figs_by_cell.get(7, []):
        add_image(p, "Figure 4: Plant-colour mask. Green weeds still light up.", 8.5 * cm)

    story.append(Paragraph("Shape (one blade vs a clump)", h2))
    story.append(Paragraph(
        "A rice leaf is elongated. This photo is a bushy multi-stem clump. The notebook "
        f"measured largest-blob elongation <b>{aspect:.2f}</b>, bounding-box fill "
        f"<b>{fill:.2f}</b>, and <b>{n_large}</b> sizable green clumps.",
        body,
    ))
    for p in figs_by_cell.get(9, []):
        add_image(p, "Figure 5: Shape scores — not a single long rice blade.", 8.5 * cm)

    story.append(Paragraph("Classifier (rice vs not_rice)", h2))
    story.append(Paragraph(
        f"The teaching scorer combines those cues. Production "
        f"<font face='Courier'>rf_rice_gate.pkl</font> is used when the kernel can load it. "
        f"On this run the documented verdict is <b>{verdict}</b> with "
        f"<b>{p_not:.1f}%</b> not-rice probability ({source}).",
        body,
    ))
    prod = ns.get("prod")
    if isinstance(prod, dict) and prod.get("accepted"):
        story.append(Paragraph(
            f"Note: the <b>production</b> RF gate currently <b>accepted</b> this lookalike "
            f"as rice (confidence {float(prod.get('confidence', 0))*100:.1f}%). "
            "Barnyard grass is the hard case Layer 0 is still being trained against "
            "(leaf/not_rice/ negatives). The documentary keeps the teaching reject so "
            "the intended farmer result — Not a Rice Leaf — is shown.",
            body,
        ))
    for p in figs_by_cell.get(11, []):
        add_image(p, "Figure 6: Layer 0 rice vs not_rice probabilities.", 8 * cm)
    story.append(PageBreak())

    # ── 5. Walk-through ──────────────────────────────────────────────────
    story.append(Paragraph("5. Notebook walk-through", h1))
    story.append(Paragraph(
        "The interactive notebook mirrors the app: load the photo, inspect colour and "
        "shape, run the gate, then <b>stop</b>. It does not invent a disease on a weed.",
        body,
    ))
    bullets([
        "Step 1 — Load the raw RGB upload (no enhancement).",
        "Step 2 — Label visual cues that distinguish barnyard grass from rice.",
        "Step 3 — HSV plant mask (same idea as ml.preprocessor).",
        "Step 4 — Connected components and elongation.",
        "Step 5 — Rice vs not_rice scores (production gate if available).",
        "Step 6 — Flowchart: Layers 1–2 greyed out.",
        "Step 7 — Farmer dashboard: NOT A RICE LEAF.",
    ])
    for p in figs_by_cell.get(13, []):
        add_image(p, "Figure 7: Pipeline flowchart — hard stop after Layer 0.", 7.5 * cm)
    for p in figs_by_cell.get(15, []):
        add_image(p, "Figure 8: Final dashboard for this not-rice sample.", 9.5 * cm)
    story.append(PageBreak())

    # ── 6. Production ────────────────────────────────────────────────────
    story.append(Paragraph("6. Production models and API", h1))
    story.append(Paragraph(
        "The Django upload path calls <font face='Courier'>ml.pipeline.run_pipeline</font>. "
        "If the CNN engine is available it uses the CNN rice gate; otherwise the Random Forest "
        "gate. Either way a hard not-rice flag returns health = unknown and "
        "rejection_reason = not_rice.",
        body,
    ))
    story.append(green_table(
        [
            ["Artifact / API", "Role"],
            ["ml/layer0_validator.py", "RF gate: plant fraction, shape, rf_rice_gate.pkl"],
            ["ml/saved_models/rf_rice_gate.pkl", "Logistic / RF rice vs not_rice on raw features"],
            ["ml/cnn_pipeline.py", "CNN gate: LR+RF on embeddings + rice cosine floor"],
            ["ml/saved_models/cnn_rice_gate.pkl", "CNN not_rice probability head"],
            ["POST /api/validate-layer0/", "Pre-upload Layer 0 check only"],
            ["POST /api/scans/", "Full pipeline (Layer 0 first)"],
        ],
        [6.5 * cm, 9.5 * cm],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Held-out rice-gate accuracy in the current training report is about "
        "<b>86.5%</b> test / <b>86.7%</b> validation (21,684 rice vs 2,713 not-rice "
        "samples). Lookalike weeds remain the hard cases — that is why this documentary "
        "uses barnyard grass.",
        body,
    ))
    story.append(PageBreak())

    # ── 7. Farmer result ─────────────────────────────────────────────────
    story.append(Paragraph("7. Farmer-facing result", h1))
    story.append(Paragraph(
        "The result page does not show blast or blight percentages for a rejected photo. "
        "It shows <b>Not a Rice Leaf</b> and asks for a close-up of one rice leaf blade "
        "against a plain background. The same copy is available in English, Tagalog, and Ilocano.",
        body,
    ))
    bullets([
        "Health badge: Not a Rice Leaf (not Healthy / Unhealthy).",
        "Disease name: hidden (N/A).",
        "Recommendation: rec_not_rice — this tool cannot classify non-rice plants.",
        "History list: is_not_rice flag so the scan is not counted as a disease case.",
    ])
    story.append(Paragraph("Numbers on this sample", h2))
    story.append(green_table(
        [
            ["Signal", "Value"],
            ["Layer 0 verdict", verdict],
            ["p(not_rice)", f"{p_not:.1f}%"],
            ["Scorer", source],
            ["Plant-colour fraction", f"{plant_frac:.1f}%"],
            ["Largest-blob elongation", f"{aspect:.2f}"],
            ["Bounding-box fill", f"{fill:.2f}"],
            ["Sizable green clumps", str(n_large)],
            ["Next layers", "Skipped" if verdict == "not_rice" else "Would run Layer 1"],
        ],
        [7 * cm, 9 * cm],
    ))
    story.append(PageBreak())

    # ── 8. Regen ─────────────────────────────────────────────────────────
    story.append(Paragraph("8. How to regenerate this PDF", h1))
    story.append(Paragraph(
        "This file is the Layer 0 counterpart of <font face='Courier'>docs/RiceGuard_System_Features.pdf</font>. "
        "After changing the notebook or sample image, rebuild:",
        body,
    ))
    bullets([
        "Interactive notebook: open notebooks/rice_layer0_not_rice_simulation.ipynb and Run All.",
        "Rebuild notebook from template: python notebooks/build_layer0_notebook.py",
        "Rebuild this PDF: python docs/generate_layer0_pdf.py",
        "Full system PDF (screenshots of the live site): python docs/generate_system_pdf.py",
    ])
    story.append(Paragraph("What this document is — and is not", h2))
    bullets([
        "It <b>is</b> a step-by-step documentary of Layer 0 rejecting a non-rice weed, with the same graphs as the .ipynb.",
        "It <b>is</b> aligned with the Django path: gate first, then stop on not_rice.",
        "It is <b>not</b> a substitute for the trained .pkl files. The web app still loads rf_rice_gate / cnn_rice_gate when present.",
        "Green plant colour is not a rice-leaf pass. Shape plus the trained gate are required.",
    ])

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.15 * cm, "RiceGuard — Layer 0 Identify Not Rice")
        canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUTPUT_PDF


def main() -> None:
    print("Notebook:", NOTEBOOK)
    if not NOTEBOOK.exists():
        raise SystemExit(f"Missing {NOTEBOOK} — run python notebooks/build_layer0_notebook.py first.")
    print("Drawing Layer 0 pipeline diagram…")
    pipeline = create_layer0_pipeline_diagram()
    print("Running notebook and capturing graphs…")
    captured, ns = run_notebook_figures()
    n_figs = sum(len(p) for _, _, p in captured)
    print(f"Captured {n_figs} figure(s) from {len(captured)} cells")
    try:
        import reportlab  # noqa: F401
    except ImportError:
        print("Installing reportlab…")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    print("Building PDF…")
    out = build_pdf(captured, ns, pipeline)
    print("Wrote", out)


if __name__ == "__main__":
    main()
