"""
Generate RiceGuard system features PDF with screenshots and diagrams.
Run from project root: python docs/generate_system_pdf.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
SCREENSHOTS_DIR = DOCS_DIR / "screenshots"
OUTPUT_PDF = DOCS_DIR / "RiceGuard_System_Features.pdf"

BASE_URL = "http://127.0.0.1:8765"
ADMIN_USER = "pc"
ADMIN_PASS = "docgen_temp_2026"

DISEASED_SCAN_ID = 384
HEALTHY_SCAN_ID = 385


def ensure_dirs() -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_scan_ids() -> tuple[int, int]:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    sys.path.insert(0, str(PROJECT_ROOT))
    import django

    django.setup()
    from detection.models import LeafScan

    diseased = LeafScan.objects.filter(result_health="diseased").order_by("-pk").first()
    healthy = LeafScan.objects.filter(result_health="healthy").order_by("-pk").first()
    return (
        diseased.pk if diseased else DISEASED_SCAN_ID,
        healthy.pk if healthy else HEALTHY_SCAN_ID,
    )


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "manage.py", "runserver", "8765", "--noreload"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(40):
        try:
            import urllib.request
            urllib.request.urlopen(BASE_URL, timeout=2)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("Django server did not start in time")


def capture_screenshots(diseased_id: int, healthy_id: int) -> dict[str, Path]:
    from playwright.sync_api import sync_playwright

    shots: dict[str, Path] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # ── Public pages (unauthenticated) ──────────────────────────────────
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()

        public_pages = [
            ("01_home",            "/",                          1400, 900),
            ("02_history",         "/history/",                  1400, 1100),
            ("03_scan_locations",  "/locations/",                1400, 900),
            ("04_result_diseased", f"/result/{diseased_id}/",    1400, 2200),
            ("05_result_healthy",  f"/result/{healthy_id}/",     1400, 1800),
        ]
        for key, path, width, height in public_pages:
            page.set_viewport_size({"width": width, "height": height})
            page.goto(f"{BASE_URL}{path}", wait_until="networkidle")
            if "result" in key:
                page.evaluate("window.scrollTo(0, 0)")
                time.sleep(0.3)
            out = SCREENSHOTS_DIR / f"{key}.png"
            page.screenshot(path=str(out), full_page=True)
            shots[key] = out
            print(f"    captured {key}.png")

        ctx.close()

        # ── Admin login page (NOT yet logged in) ────────────────────────────
        login_ctx = browser.new_context(viewport={"width": 1200, "height": 800})
        login_page = login_ctx.new_page()
        login_page.goto(f"{BASE_URL}/admin/login/", wait_until="networkidle")
        out = SCREENSHOTS_DIR / "06_admin_login.png"
        login_page.screenshot(path=str(out), full_page=True)
        shots["06_admin_login"] = out
        print("    captured 06_admin_login.png")
        login_ctx.close()

        # ── Admin pages (logged in) ──────────────────────────────────────────
        admin_ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        admin_page = admin_ctx.new_page()
        admin_page.goto(f"{BASE_URL}/admin/login/", wait_until="networkidle")
        admin_page.fill('input[name="username"]', ADMIN_USER)
        admin_page.fill('input[name="password"]', ADMIN_PASS)
        admin_page.click('input[type="submit"]')
        admin_page.wait_for_load_state("networkidle")

        admin_pages = [
            ("07_admin_dashboard",      "/admin/",                                              1400, 900),
            ("08_admin_site_settings",  "/admin/detection/sitesettings/1/change/",              1400, 900),
            ("09_admin_leafscans",      "/admin/detection/leafscan/",                           1400, 1100),
            ("10_admin_leafscan_detail",f"/admin/detection/leafscan/{diseased_id}/change/",     1400, 1400),
        ]
        for key, path, width, height in admin_pages:
            admin_page.set_viewport_size({"width": width, "height": height})
            admin_page.goto(f"{BASE_URL}{path}", wait_until="networkidle")
            out = SCREENSHOTS_DIR / f"{key}.png"
            admin_page.screenshot(path=str(out), full_page=True)
            shots[key] = out
            print(f"    captured {key}.png")

        admin_ctx.close()
        browser.close()

    return shots


# ── Diagrams ──────────────────────────────────────────────────────────────────

def create_pipeline_diagram() -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("RiceGuard 4-Layer AI Pipeline (Random Forest)", fontsize=15, fontweight="bold", pad=12)

    layers = [
        (0.4,  "Layer 0\nRice Leaf Gate",  "#2d6a4f",
         "Validates image is a rice leaf\n(heuristic + RF gate model)"),
        (3.4,  "Layer 1\nHealth Check",    "#40916c",
         "Healthy vs. Unhealthy\n(RF voting ensemble + TTA)"),
        (6.4,  "Layer 2\nDisease ID",      "#52b788",
         "Specific disease label\n(brown spot, blast, BLB, …)"),
        (9.4,  "Layer 3\nRecommendation",  "#74c69d",
         "PhilRice-aligned treatment\nguide (EN / Tagalog / Ilocano)"),
    ]

    for x, title, color, desc in layers:
        box = FancyBboxPatch(
            (x, 2.2), 2.6, 2.0,
            boxstyle="round,pad=0.08,rounding_size=0.2",
            facecolor=color, edgecolor="white", linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x + 1.3, 3.35, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(x + 1.3, 1.1, desc, ha="center", va="top", fontsize=8.5, color="#1b4332")

    for x in (3.0, 6.0, 9.0):
        ax.add_patch(FancyArrowPatch(
            (x, 3.2), (x + 0.35, 3.2),
            arrowstyle="-|>", mutation_scale=14, color="#1b4332", lw=2,
        ))

    ax.text(6.5, 0.28,
            "Health accuracy: 94.03%  |  Disease accuracy: 90.48%  |  "
            "9,129 deduplicated training images",
            ha="center", fontsize=9, color="#495057")

    out = SCREENSHOTS_DIR / "00_pipeline_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close(fig)
    return out


def create_architecture_diagram() -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("RiceGuard System Architecture", fontsize=15, fontweight="bold", pad=12)

    boxes = [
        # row 1
        (0.4,  4.2, 3.5, 1.2, "#198754", "Web Frontend\n(Bootstrap 5 + i18n)"),
        (4.5,  4.2, 3.5, 1.2, "#0d6efd", "Django Backend\n(views, forms, REST API)"),
        (8.6,  4.2, 3.5, 1.2, "#6f42c1", "ML Pipeline\n(Random Forest ensemble)"),
        # row 2
        (0.4,  2.3, 3.5, 1.2, "#fd7e14", "SQLite Database\n(LeafScan records)"),
        (4.5,  2.3, 3.5, 1.2, "#dc3545", "Django Admin\n(Site settings + scans)"),
        (8.6,  2.3, 3.5, 1.2, "#20c997", "Media Storage\n(leaf upload images)"),
        # row 3 — ML internals
        (0.4,  0.4, 3.5, 1.1, "#495057", "Preprocessor\n(color, LBP, GLCM, HOG, edges)"),
        (4.5,  0.4, 3.5, 1.1, "#495057", "Voting Ensemble\n(RF × 2 + ExtraTrees + GB + SVM)"),
        (8.6,  0.4, 3.5, 1.1, "#495057", "PCA + Scaler\n(StandardScaler → PCA 95%)"),
    ]
    for x, y, w, h, color, label in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.15",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.92,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white")

    for (x1, y1), (x2, y2) in [
        ((3.9, 4.8), (4.5, 4.8)),
        ((8.0, 4.8), (8.6, 4.8)),
        ((2.15, 4.2), (2.15, 3.5)),
        ((6.25, 4.2), (6.25, 3.5)),
        ((10.35, 4.2), (10.35, 3.5)),
        ((6.25, 2.3), (6.25, 1.5)),
        ((10.35, 2.3), (10.35, 1.5)),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#6c757d", lw=1.5))

    ax.text(6.5, 0.0,
            "REST API: GET/POST /api/scans/  |  POST /api/validate-layer0/  |  "
            "POST /api/locate/  |  GET /lang/<lang>/",
            ha="center", fontsize=8.5, color="#495057")

    out = SCREENSHOTS_DIR / "00_architecture_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close(fig)
    return out


# ── PDF builder ───────────────────────────────────────────────────────────────

def build_pdf(shots: dict[str, Path], pipeline_diag: Path, arch_diag: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=24, spaceAfter=20,
        textColor=colors.HexColor("#198754"),
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=18, spaceBefore=16, spaceAfter=10,
        textColor=colors.HexColor("#198754"),
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=14, spaceBefore=12, spaceAfter=6,
        textColor=colors.HexColor("#2d6a4f"),
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10.5, leading=14, alignment=TA_JUSTIFY,
    )
    caption = ParagraphStyle(
        "Caption", parent=styles["BodyText"], fontSize=9, textColor=colors.grey,
        alignment=TA_CENTER, spaceAfter=12,
    )
    bullet = ParagraphStyle(
        "Bullet", parent=body, leftIndent=14, bulletIndent=0, spaceAfter=4,
    )
    code_style = ParagraphStyle(
        "Code", parent=styles["Code"], fontSize=9, leading=12, leftIndent=14,
        backColor=colors.HexColor("#f8f9fa"),
    )

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="RiceGuard System Features",
        author="Rice Leaf Disease Detector",
    )
    story = []
    page_width = A4[0] - 4 * cm

    def add_image(path: Path | None, caption_text: str = "", max_height: float = 16 * cm) -> None:
        if path and path.exists():
            img = Image(str(path))
            ratio = min(page_width / img.drawWidth, max_height / img.drawHeight, 1.0)
            img.drawWidth *= ratio
            img.drawHeight *= ratio
            story.append(img)
            if caption_text:
                story.append(Paragraph(caption_text, caption))
        else:
            story.append(Paragraph(f"<i>[Screenshot missing: {caption_text}]</i>", body))

    def bullets(items: list[str]) -> None:
        for item in items:
            story.append(Paragraph(f"• {item}", bullet))

    def green_table(data: list[list[str]], col_widths: list[float]) -> Table:
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#198754")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9.5),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        return t

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("RiceGuard", title_style))
    story.append(Paragraph(
        "<b>Rice Leaf Disease Detection System</b><br/>Complete Features Documentation",
        ParagraphStyle("Sub", parent=styles["Heading2"], alignment=TA_CENTER, fontSize=14),
    ))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}<br/>"
        "Platform: Django + scikit-learn Random Forest AI Pipeline",
        ParagraphStyle("Meta", parent=body, alignment=TA_CENTER),
    ))
    story.append(PageBreak())

    # ── Table of contents ────────────────────────────────────────────────────
    story.append(Paragraph("Table of Contents", h1))
    bullets([
        "1. System Overview",
        "2. AI Pipeline (4 Layers)",
        "3. Home Page — Upload & Scan",
        "4. Scan Result Page",
        "5. Scan History & Location Map",
        "6. Multi-Language Support (EN / Tagalog / Ilocano)",
        "7. REST API",
        "8. Django Admin Panel",
        "9. Site Settings (Admin)",
        "10. Leaf Scan Management (Admin)",
        "11. Supported Diseases & Farmer Guides",
        "12. Model Performance",
        "13. Training & Retraining",
    ])
    story.append(PageBreak())

    # ── 1. Overview ──────────────────────────────────────────────────────────
    story.append(Paragraph("1. System Overview", h1))
    story.append(Paragraph(
        "RiceGuard is a web-based rice leaf disease detection system designed for farmers "
        "and agricultural technicians in the Philippines. Users upload a photo of a rice leaf, "
        "and the system runs a multi-layer Random Forest AI pipeline to determine health status, "
        "identify specific diseases, and provide PhilRice-aligned treatment recommendations "
        "in English, Tagalog, or Ilocano.",
        body,
    ))
    add_image(arch_diag, "Figure 1: System architecture showing all major components.")
    story.append(PageBreak())

    # ── 2. Pipeline ──────────────────────────────────────────────────────────
    story.append(Paragraph("2. AI Pipeline (4 Layers)", h1))
    story.append(Paragraph(
        "Every uploaded image passes through four sequential analysis layers "
        "powered by a Random Forest voting ensemble.",
        body,
    ))
    add_image(pipeline_diag, "Figure 2: The 4-layer AI detection pipeline.")
    story.append(Paragraph("Layer Details", h2))
    layer_details = [
        "<b>Layer 0 — Rice Leaf Gate:</b> Validates that the uploaded image is a rice leaf "
        "using color/shape heuristics and a Random Forest gate model trained on rice vs. "
        "non-rice images. Hard rejections (not_rice / wrong_shape) stop the pipeline immediately "
        "and return an advisory result. Soft flags (low_confidence) continue analysis with a "
        "caution banner displayed on the result page.",

        "<b>Layer 1 — Health Check:</b> Classifies the leaf as Healthy or Unhealthy using a "
        "soft-voting ensemble: RandomForest (weight 2) + ExtraTrees + GradientBoosting + "
        "calibrated LinearSVC, with Test-Time Augmentation (TTA) averaging probabilities "
        "over 5 augmented variants. A minimum confidence threshold prevents uncertain "
        "predictions from reaching the next layer.",

        "<b>Layer 2 — Disease Identification:</b> If the leaf is unhealthy, identifies the "
        "specific disease (e.g., brown spot, blast, bacterial leaf blight) using the same "
        "voting ensemble with confidence scoring and margin checks. If the top two classes "
        "are nearly tied, the pipeline reports 'unknown' rather than guessing.",

        "<b>Layer 3 — Recommendation:</b> Returns a structured farmer guide with immediate "
        "actions, full treatment procedure, cultural management, FPA-registered pesticide "
        "options, spray guidelines, and prevention advice — all based on PhilRice publications "
        "and delivered in the user's selected language (EN / Tagalog / Ilocano).",
    ]
    for item in layer_details:
        story.append(Paragraph(item, body))
        story.append(Spacer(1, 5))
    story.append(Paragraph("RF Ensemble Details", h2))
    bullets([
        "Features: ~427 dimensions — color histograms, LBP texture, GLCM, HOG, and edge maps (ml/preprocessor.py).",
        "Ensemble: RandomForest (weight 2) + ExtraTrees + GradientBoosting + calibrated LinearSVC inside a VotingClassifier.",
        "StandardScaler → PCA (95% variance retained) applied before classification.",
        "SMOTE oversampling on the final training fit for Layers 1 and 2 (Layer 0 gate skips SMOTE).",
        "Hyperparameter CV: StratifiedGroupKFold (5-fold) over n_estimators × max_depth × min_samples_leaf, scored by macro-F1.",
        "TTA: 5 augmentation variants (original + crop/jitter/blur/flip combinations) — probabilities averaged.",
        "Acceptance uses max(TTA, single-pass) confidence and margin to match held-out evaluation.",
    ])
    story.append(PageBreak())

    # ── 3. Home page ─────────────────────────────────────────────────────────
    story.append(Paragraph("3. Home Page — Upload & Scan", h1))
    story.append(Paragraph("Key features on the home page:", body))
    bullets([
        "Hero banner with customizable title and tagline (editable in Admin → Site Settings).",
        "Drag-and-drop upload zone with file browser and camera capture support.",
        "Layer 0 pre-upload validation — checks the image before the full analysis runs.",
        "Image preview with filename, remove/change options, and warning alerts.",
        "Accepted formats: JPG, PNG, WEBP (max 10 MB).",
        "Recent scans sidebar showing the last 5 scans with health badges.",
        "'How it works' section explaining the 3-step user flow.",
        "Language switcher in the navbar — English / Filipino (Tagalog) / Ilocano.",
    ])
    add_image(shots.get("01_home"), "Figure 3: Home page with upload form and recent scans.")
    story.append(PageBreak())

    # ── 4. Result page ───────────────────────────────────────────────────────
    story.append(Paragraph("4. Scan Result Page", h1))
    story.append(Paragraph(
        "After analysis, users see a detailed result page with health status, disease name, "
        "confidence score, treatment recommendations, and reference training samples.",
        body,
    ))
    bullets([
        "Health status hero banner (Healthy / Unhealthy / Unknown).",
        "Uploaded leaf photo displayed alongside analysis confirmation.",
        "Summary grid: result, disease name, scan timestamp, and capture location with embedded map.",
        "PhilRice-aligned farmer guide with immediate actions and full treatment procedure.",
        "Expandable sections: symptoms, cultural management, pesticides, spray guidelines, prevention.",
        "Reference sample photos from the training dataset for visual comparison.",
        "Best training match with similarity score.",
        "Gate warning banner when Layer 0 had low confidence (lenient pass).",
        "Not-rice rejection page when Layer 0 hard-rejects the image.",
        "Print-friendly layout with one-click print button.",
        "All text translated when language cookie is set to Tagalog or Ilocano.",
    ])
    add_image(shots.get("04_result_diseased"), "Figure 4: Disease detection result (diseased leaf example).", max_height=14 * cm)
    add_image(shots.get("05_result_healthy"), "Figure 5: Healthy leaf result with maintenance guidance.", max_height=14 * cm)
    story.append(PageBreak())

    # ── 5. History & locations ────────────────────────────────────────────────
    story.append(Paragraph("5. Scan History & Location Map", h1))
    story.append(Paragraph(
        "All past scans are stored and accessible from the History and Locations pages.",
        body,
    ))
    bullets([
        "Dashboard statistics: total scans, healthy count, unhealthy count.",
        "Filter by health status: All, Healthy, or Unhealthy.",
        "Card grid with thumbnail, badge, disease name, and date for each scan.",
        "Click any card to view the full result page.",
        "Scan Locations page (/locations/) — interactive OpenStreetMap showing all scan pins "
        "color-coded by health status.",
        "Location captured from the browser's Geolocation API at upload time (user permission required).",
        "Reverse-geocoded location name stored with each scan via OpenStreetMap Nominatim.",
        "Each scan detail page shows an embedded map when coordinates are available.",
    ])
    add_image(shots.get("02_history"), "Figure 6: Scan history with statistics and filters.")
    add_image(shots.get("03_scan_locations"), "Figure 7: Scan locations map.")
    story.append(PageBreak())

    # ── 6. i18n ───────────────────────────────────────────────────────────────
    story.append(Paragraph("6. Multi-Language Support (EN / Tagalog / Ilocano)", h1))
    story.append(Paragraph(
        "RiceGuard supports three languages selectable at any time from the navbar language "
        "switcher. The chosen language is stored in the <i>riceguard_lang</i> browser cookie "
        "and persists across sessions.",
        body,
    ))
    lang_data = [
        ["Cookie value", "Language",              "Notes"],
        ["en",           "English",               "Default for new visitors"],
        ["fil",          "Filipino (Tagalog)",     "Full UI + disease guides translated"],
        ["ilo",          "Ilocano",                "Full UI + disease guides via fil_ilocano_dict glossary"],
    ]
    story.append(green_table(lang_data, [3 * cm, 4.5 * cm, 8 * cm]))
    story.append(Spacer(1, 10))
    bullets([
        "UI strings (buttons, labels, alerts, error messages) — detection/i18n.py.",
        "Disease guides (Tagalog) — detection/i18n_guides.py.",
        "Disease guides (Ilocano) — detection/i18n_guides_ilo.py.",
        "Pesticide section — detection/i18n_pesticides.py.",
        "JavaScript alert strings are injected per-language so browser-side validation "
        "messages also appear in the selected language.",
        "Template tag {% ui %} renders the correct string for the active language.",
        "Language switch URL: GET /lang/<lang>/  (en | fil | ilo).",
    ])
    story.append(PageBreak())

    # ── 7. API ────────────────────────────────────────────────────────────────
    story.append(Paragraph("7. REST API", h1))
    story.append(Paragraph(
        "RiceGuard exposes a REST API (Django REST Framework) for programmatic access.",
        body,
    ))
    api_data = [
        ["Endpoint",               "Method",    "Description"],
        ["/api/scans/",            "GET",        "List all scans (optional ?health= filter)"],
        ["/api/scans/",            "POST",       "Upload image and run full analysis pipeline"],
        ["/api/scans/<id>/",       "GET",        "Retrieve a single scan result by ID"],
        ["/api/scan/",             "POST",       "Alias for POST /api/scans/"],
        ["/api/validate-layer0/",  "POST",       "Pre-upload Layer 0 validation check only"],
        ["/api/locate/",           "POST",       "Reverse-geocode coordinates to a location name"],
        ["/lang/<lang>/",          "GET",        "Set language cookie (en | fil | ilo) and redirect"],
    ]
    story.append(green_table(api_data, [4.5 * cm, 2 * cm, 9 * cm]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Scan API response fields: id, image_url, uploaded_at, result_health, disease_name, "
        "confidence, confidence_percent, recommendation, gate_warning, rejection_reason, "
        "processing_error, latitude, longitude, location_name, location_display, maps_url.",
        body,
    ))
    story.append(Paragraph(
        "rejection_reason values: <b>not_rice</b> — image rejected as non-rice leaf; "
        "<b>wrong_shape</b> — wrong image shape; <b>low_confidence</b> — pipeline could not "
        "reach the confidence threshold; empty string for a normal result.",
        body,
    ))
    story.append(PageBreak())

    # ── 8. Admin ──────────────────────────────────────────────────────────────
    story.append(Paragraph("8. Django Admin Panel", h1))
    story.append(Paragraph(
        "Administrators access the Django Admin at /admin/ to manage site content "
        "and review all scans.",
        body,
    ))
    bullets([
        "Secure login with username and password.",
        "Dashboard with links to Site Settings and Leaf Scans.",
        "Branded header: 'Rice Leaf Disease Detector' / 'RiceGuard Admin'.",
    ])
    add_image(shots.get("06_admin_login"), "Figure 8: Admin login page.")
    add_image(shots.get("07_admin_dashboard"), "Figure 9: Admin dashboard.")
    story.append(PageBreak())

    # ── 9. Site settings ──────────────────────────────────────────────────────
    story.append(Paragraph("9. Site Settings (Admin)", h1))
    story.append(Paragraph(
        "Site Settings is a singleton model — only one record exists and it controls "
        "all homepage branding.",
        body,
    ))
    bullets([
        "<b>Hero Title:</b> Main headline on the home page (default: 'Rice Leaf Disease Detector').",
        "<b>Hero Tagline:</b> Subtitle text shown under the headline.",
        "<b>Site Brand:</b> Name displayed in the navbar and browser tab (default: 'RiceGuard').",
        "Cannot add duplicate settings or delete the settings record.",
    ])
    add_image(shots.get("08_admin_site_settings"), "Figure 10: Site settings editor in Admin.")
    story.append(PageBreak())

    # ── 10. Leaf scan admin ───────────────────────────────────────────────────
    story.append(Paragraph("10. Leaf Scan Management (Admin)", h1))
    story.append(Paragraph(
        "Every user scan is stored as a LeafScan record and fully manageable in the Admin panel.",
        body,
    ))
    bullets([
        "List view with thumbnail, health status, disease name, confidence %, and upload date.",
        "Filter sidebar by health status and upload date.",
        "Search by disease name and recommendation text.",
        "Detail view with large image preview and all scan fields.",
        "Stored fields: image, result_health, disease_name, confidence, all_disease_probs (JSON), "
        "recommendation, processing_error, gate_warning, rejection_reason, "
        "latitude, longitude, location_name, location_source, uploaded_at.",
        "is_not_rice flag drives the warning badge in history and admin list views.",
    ])
    add_image(shots.get("09_admin_leafscans"), "Figure 11: Leaf scan list in Admin.")
    add_image(shots.get("10_admin_leafscan_detail"), "Figure 12: Individual scan detail in Admin.")
    story.append(PageBreak())

    # ── 11. Diseases ──────────────────────────────────────────────────────────
    story.append(Paragraph("11. Supported Diseases & Farmer Guides", h1))
    story.append(Paragraph(
        "The system detects multiple rice diseases and provides detailed PhilRice-aligned "
        "management guides in English, Filipino (Tagalog), and Ilocano.",
        body,
    ))
    bullets([
        "Bacterial Leaf Blight (BLB)", "Bacterial Leaf Streak", "Brown Spot",
        "False Smut", "Leaf Blast", "Leaf Scald", "Leaf Smut",
        "Narrow Brown Spot", "Sheath Blight", "Sheath Rot", "Tungro",
        "Healthy (maintenance guidance)",
    ])
    story.append(Spacer(1, 8))
    story.append(Paragraph("Each disease guide includes:", h2))
    bullets([
        "Disease name, pathogen, and local Filipino/Ilocano names.",
        "Summary and symptoms to confirm diagnosis.",
        "Immediate actions and complete treatment procedure.",
        "Cultural management practices.",
        "FPA-registered pesticide options with rates and timing.",
        "Spray guidelines and prevention measures.",
        "Resistant rice variety recommendations.",
        "Farmer tips and PhilRice source citation.",
    ])
    story.append(PageBreak())

    # ── 12. Performance ───────────────────────────────────────────────────────
    story.append(Paragraph("12. Model Performance", h1))
    story.append(Paragraph(
        "Models are trained on 9,129 deduplicated images using the Random Forest voting "
        "ensemble with Test-Time Augmentation. Results on the held-out test split (20%):",
        body,
    ))
    perf_data = [
        ["Metric",      "Health Model", "Disease Model"],
        ["Accuracy",    "94.03%",       "90.48%"],
        ["Precision",   "94.02%",       "91.53%"],
        ["Recall",      "94.03%",       "90.48%"],
        ["F1-Score",    "94.00%",       "90.98%"],
        ["Test samples","1,826",        "1,177"],
    ]
    perf_table = Table(perf_data, colWidths=[4 * cm, 5 * cm, 5 * cm])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#198754")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ALIGN",      (1, 1), (-1, -1), "CENTER"),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "<b>Dataset:</b> 9,129 deduplicated images (20,634 raw files; 8,035 removed — "
        "7,528 byte-identical MD5 duplicates + 507 dHash re-encoded duplicates).<br/>"
        "<b>Classes:</b> healthy, bacterial blight, blast, brown spot "
        "(train-only extras: physiological stress, senescent healthy, not-rice).<br/>"
        "<b>Model files:</b> rf_health.pkl, rf_disease.pkl, rf_rice_gate.pkl<br/>"
        "<b>Training script:</b> ml/train_models.py  |  "
        "<b>Evaluation:</b> ml/evaluate_models.py",
        body,
    ))
    story.append(PageBreak())

    # ── 13. Training ──────────────────────────────────────────────────────────
    story.append(Paragraph("13. Training & Retraining", h1))
    story.append(Paragraph(
        "All three model layers are trained from a single command. "
        "The script discovers images automatically under the data root.",
        body,
    ))
    story.append(Paragraph("Quick-start commands", h2))
    cmds = [
        ("# Full training (RF models)", True),
        ("python ml/train_models.py --data data/sources", False),
        ("", None),
        ("# Quick smoke-test run (~600 images)", True),
        ("python ml/train_models.py --data data/sources --quick", False),
        ("", None),
        ("# Post-train evaluation", True),
        ("python ml/evaluate_models.py --data data/sources", False),
        ("", None),
        ("# Validate preprocessing consistency", True),
        ("python ml/validate_preprocessing.py --data data/sources", False),
    ]
    comment_style = ParagraphStyle(
        "Comment", parent=code_style, textColor=colors.HexColor("#6c757d"),
    )
    for text, is_comment in cmds:
        if text == "":
            story.append(Spacer(1, 4))
        elif is_comment:
            story.append(Paragraph(text, comment_style))
        else:
            story.append(Paragraph(text, code_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Training Pipeline Details", h2))
    bullets([
        "Data split: 70% train / 10% validation / 20% test (stratified by class).",
        "5-fold StratifiedGroupKFold cross-validation on path-level groups "
        "(prevents augmented copies leaking into the validation fold).",
        "Augmentation: random crop, colour jitter, blur/sharpen, Gaussian noise — "
        "scales with dataset size (more augmentation when data is scarce).",
        "On-the-fly augmentation applied to the final training fit only; "
        "validation and test sets remain single-pass.",
        "Hyperparameter CV tunes RF alone (fast) then reuses the best settings "
        "inside the full voting ensemble.",
        "Per-feature-block importance (color, LBP, GLCM, HOG, edges) is projected "
        "back from PCA components and printed for interpretability after each run.",
        "Training report saved to ml/saved_models/training_report.json; "
        "validation summary to ml/saved_models/validation_summary.json.",
    ])

    doc.build(story)
    return OUTPUT_PDF


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()
    print("Resolving scan IDs from database...")
    diseased_id, healthy_id = resolve_scan_ids()
    print(f"  Diseased scan: {diseased_id}, Healthy scan: {healthy_id}")

    print("Creating diagrams...")
    pipeline = create_pipeline_diagram()
    architecture = create_architecture_diagram()

    server = None
    shots: dict[str, Path] = {}
    try:
        print("Starting Django server...")
        server = start_server()
        print("Capturing screenshots...")
        shots = capture_screenshots(diseased_id, healthy_id)
        print(f"  Captured {len(shots)} screenshots.")
    finally:
        if server:
            server.terminate()
            server.wait(timeout=5)

    print("Building PDF...")
    pdf_path = build_pdf(shots, pipeline, architecture)
    print(f"\nDone! PDF saved to:\n  {pdf_path}")


if __name__ == "__main__":
    main()
