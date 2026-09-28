"""
Generate the RiceGuard USER MANUAL (PDF) with annotated screenshots.

Run from the project root:
    python docs/generate_user_manual.py

What it does
------------
1. Starts a temporary Django dev server on port 8766.
2. Drives every page with Playwright (Chromium, headless), capturing
   screenshots of each feature. Key controls are marked with numbered
   red callouts so the reader can match them to the legend tables.
3. Builds docs/RiceGuard_User_Manual.pdf with ReportLab.
4. Deletes the demo scan it created during the upload walkthrough.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
SHOT_DIR = DOCS_DIR / "manual_screenshots"
OUTPUT_PDF = DOCS_DIR / "RiceGuard_User_Manual.pdf"   # renamed after the site brand at runtime
BRAND = "RiceGuard"

PORT = 8766
BASE_URL = f"http://127.0.0.1:{PORT}"
HOST = "127.0.0.1"
ADMIN_USER = "pc"
ADMIN_PASS = "docgen_temp_2026"

SCALE = 1.5                       # device scale factor for crisp screenshots
CSS_PX_PER_INCH = 96              # a screenshot printed at "on-screen size"
FIG_MAX_H_CM = 22.5               # tallest printed figure: a full A4 page with heading and caption
DESKTOP = {"width": 1280, "height": 900}
MOBILE = {"width": 390, "height": 844}

RED = (220, 38, 38)
WHITE = (255, 255, 255)

DISEASED_IMAGE = PROJECT_ROOT / "test_brownspot.png"
DEMO_GEO = {"latitude": 17.1125, "longitude": 120.4375}   # Ilocos Sur (rice area)
# A barnyard-grass photo the Layer 0 heuristic flags (used for the warning demo)
NOT_RICE_IMAGE = PROJECT_ROOT / "downloads" / "rice_lookalikes" / "01_echinochloa_crus_galli" / "obs103237190_p172735828.jpg"

created_scan_ids: list[int] = []


# ═══════════════════════════════════════════════════════════════════════════
# Django helpers
# ═══════════════════════════════════════════════════════════════════════════

def django_setup() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    sys.path.insert(0, str(PROJECT_ROOT))
    import django

    django.setup()


def pick_sample_scans() -> dict:
    """Choose existing scans that best illustrate each result type."""
    from detection.models import LeafScan

    def first(qs):
        obj = qs.first()
        return obj.pk if obj else None

    diseased = first(
        LeafScan.objects.filter(result_health="diseased", confidence__gte=0.8)
        .exclude(latitude=None).order_by("-pk")
    ) or first(LeafScan.objects.filter(result_health="diseased").order_by("-pk"))
    healthy = first(
        LeafScan.objects.filter(result_health="healthy", confidence__gte=0.8)
        .exclude(latitude=None).order_by("-pk")
    ) or first(LeafScan.objects.filter(result_health="healthy").order_by("-pk"))
    not_rice_obj = (
        LeafScan.objects.filter(rejection_reason__in=["not_rice", "wrong_shape"])
        .order_by("-pk").first()
    )
    warned = first(
        LeafScan.objects.exclude(gate_warning="")
        .filter(result_health__in=["healthy", "diseased"]).order_by("-pk")
    )
    return {
        "diseased": diseased,
        "healthy": healthy,
        "not_rice": not_rice_obj.pk if not_rice_obj else None,
        "not_rice_image": not_rice_obj.image.path if not_rice_obj and not_rice_obj.image else None,
        "warned": warned,
    }


def disease_reference() -> list[dict]:
    from detection.disease_guide import get_farmer_guide

    rows = []
    for key in ("brown spot", "blast", "bacterial blight"):
        g = get_farmer_guide(key)
        if not g:
            continue
        rows.append({
            "name": g.name,
            "pathogen": g.pathogen,
            "type": g.disease_type,
            "summary": g.summary,
            "symptoms": list(g.symptoms)[:3],
        })
    return rows


RUN_STARTED = None


def cleanup_created_scans() -> None:
    from detection.models import LeafScan

    demo = set(created_scan_ids)
    if RUN_STARTED is not None:
        demo.update(LeafScan.objects.filter(uploaded_at__gte=RUN_STARTED).values_list("pk", flat=True))
    for pk in sorted(demo):
        try:
            scan = LeafScan.objects.get(pk=pk)
        except LeafScan.DoesNotExist:
            continue
        try:
            if scan.image and os.path.exists(scan.image.path):
                os.remove(scan.image.path)
        except OSError:
            pass
        scan.delete()
        print(f"    removed demo scan #{pk}")


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "manage.py", "runserver", f"{HOST}:{PORT}", "--noreload"],
        cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            urllib.request.urlopen(BASE_URL + "/", timeout=2)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("Django server did not start in time")


# ═══════════════════════════════════════════════════════════════════════════
# Image helpers (callouts, crops)
# ═══════════════════════════════════════════════════════════════════════════

def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    for base in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")):
        p = base / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def annotate(src: Path, dst: Path, boxes: list[tuple[int, dict]], offset=(0, 0)) -> Path:
    """Draw numbered red callouts. boxes = [(number, {x,y,width,height}), ...] in CSS px."""
    im = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _font(int(15 * SCALE))
    r = int(14 * SCALE)
    pad = int(5 * SCALE)
    lw = max(2, int(2.5 * SCALE))
    for number, box in boxes:
        if not box:
            continue
        x = (box["x"] - offset[0]) * SCALE
        y = (box["y"] - offset[1]) * SCALE
        w = box["width"] * SCALE
        h = box["height"] * SCALE
        x0, y0 = max(x - pad, 1), max(y - pad, 1)
        x1, y1 = min(x + w + pad, im.width - 2), min(y + h + pad, im.height - 2)
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rounded_rectangle([x0, y0, x1, y1], radius=int(6 * SCALE), outline=RED, width=lw)
        cx = max(x0, r + 2)
        cy = max(y0, r + 2)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED, outline=WHITE, width=max(2, int(1.5 * SCALE)))
        label = str(number)
        bb = draw.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((cx - tw / 2 - bb[0], cy - th / 2 - bb[1]), label, fill=WHITE, font=font)
    im.save(dst)
    return dst


def crop(src: Path, dst: Path, region: dict, pad: int = 12) -> tuple[Path, tuple[int, int]]:
    """Crop a CSS-px region out of a screenshot. Returns (path, css offset)."""
    im = Image.open(src)
    x0 = max(int((region["x"] - pad) * SCALE), 0)
    y0 = max(int((region["y"] - pad) * SCALE), 0)
    x1 = min(int((region["x"] + region["width"] + pad) * SCALE), im.width)
    y1 = min(int((region["y"] + region["height"] + pad) * SCALE), im.height)
    im.crop((x0, y0, x1, y1)).save(dst)
    return dst, (x0 / SCALE, y0 / SCALE)


def to_jpeg(src: Path, quality: int = 82) -> Path:
    """Re-encode a PNG screenshot as JPEG for a much smaller PDF."""
    if src.suffix.lower() in (".jpg", ".jpeg"):
        return src
    dst = src.with_name(src.stem + "_pdf.jpg")
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        Image.open(src).convert("RGB").save(dst, "JPEG", quality=quality, optimize=True)
    return dst


def cut_rows(im: Image.Image, n: int, max_h: int) -> list[tuple[int, int]]:
    """Row ranges that split an image into n parts no taller than max_h rows.

    Each cut lands in the widest blank band near an even split, so parts
    break between cards rather than between two lines of one paragraph."""
    import numpy as np

    if n <= 1:
        return [(0, im.height)]
    g = np.asarray(im.convert("L"), dtype=np.int16)
    detail = np.abs(np.diff(g, axis=1)).sum(axis=1).astype(float)    # detail across each row
    detail = np.minimum(detail / (np.percentile(detail, 90) or 1.0), 1.0)

    def clear(r: int):             # worst detail within r rows either side
        return np.lib.stride_tricks.sliding_window_view(np.pad(detail, r, mode="edge"), 2 * r + 1).max(axis=1)

    busy = (clear(4) + clear(12) + clear(30)) / 3
    cuts = [0]
    for k in range(1, n):
        lo = max(cuts[-1] + 1, im.height - (n - k) * max_h)
        hi = min(cuts[-1] + max_h, im.height - 1)
        target = cuts[-1] + (im.height - cuts[-1]) / (n - k + 1)
        rows = np.arange(lo, hi + 1)
        cost = busy[lo:hi + 1] + np.abs(rows - target) / (im.height / n)
        cuts.append(int(rows[cost.argmin()]))
    cuts.append(im.height)
    return list(zip(cuts, cuts[1:]))


def split_tall(src: Path, max_ratio: float = 1.25) -> list[Path]:
    """Split a very tall screenshot into parts (height <= width*max_ratio)."""
    im = Image.open(src)
    max_h = int(im.width * max_ratio)
    ranges = cut_rows(im, math.ceil(im.height / max_h), max_h)
    if len(ranges) == 1:
        return [src]
    parts = []
    for n, (y0, y1) in enumerate(ranges, 1):
        part = src.with_name(f"{src.stem}_part{n}.png")
        im.crop((0, y0, im.width, y1)).save(part)
        parts.append(part)
    return parts


def tighten(im: Image.Image, margin: int = 24) -> Image.Image:
    """Spend the printed width on content: crop plain page background around
    the screenshot and, in wide strips (toolbar, filter bar, navbar), shorten
    long empty stretches between the controls."""
    import numpy as np

    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    colours, counts = np.unique(border, axis=0, return_counts=True)
    ink = np.abs(a - colours[counts.argmax()]).max(axis=2) > 8
    ys, xs = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    if ys.size:
        a = a[max(ys[0] - margin, 0):ys[-1] + margin + 1, max(xs[0] - margin, 0):xs[-1] + margin + 1]
    if a.shape[0] < 0.2 * a.shape[1]:
        repeats = np.abs(np.diff(a, axis=1)).max(axis=(0, 2)) <= 4   # column x+1 looks like column x
        keep = np.ones(a.shape[1], bool)
        run = 0
        for x, same in enumerate(repeats, 1):
            run = run + 1 if same else 0
            keep[x] = run <= margin
        a = a[:, keep]
    return Image.fromarray(a.astype(np.uint8))


def layout_figure(src: Path, max_w: float, max_h: float, inch: float) -> list[tuple[Path, float, float]]:
    """Plan a screenshot for print as [(jpeg, width, height), ...].

    Lengths use the caller's unit (`inch` is one inch in it). The figure fills
    max_w but is never printed larger than on screen. A figure too tall for
    one page may shrink to 85% of that width (or 75% of on-screen size);
    beyond that it is cut at blank rows into the fewest parts that fit at
    such a width, and printed as wide as its tallest part allows."""
    im = tighten(Image.open(src))
    real_w = im.width / SCALE / CSS_PX_PER_INCH * inch
    best_w = min(max_w, real_w)
    min_w = min(0.85 * best_w, 0.75 * real_w)
    part_h = int(max_h / min_w * im.width)        # tallest part (px) that fits at min_w
    ranges = cut_rows(im, math.ceil(im.height / part_h), part_h)
    tallest = max(y1 - y0 for y0, y1 in ranges)
    w = min(best_w, max_h / tallest * im.width)
    out = []
    for i, (y0, y1) in enumerate(ranges, 1):
        dst = src.with_name(f"{src.stem}_print{i if len(ranges) > 1 else ''}.jpg")
        im.crop((0, y0, im.width, y1)).save(dst, "JPEG", quality=85, optimize=True)
        out.append((dst, w, w * (y1 - y0) / im.width))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Screenshot capture
# ═══════════════════════════════════════════════════════════════════════════

class Shooter:
    def __init__(self, page):
        self.page = page

    def box(self, selector: str):
        try:
            el = self.page.query_selector(selector)
            if el is None:
                return None
            b = el.bounding_box()
            return b
        except Exception:
            return None

    def boxes(self, spec: list[tuple[int, str]]):
        return [(n, self.box(sel)) for n, sel in spec]

    def goto(self, path: str, wait: float = 1.2, until: str = "domcontentloaded"):
        # External map tiles / iframes can keep "load" pending forever in
        # headless mode, so wait for DOM only and then settle for a moment.
        self.page.goto(BASE_URL + path, wait_until="domcontentloaded", timeout=60000)
        try:
            self.page.wait_for_load_state("load", timeout=8000)
        except Exception:
            pass
        self.page.evaluate("window.scrollTo(0, 0)")
        time.sleep(wait)

    def full(self, name: str, callouts: list[tuple[int, str]] | None = None, clip_height: int | None = None) -> Path:
        """Full-page screenshot (optionally clipped to the top N css px) with callouts."""
        self.page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.2)
        raw = SHOT_DIR / f"{name}_raw.png"
        if clip_height:
            self.page.screenshot(path=str(raw), clip={"x": 0, "y": 0, "width": self.page.viewport_size["width"], "height": clip_height}, full_page=True)
        else:
            self.page.screenshot(path=str(raw), full_page=True)
        out = SHOT_DIR / f"{name}.png"
        if callouts:
            annotate(raw, out, self.boxes(callouts))
        else:
            shutil.copy(raw, out)
        print(f"    captured {name}.png")
        return out

    def region(self, name: str, region_selector: str, callouts: list[tuple[int, str]] | None = None, pad: int = 14) -> Path | None:
        """Screenshot of one element's area (with padding) plus callouts."""
        self.page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.2)
        reg = self.box(region_selector)
        if not reg:
            print(f"    !! region not found: {region_selector}")
            return None
        raw = SHOT_DIR / f"{name}_raw.png"
        self.page.screenshot(path=str(raw), full_page=True)
        cropped = SHOT_DIR / f"{name}_crop.png"
        _, offset = crop(raw, cropped, reg, pad=pad)
        out = SHOT_DIR / f"{name}.png"
        annotate(cropped, out, self.boxes(callouts or []), offset=offset)
        print(f"    captured {name}.png")
        return out

    def viewport(self, name: str, callouts: list[tuple[int, str]] | None = None) -> Path:
        raw = SHOT_DIR / f"{name}_raw.png"
        self.page.screenshot(path=str(raw), full_page=False)
        out = SHOT_DIR / f"{name}.png"
        sy = self.page.evaluate("window.scrollY")
        annotate(raw, out, self.boxes(callouts or []), offset=(0, sy))
        print(f"    captured {name}.png")
        return out


def capture_all(samples: dict) -> dict[str, Path | list[Path] | None]:
    from playwright.sync_api import sync_playwright

    shots: dict = {}
    scratch_txt = SHOT_DIR / "not_an_image.txt"
    scratch_txt.write_text("This is not an image file.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # ── Desktop context, English UI ──────────────────────────────────
        ctx = browser.new_context(
            viewport=DESKTOP, device_scale_factor=SCALE, locale="en-PH",
            geolocation=DEMO_GEO, permissions=["geolocation"],
        )
        ctx.add_cookies([{"name": "riceguard_lang", "value": "en", "domain": HOST, "path": "/"}])
        page = ctx.new_page()
        s = Shooter(page)

        # 1. Home page overview
        s.goto("/")
        shots["home"] = s.full("01_home", [
            (1, ".navbar-brand"),
            (2, "#navMenu > ul"),
            (3, ".hero-banner"),
            (4, "#drop-zone"),
            (5, "#id_image"),
            (6, "#use-location-btn"),
            (7, "#camera-btn"),
            (8, ".recent-scans-card"),
            (9, ".row.mt-5"),
        ])

        # 2. Navigation bar close-up
        nav_links = page.query_selector_all("#navMenu a.nav-link")
        nav_callouts = [(1, ".navbar-brand")]
        for i in range(len(nav_links)):
            nav_callouts.append((i + 2, f"#navMenu a.nav-link >> nth={i}"))
        nav_callouts.append((len(nav_links) + 2, "#langMenuBtn"))
        shots["navbar"] = s.region("02_navbar", "nav.navbar", nav_callouts, pad=4)

        # 3. Language menu open
        page.click("#langMenuBtn")
        time.sleep(0.6)
        raw = SHOT_DIR / "03_language_menu_raw.png"
        page.screenshot(path=str(raw), clip={"x": 0, "y": 0, "width": DESKTOP["width"], "height": 330})
        shots["lang_menu"] = annotate(raw, SHOT_DIR / "03_language_menu.png", [
            (1, s.box("#langMenuBtn")),
            (2, s.box(".lang-menu-option >> nth=0")),
            (3, s.box(".lang-menu-option >> nth=1")),
            (4, s.box(".lang-menu-option >> nth=2")),
        ])
        page.keyboard.press("Escape")
        print("    captured 03_language_menu.png")

        # 4. Upload walkthrough: choose a rice-leaf photo → Layer 0 check → ready
        page.set_input_files("#id_image", str(DISEASED_IMAGE))
        try:
            page.wait_for_selector("#preview-processing:not(.d-none)", timeout=5000)
            shots["checking"] = s.region("04a_checking_image", ".col-lg-7 > .card", [
                (1, "#preview-processing"),
            ])
        except Exception:
            shots["checking"] = None
        page.wait_for_selector("#analysis-actions:not(.d-none)", timeout=120000)
        time.sleep(2.5)   # let place-name lookup / map finish
        shots["preview"] = s.region("04_upload_preview", ".col-lg-7 > .card", [
            (1, "#preview-filename"),
            (2, "#remove-preview-btn"),
            (3, "#preview-original"),
            (4, "#preview-location-card"),
            (5, "#preview-use-location-btn"),
            (6, "#change-image-btn"),
            (7, "#submit-btn"),
        ])

        # 5. Click Analyse → loading state → result
        page.click("#submit-btn")
        try:
            page.wait_for_selector("#loading-overlay:not(.d-none)", timeout=5000)
            time.sleep(0.3)
            shots["analysing"] = s.region("05_analysing", ".col-lg-7", [(1, "#loading-overlay")])
        except Exception:
            shots["analysing"] = None
        # NB: a plain glob "**/result/*" never matches the trailing slash.
        page.wait_for_url(lambda u: "/result/" in u, wait_until="commit", timeout=180000)
        page.wait_for_selector(".result-hero-title", timeout=60000)
        try:
            page.wait_for_load_state("load", timeout=8000)
        except Exception:
            pass
        try:
            new_id = int(page.url.rstrip("/").split("/")[-1])
            created_scan_ids.append(new_id)
        except ValueError:
            new_id = None
        time.sleep(0.8)
        page.evaluate("window.scrollTo(0, 0)")
        shots["result_fresh"] = s.full("06_result_after_analyse", [
            (1, ".result-toolbar"),
            (2, "xpath=//header[.//div[contains(@class,'result-hero-body')]]"),
            (3, ".scan-photo-card"),
            (4, ".card.result-panel:has(.result-summary-grid)"),
            (5, ".treatment-card"),
            (6, "#guideAccordion"),
            (7, "#printResultBtnBottom"),
        ], clip_height=None)

        # 6. Layer 0 warning: choose a photo that is not a rice leaf
        s.goto("/?new_scan=1")
        not_rice_demo = NOT_RICE_IMAGE if NOT_RICE_IMAGE.exists() else samples.get("not_rice_image")
        if not_rice_demo and os.path.exists(str(not_rice_demo)):
            page.set_input_files("#id_image", str(not_rice_demo))
            try:
                page.wait_for_selector("#layer0-warning:not(.d-none), #upload-error-sign:not(.d-none)", timeout=120000)
                time.sleep(1.0)
                shots["layer0_warning"] = s.region("07_layer0_warning", ".col-lg-7 > .card", [
                    (1, "#layer0-warning:not(.d-none)"),
                    (2, "#upload-error-sign:not(.d-none)"),
                    (3, "#submit-btn"),
                    (4, "#change-image-btn"),
                ])
            except Exception as exc:
                print("    !! layer0 warning not captured:", exc)
                shots["layer0_warning"] = None

        # 7. Unsupported file error
        s.goto("/?new_scan=1")
        page.set_input_files("#id_image", str(scratch_txt))
        try:
            page.wait_for_selector("#upload-error-sign:not(.d-none)", timeout=10000)
            time.sleep(0.4)
            shots["bad_file"] = s.region("08_unsupported_file", ".col-lg-7 > .card", [
                (1, "#upload-error-sign"),
                (2, "#upload-error-dismiss"),
            ])
        except Exception:
            shots["bad_file"] = None

        # 8. Result pages (existing samples)
        hero = "xpath=//header[.//div[contains(@class,'result-hero-body')]]"
        if samples["diseased"]:
            s.goto(f"/result/{samples['diseased']}/")
            shots["result_diseased"] = s.full("09_result_diseased", [
                (1, ".result-toolbar"),
                (2, hero),
                (3, ".scan-photo-card"),
                (4, ".card.result-panel:has(.result-summary-grid)"),
                (5, ".treatment-card"),
                (6, "#guideAccordion"),
                (7, "#printResultBtnBottom"),
            ])
            shots["result_toolbar"] = s.region("10_result_toolbar", ".result-toolbar", [
                (1, ".result-toolbar a[href='/?new_scan=1']"),
                (2, ".result-toolbar a[href*='history']"),
                (3, "#printResultBtn"),
                (4, ".result-meta"),
            ], pad=6)
            shots["result_hero"] = s.region("11_result_verdict", hero, [
                (1, ".result-hero-icon"),
                (2, ".result-hero-title"),
                (3, ".result-hero-disease"),
            ])
            shots["result_photo"] = s.region("12_result_photo", ".scan-photo-card", [
                (1, ".scan-result-photo"),
                (2, ".result-photo-note"),
            ])
            shots["result_glance"] = s.region("13_result_at_a_glance", ".card.result-panel:has(.result-summary-grid)", [
                (1, ".result-summary-item >> nth=0"),
                (2, ".result-summary-item >> nth=1"),
                (3, ".result-summary-item >> nth=2"),
                (4, ".result-summary-item >> nth=3"),
            ])
            shots["result_treatment"] = s.region("14_result_treatment", ".treatment-card", [])
            # open pesticide guide
            try:
                page.click("button[data-bs-target='#guidePesticides']")
                page.wait_for_selector("#guidePesticides.show", timeout=5000)
                time.sleep(0.6)
                shots["result_pesticides"] = s.region("15_result_pesticide_guide", "#guideAccordion", [
                    (1, "button[data-bs-target='#guideSymptoms']"),
                    (2, "button[data-bs-target='#guideCultural']"),
                    (3, "button[data-bs-target='#guidePesticides']"),
                    (4, "#guidePesticides .accordion-body"),
                    (5, "button[data-bs-target='#guidePrevention']"),
                    (6, "button[data-bs-target='#guideVarieties']"),
                ])
            except Exception as exc:
                print("    !! pesticide accordion:", exc)
                shots["result_pesticides"] = None
            # photo zoom modal
            try:
                page.click(".result-photo-open")
                page.wait_for_selector("#photoPreviewModal:not([hidden])", timeout=5000)
                time.sleep(0.5)
                shots["photo_modal"] = s.viewport("17_photo_zoom", [(1, "#photoPreviewImage")])
                page.keyboard.press("Escape")
                time.sleep(0.3)
            except Exception as exc:
                print("    !! photo modal:", exc)
                shots["photo_modal"] = None
            # print layout
            try:
                page.emulate_media(media="print")
                time.sleep(0.5)
                shots["print_view"] = s.full("18_print_view", [(1, ".print-header")], clip_height=1100)
                page.emulate_media(media="screen")
            except Exception as exc:
                print("    !! print view:", exc)
                shots["print_view"] = None

        if samples["healthy"]:
            s.goto(f"/result/{samples['healthy']}/")
            shots["result_healthy"] = s.full("19_result_healthy", [
                (1, hero), (2, ".treatment-card"), (3, "#guideAccordion"),
            ], clip_height=1500)

        if samples["not_rice"]:
            s.goto(f"/result/{samples['not_rice']}/")
            shots["result_not_rice"] = s.full("20_result_not_rice", [
                (1, hero), (2, ".scan-photo-card"), (3, ".treatment-card"),
            ], clip_height=1300)

        if samples["warned"]:
            s.goto(f"/result/{samples['warned']}/")
            shots["result_warned"] = s.full("21_result_leaf_check_warning", [
                (1, ".result-alert"),
            ], clip_height=700)

        # 9. History
        s.goto("/history/")
        shots["history"] = s.full("22_history", [
            (1, "a.btn-success[href='/?new_scan=1']"),
            (2, ".row.row-cols-2"),
            (3, ".card:has(a[href='?health=healthy'])"),
            (4, ".card.scan-card:has(img.scan-thumb)"),
            (5, ".d-flex.justify-content-between.mt-4"),
        ])
        shots["history_filter"] = s.region("23_history_filters", ".card:has(a[href='?health=healthy'])", [
            (1, "a[href='/history/']"),
            (2, "a.btn[href='?health=healthy']"),
            (3, "a.btn[href='?health=diseased']"),
            (4, "a.btn[href='?health=unknown']"),
        ], pad=6)
        shots["history_card"] = s.region("24_history_scan_card", ".card.scan-card:has(img.scan-thumb)", [
            (1, ".card.scan-card:has(img.scan-thumb) img.scan-thumb"),
            (2, ".card.scan-card:has(img.scan-thumb) .badge"),
        ])
        s.goto("/history/?health=diseased")
        shots["history_diseased"] = s.full("25_history_filtered", [
            (1, ".card:has(a[href='?health=healthy'])"),
        ], clip_height=1000)

        # 10. Locations
        s.goto("/locations/", wait=3.0, until="load")
        shots["locations"] = s.full("26_locations", [
            (1, ".row:has(.locations-stat)"),
            (2, "#locations-map"),
            (3, ".card:has(#barangay-scans-chart)"),
            (4, ".card:has(#scan-timeline-chart)"),
            (5, ".card:has(.locations-rank-list)"),
            (6, ".card:has(.locations-disease-item)"),
            (7, ".card:has(.locations-legend-item):not(:has(#locations-map))"),
        ])
        shots["locations_map"] = s.region("27_locations_map", ".card:has(#locations-map)", [
            (1, "#locations-map"),
        ])
        shots["locations_charts"] = s.region("28_locations_barangay_chart", ".card:has(#barangay-scans-chart)", [])
        shots["locations_timeline"] = s.region("29_locations_timeline", ".card:has(#scan-timeline-chart)", [])
        shots["locations_rank"] = s.region("30_locations_ranked_places", ".card:has(.locations-rank-list)", [
            (1, ".locations-rank-list a >> nth=0"),
        ])
        first_place = page.query_selector(".locations-rank-list a")
        if first_place:
            href = first_place.get_attribute("href")
            if href:
                s.goto(href if href.startswith("/") else "/locations/" + href, wait=3.0, until="load")
                shots["locations_place"] = s.full("31_locations_place_detail", [], clip_height=1500)

        # 11. Scan not found + API
        s.goto("/result/999999/")
        shots["not_found"] = s.viewport("32_scan_not_found")
        s.goto("/api/scans/")
        shots["api"] = s.full("33_api_browsable", [], clip_height=900)

        # 12. Other languages
        for code, key, fname in (("fil", "home_fil", "34_home_tagalog"), ("ilo", "home_ilo", "35_home_ilocano")):
            s.goto(f"/lang/{code}/?next=/")
            shots[key] = s.full(fname, [(1, "#langMenuBtn"), (2, ".hero-banner")], clip_height=760)
        s.goto("/lang/en/?next=/")
        ctx.close()

        # ── Mobile context ───────────────────────────────────────────────
        mctx = browser.new_context(
            viewport=MOBILE, device_scale_factor=SCALE, is_mobile=True, has_touch=True,
            user_agent=("Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36"),
        )
        mctx.add_cookies([{"name": "riceguard_lang", "value": "en", "domain": HOST, "path": "/"}])
        mpage = mctx.new_page()
        ms = Shooter(mpage)
        ms.goto("/")
        shots["mobile_home"] = ms.full("36_mobile_home", [
            (1, ".navbar-toggler"),
            (2, "#drop-zone"),
            (3, "#id_image"),
            (4, "#use-location-btn"),
            (5, "#camera-btn"),
        ], clip_height=1000)
        mpage.click(".navbar-toggler")
        time.sleep(0.7)
        shots["mobile_menu"] = ms.viewport("37_mobile_menu", [(1, "#navMenu")])
        if samples["diseased"]:
            ms.goto(f"/result/{samples['diseased']}/")
            shots["mobile_result"] = ms.full("38_mobile_result", [], clip_height=1700)
        mctx.close()

        # ── Admin ────────────────────────────────────────────────────────
        actx = browser.new_context(viewport=DESKTOP, device_scale_factor=SCALE)
        apage = actx.new_page()
        a = Shooter(apage)
        a.goto("/admin/login/")
        shots["admin_login"] = a.viewport("39_admin_login", [
            (1, "#id_username"), (2, "#id_password"), (3, "input[type=submit]"),
        ])
        apage.fill("#id_username", ADMIN_USER)
        apage.fill("#id_password", ADMIN_PASS)
        apage.click("input[type=submit]")
        apage.wait_for_load_state("networkidle")
        if "/admin/login" in apage.url:
            print("    !! admin login failed — admin screenshots skipped")
        else:
            a.goto("/admin/")
            shots["admin_dashboard"] = a.viewport("40_admin_dashboard", [
                (1, ".app-detection"),
                (2, "tr.model-leafscan"),
                (3, "tr.model-sitesettings"),
                (4, "#user-tools"),
                (5, "#recent-actions-module"),
            ])
            a.goto("/admin/detection/sitesettings/1/change/")
            shots["admin_settings"] = a.full("41_admin_site_settings", [
                (1, "#id_hero_title"), (2, "#id_hero_tagline"), (3, "#id_site_brand"),
                (4, "input[name=_save]"),
            ], clip_height=900)
            a.goto("/admin/detection/leafscan/")
            shots["admin_scans"] = a.full("42_admin_leaf_scans", [
                (1, "#searchbar"),
                (2, "#changelist-filter"),
                (3, "#result_list"),
                (4, "select[name=action]"),
                (5, ".paginator"),
            ], clip_height=1300)
            if samples["diseased"]:
                a.goto(f"/admin/detection/leafscan/{samples['diseased']}/change/")
                shots["admin_scan_detail"] = a.full("43_admin_leaf_scan_detail", [
                    (1, ".field-thumbnail_large"),
                    (2, "#id_result_health"),
                    (3, "#id_disease_name"),
                    (4, "#id_recommendation"),
                    (5, ".deletelink"),
                    (6, "input[name=_save]"),
                ])
        actx.close()
        browser.close()

    return shots


# ═══════════════════════════════════════════════════════════════════════════
# Diagrams
# ═══════════════════════════════════════════════════════════════════════════

def create_workflow_diagram() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    steps = [
        ("1", f"Open {BRAND}", "in any web browser\n(phone or computer)", "#1b4332"),
        ("2", "Choose a photo", "browse, drag & drop,\nor use the camera", "#2d6a4f"),
        ("3", "Leaf check", "Layer 0 confirms it\nlooks like a rice leaf", "#40916c"),
        ("4", "Analyse Leaf", "AI checks health and\nidentifies the disease", "#52b788"),
        ("5", "Read the result", "verdict, confidence and\ntreatment guide", "#74c69d"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    for i, (num, title, desc, color) in enumerate(steps):
        x = 0.3 + i * 2.55
        ax.add_patch(FancyBboxPatch((x, 1.05), 2.2, 2.1, boxstyle="round,pad=0.05,rounding_size=0.18",
                                    facecolor=color, edgecolor="white", linewidth=2))
        ax.add_patch(plt.Circle((x + 0.32, 2.85), 0.22, color="white"))
        ax.text(x + 0.32, 2.85, num, ha="center", va="center", fontsize=11, fontweight="bold", color=color)
        ax.text(x + 1.1, 2.3, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")
        ax.text(x + 1.1, 1.6, desc, ha="center", va="center", fontsize=8.5, color="white")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 2.25, 2.1), (x + 2.52, 2.1), arrowstyle="-|>",
                                         mutation_scale=16, color="#1b4332", lw=2))
    ax.text(6.5, 0.45, "Scan → check → analyse → act.  Every scan is saved in History and, when location is on, in Locations.",
            ha="center", fontsize=9.5, color="#495057")
    out = SHOT_DIR / "00_workflow.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close(fig)
    return out


def create_photo_tips_diagram() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    good = [
        "One leaf blade fills the frame",
        "Plain background (sky, soil, paper, palm)",
        "Daylight, no harsh shadow or flash",
        "Sharp focus on the spots or lesions",
        "Hold the phone steady, 15–25 cm away",
    ]
    bad = [
        "Whole plant or many leaves at once",
        "Broad leaves of other plants, grass, weeds",
        "Blurry, dark, or backlit photos",
        "Screenshots, drawings, or edited images",
        "Files larger than 10 MB or not JPG/PNG/WEBP",
    ]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    for x, title, items, color in ((0.3, "GOOD PHOTO", good, "#2d6a4f"), (6.7, "AVOID", bad, "#b02a37")):
        ax.add_patch(FancyBboxPatch((x, 0.3), 6.0, 3.6, boxstyle="round,pad=0.05,rounding_size=0.2",
                                    facecolor="white", edgecolor=color, linewidth=2.5))
        ax.add_patch(FancyBboxPatch((x, 3.2), 6.0, 0.7, boxstyle="round,pad=0.05,rounding_size=0.2",
                                    facecolor=color, edgecolor=color, linewidth=2.5))
        ax.text(x + 3.0, 3.55, title, ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        mark = "✓" if color == "#2d6a4f" else "✗"
        for j, item in enumerate(items):
            ax.text(x + 0.35, 2.75 - j * 0.52, f"{mark}  {item}", ha="left", va="center", fontsize=10, color="#212529")
    out = SHOT_DIR / "00_photo_tips.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="#f8f9fa")
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PDF
# ═══════════════════════════════════════════════════════════════════════════

def build_pdf(shots: dict, workflow: Path, photo_tips: Path, diseases: list[dict]) -> Path:
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph as _Paragraph

    class Paragraph(_Paragraph):
        """Paragraph that swaps the generic 'RiceGuard' name for the live site brand."""
        def __init__(self, text, style, **kw):
            super().__init__(str(text).replace("RiceGuard", BRAND), style, **kw)

    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Flowable, Image, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    from reportlab.platypus.tableofcontents import TableOfContents

    # Fonts (Arial gives us ✓ ✗ → and Filipino characters)
    font_dir = Path("C:/Windows/Fonts")
    if (font_dir / "arial.ttf").exists():
        pdfmetrics.registerFont(TTFont("Manual", str(font_dir / "arial.ttf")))
        pdfmetrics.registerFont(TTFont("Manual-Bold", str(font_dir / "arialbd.ttf")))
        pdfmetrics.registerFont(TTFont("Manual-Italic", str(font_dir / "ariali.ttf")))
        pdfmetrics.registerFont(TTFont("Manual-BoldItalic", str(font_dir / "arialbi.ttf")))
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily("Manual", normal="Manual", bold="Manual-Bold", italic="Manual-Italic", boldItalic="Manual-BoldItalic")
        FONT, FONT_B = "Manual", "Manual-Bold"
    else:
        FONT, FONT_B = "Helvetica", "Helvetica-Bold"

    GREEN = colors.HexColor("#198754")
    DARK_GREEN = colors.HexColor("#1b4332")
    LIGHT = colors.HexColor("#f1f8f4")
    GREY = colors.HexColor("#6c757d")

    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName=FONT, fontSize=10.5, leading=15, spaceAfter=6)
    small = ParagraphStyle("Small", parent=body, fontSize=9, leading=12)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=FONT_B, fontSize=20, leading=24, textColor=GREEN, spaceBefore=6, spaceAfter=10, keepWithNext=True)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=FONT_B, fontSize=14, leading=18, textColor=DARK_GREEN, spaceBefore=12, spaceAfter=6, keepWithNext=True)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName=FONT_B, fontSize=11.5, leading=15, textColor=DARK_GREEN, spaceBefore=8, spaceAfter=4, keepWithNext=True)
    caption = ParagraphStyle("Caption", parent=body, fontSize=9, leading=12, textColor=GREY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=16, bulletIndent=4, spaceAfter=3)
    step_style = ParagraphStyle("Step", parent=body, leftIndent=22, bulletIndent=4, spaceAfter=4)
    cell = ParagraphStyle("Cell", parent=body, fontSize=9.5, leading=12.5, spaceAfter=0)
    cell_b = ParagraphStyle("CellB", parent=cell, fontName=FONT_B)
    title_style = ParagraphStyle("DocTitle", parent=styles["Title"], fontName=FONT_B, fontSize=30, leading=36, textColor=GREEN, alignment=TA_CENTER, spaceAfter=8)
    subtitle = ParagraphStyle("Sub", parent=body, fontName=FONT_B, fontSize=15, leading=20, alignment=TA_CENTER, textColor=DARK_GREEN)
    meta = ParagraphStyle("Meta", parent=body, alignment=TA_CENTER, textColor=GREY)

    PAGE_W = A4[0] - 3.4 * cm

    class ManualDoc(SimpleDocTemplate):
        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph):
                name = flowable.style.name
                if name in ("H1", "H2"):
                    level = 0 if name == "H1" else 1
                    text = flowable.getPlainText()
                    if text == "Contents":
                        return
                    key = f"toc-{level}-{abs(hash(text))}-{self.page}"
                    self.canv.bookmarkPage(key)
                    self.canv.addOutlineEntry(text, key, level=level, closed=False)
                    self.notify("TOCEntry", (level, text, self.page, key))

    class FitFigure(Flowable):
        """A picture (image or table of images) with its caption, kept on one page.
        Near the foot of a page the picture may print down to min_scale of its
        size instead of moving to the next page and leaving a gap."""

        def __init__(self, body, caption_para, min_scale=1.0):
            super().__init__()
            self.body, self.caption, self.min_scale = body, caption_para, min_scale

        def wrap(self, aW, aH):
            self.bw, self.bh = self.body.wrap(aW, aH)
            self.caption.wrap(aW, aH)
            cap = self.caption.height + self.caption.getSpaceBefore() + self.caption.getSpaceAfter()
            fit = (aH - cap) / self.bh
            self.scale = fit if self.min_scale <= fit < 1 else 1.0
            self.width, self.height = aW, self.bh * self.scale + cap
            return self.width, self.height

        def draw(self):
            s = self.scale
            self.canv.saveState()
            self.canv.translate((self.width - self.bw * s) / 2, self.height - self.bh * s)
            self.canv.scale(s, s)
            self.body.drawOn(self.canv, 0, 0)
            self.canv.restoreState()
            self.caption.drawOn(self.canv, 0, self.caption.getSpaceAfter())

    def on_page(canv, doc):
        canv.saveState()
        canv.setFont(FONT, 8.5)
        canv.setFillColor(GREY)
        canv.drawString(1.7 * cm, 1.1 * cm, f"{BRAND} — User Manual")
        canv.drawRightString(A4[0] - 1.7 * cm, 1.1 * cm, f"Page {doc.page}")
        canv.setStrokeColor(colors.HexColor("#d0e3d7"))
        canv.line(1.7 * cm, 1.5 * cm, A4[0] - 1.7 * cm, 1.5 * cm)
        canv.restoreState()

    doc = ManualDoc(
        str(OUTPUT_PDF), pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.8 * cm, bottomMargin=2.0 * cm,
        title=f"{BRAND} User Manual", author=f"{BRAND} — Rice Leaf Disease Detector",
        subject=f"How to use the {BRAND} rice leaf disease detection system",
    )
    story: list = []
    fig_no = [0]

    # ── helpers ──────────────────────────────────────────────────────────
    def P(text, style=body):
        story.append(Paragraph(text, style))

    def bullets(items, style=bullet):
        for item in items:
            story.append(Paragraph(item, style, bulletText="•"))

    def steps(items):
        for i, item in enumerate(items, 1):
            story.append(Paragraph(item, step_style, bulletText=f"{i}."))

    def figure(path, caption_text, max_h=FIG_MAX_H_CM * cm, width_ratio=1.0):
        """As large as the page allows; very tall screenshots continue over several parts."""
        if not path or not Path(path).exists():
            return
        fig_no[0] += 1
        parts = layout_figure(Path(path), PAGE_W * width_ratio, max_h, inch)
        for i, (jpg, w, h) in enumerate(parts, 1):
            text = caption_text if len(parts) == 1 else f"{caption_text} (part {i} of {len(parts)})"
            # parts of one screenshot keep one size; a lone figure may shrink a little to finish a page
            story.append(FitFigure(Image(str(jpg), width=w, height=h),
                                   Paragraph(f"Figure {fig_no[0]}. {text}", caption),
                                   min_scale=1.0 if len(parts) > 1 else 0.8))

    def side_by_side(paths, caption_text, max_h=20 * cm):
        """Several images in one row (used for narrow phone screenshots)."""
        paths = [Path(p) for p in paths if p and Path(p).exists()]
        if not paths:
            return
        fig_no[0] += 1
        gap = 0.4 * cm
        col_w = (PAGE_W - gap * (len(paths) - 1)) / len(paths)
        imgs = []
        for p in paths:
            img = Image(str(to_jpeg(p)))
            ratio = min(col_w / img.drawWidth, max_h / img.drawHeight)
            img.drawWidth *= ratio
            img.drawHeight *= ratio
            imgs.append(img)
        t = Table([imgs], colWidths=[col_w] * len(paths))
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), gap / 2), ("RIGHTPADDING", (0, 0), (-1, -1), gap / 2),
        ]))
        story.append(FitFigure(t, Paragraph(f"Figure {fig_no[0]}. {caption_text}", caption), min_scale=0.8))

    def legend(rows, col_widths=(1.2 * cm, 4.3 * cm, None)):
        """Numbered callout legend: rows = [(n, name, description)]."""
        widths = list(col_widths)
        if widths[2] is None:
            widths[2] = PAGE_W - widths[0] - widths[1]
        data = [[Paragraph("No.", cell_b), Paragraph("Element", cell_b), Paragraph("What it is / what it does", cell_b)]]
        for n, name, desc in rows:
            data.append([Paragraph(f"<b>{n}</b>", cell), Paragraph(name, cell_b), Paragraph(desc, cell)])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6cd")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    def table(header, rows, widths):
        data = [[Paragraph(h, cell_b) for h in header]]
        for r in rows:
            data.append([Paragraph(str(c), cell) for c in r])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6cd")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    def note(text, kind="tip"):
        color = {"tip": "#198754", "warn": "#c77700", "info": "#0d6efd"}[kind]
        label = {"tip": "TIP", "warn": "IMPORTANT", "info": "NOTE"}[kind]
        t = Table([[Paragraph(f"<b>{label}</b>", ParagraphStyle("nl", parent=cell, fontSize=8.5, textColor=colors.HexColor(color))),
                    Paragraph(text, cell)]], colWidths=[2.7 * cm, PAGE_W - 2.7 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f9f7")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(color)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    def badge_table():
        rows = [
            ("#198754", "Healthy", "The leaf shows no sign of disease. The result includes upkeep advice for a healthy crop."),
            ("#dc3545", "Unhealthy", "A disease was detected. The disease name is shown under the verdict, with a full treatment guide."),
            ("#ffc107", "Not a Rice Leaf", "The photo does not look like a rice leaf, so no diagnosis was made. Retake the photo."),
            ("#6c757d", "Unknown", "The AI could not make a confident decision, or processing failed. Try a clearer photo."),
        ]
        data = [[Paragraph("Colour", cell_b), Paragraph("Verdict", cell_b), Paragraph("Meaning", cell_b)]]
        for color, name, desc in rows:
            data.append(["", Paragraph(f"<b>{name}</b>", cell), Paragraph(desc, cell)])
        t = Table(data, colWidths=[2.0 * cm, 3.6 * cm, PAGE_W - 5.6 * cm])
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), GREEN), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6cd")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        for i, (color, _, _) in enumerate(rows, 1):
            style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(color)))
        t.setStyle(TableStyle(style))
        story.append(t)
        story.append(Spacer(1, 8))

    # ── Cover ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 4.5 * cm))
    P("RiceGuard", title_style)
    P("Rice Leaf Disease Detection System", subtitle)
    story.append(Spacer(1, 0.4 * cm))
    P("USER MANUAL", ParagraphStyle("um", parent=subtitle, fontSize=22, leading=28, textColor=GREEN))
    story.append(Spacer(1, 1.2 * cm))
    if shots.get("home"):
        cover_img = Image(str(to_jpeg(Path(shots["home"]))))
        ratio = min((PAGE_W * 0.85) / cover_img.drawWidth, (9.5 * cm) / cover_img.drawHeight)
        cover_img.drawWidth *= ratio
        cover_img.drawHeight *= ratio
        cover_img.hAlign = "CENTER"
        story.append(cover_img)
    story.append(Spacer(1, 1.0 * cm))
    P(f"Version 1.0 &nbsp;·&nbsp; {datetime.now().strftime('%B %d, %Y')}", meta)
    P("For farmers, agricultural technicians and system administrators", meta)
    story.append(PageBreak())

    # ── TOC ──────────────────────────────────────────────────────────────
    P("Contents", h1)
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC0", parent=body, fontName=FONT_B, fontSize=11.5, leading=16, spaceBefore=6),
        ParagraphStyle("TOC1", parent=body, fontSize=10, leading=14, leftIndent=18),
    ]
    story.append(toc)
    story.append(Spacer(1, 12))
    P("<b>How to read this manual.</b> Screenshots carry red numbered markers. Each number matches a row "
      "in the legend table right below the picture, so you can find every button and panel on your own screen.", body)
    story.append(PageBreak())

    # ── 1. Introduction ──────────────────────────────────────────────────
    P("1. Introduction", h1)
    P("RiceGuard is a web application that helps you find out whether a rice leaf is healthy or diseased. "
      "You take or upload a photo of one rice leaf, and the system's AI pipeline checks the image, decides if the "
      "leaf is healthy, names the disease when there is one, and gives a treatment guide aligned with PhilRice "
      "recommendations. The guide is available in English, Tagalog and Ilocano.", body)
    P("1.1 Who this manual is for", h2)
    bullets([
        "<b>Farmers and field workers</b> who want a quick disease check from a phone photo (Sections 2–7).",
        "<b>Agricultural technicians</b> who review scan history and disease maps for an area (Sections 5 and 6).",
        "<b>Administrators</b> who manage the site text and scan records (Section 8).",
        "<b>Developers</b> who want to connect other software through the REST API (Section 9).",
    ])
    P("1.2 What the system can detect", h2)
    P("The AI is trained on rice leaf images and recognises the following conditions:", body)
    table(["Condition", "Type", "What you see on the leaf"], [
        [f"<b>{d['name']}</b><br/><i>{d['pathogen']}</i>", d["type"].title(), "<br/>".join(f"• {sx}" for sx in d["symptoms"])]
        for d in diseases
    ] + [["<b>Healthy</b>", "—", "No lesions; uniform green blade. The result gives crop-care reminders."]],
        [4.6 * cm, 2.2 * cm, PAGE_W - 6.8 * cm])
    P("1.3 What you need", h2)
    bullets([
        "A phone, tablet or computer with a modern web browser (Chrome, Edge, Safari or Firefox).",
        "An internet connection to reach the RiceGuard server (ask your administrator for the address, e.g. <b>http://&lt;server&gt;:8000/</b>).",
        "A clear photo of a single rice leaf, saved as JPG, PNG or WEBP and no larger than 10 MB.",
        "Optional: allow the browser to use your <b>camera</b> (to take the photo in the app) and <b>location</b> (to map where the leaf was found).",
    ])
    P("1.4 How a scan works", h2)
    figure(workflow, "The five steps of a scan, from opening the app to reading the recommendation.")
    note("RiceGuard gives an AI-assisted suggestion. Always confirm a diagnosis and any pesticide use with your "
         "Municipal Agriculture Office (MAO) or a PhilRice technologist before treating the field.", "warn")
    story.append(PageBreak())

    # ── 2. Getting started ───────────────────────────────────────────────
    P("2. Getting Started", h1)
    P("2.1 The home page", h2)
    P("Open the RiceGuard address in your browser. The home page is where every scan begins.", body)
    figure(shots.get("home"), "The home page with its main areas marked.")
    legend([
        (1, "Brand / Home link", "Click the leaf logo or site name at any time to return to the home page."),
        (2, "Navigation menu", "Home, History, Locations, API and Admin. See Figure below for details."),
        (3, "Headline banner", "The title and tagline. Administrators can change this text (Section 8.3)."),
        (4, "Drop zone", "Drag a photo file from your computer and drop it here, or click it to browse for a file."),
        (5, "Choose image", "Standard file picker. Accepts JPG, PNG and WEBP up to 10 MB."),
        (6, "Use my location", "Asks the browser for your GPS position so the scan can record where the leaf was photographed."),
        (7, "Use Camera", "Opens the device camera inside the page so you can photograph the leaf directly."),
        (8, "Recent Scans", "The five latest scans with their verdict badge, time and place. Click one to reopen its result."),
        (9, "How it works", "A three-step reminder: upload a leaf photo, the AI analyses it, you get a recommendation."),
    ])
    P("2.2 The navigation bar", h2)
    figure(shots.get("navbar"), "The green navigation bar appears on every page.")
    legend([
        (1, "RiceGuard", "Site name. Click to go to the home page."),
        (2, "Home", "The upload page (Section 3)."),
        (3, "History", "All past scans with counters and filters (Section 5)."),
        (4, "Locations", "Map, charts and ranked places where leaves were scanned (Section 6)."),
        (5, "API", "Opens the machine-readable scan list in a new tab (Section 9)."),
        (6, "Admin", "Opens the administrator login in a new tab (Section 8)."),
        (7, "Language menu", "Switch the interface between English, Tagalog and Ilocano (Section 2.3)."),
    ])
    P("2.3 Choosing your language", h2)
    P("Click the language button at the right end of the navigation bar. Pick <b>English</b>, <b>Tagalog</b> or "
      "<b>Ilocano</b>. The choice is remembered on your device, and it changes the page text, the result verdicts and "
      "the treatment guides.", body)
    figure(shots.get("lang_menu"), "The language menu. (1) opens the menu; (2), (3) and (4) select a language.")
    figure(shots.get("home_fil"), "Home page in Tagalog.")
    figure(shots.get("home_ilo"), "Home page in Ilocano. Ilocano is the default language when no choice has been made yet.")
    story.append(PageBreak())

    # ── 3. Scanning a leaf ───────────────────────────────────────────────
    P("3. Scanning a Rice Leaf", h1)
    P("3.1 Taking a good photo", h2)
    P("The quality of the photo decides the quality of the result. Follow these guidelines before you upload.", body)
    figure(photo_tips, "Photo guidelines: what makes a good leaf photo and what to avoid.")
    P("3.2 Step by step", h2)
    steps([
        "Open the <b>Home</b> page.",
        "Add the photo in one of three ways: click <b>Choose image</b> and pick a file; <b>drag and drop</b> the file onto the drop zone; or tap <b>Use Camera</b>, centre the leaf inside the frame guide and press <b>Capture</b>.",
        "Wait for the message <i>Checking image…</i> to finish. This is the <b>Layer 0 leaf check</b>: the system confirms the photo looks like a rice leaf before analysis.",
        "Optional: tap <b>Use my location</b> and allow the permission prompt so the place name and map are saved with the scan.",
        "Review the preview. If it is the wrong photo, press <b>Remove photo</b> or <b>Choose a different image</b>.",
        "Press the green <b>Analyse Leaf</b> button. Analysis takes a few seconds; do not close the page.",
        "Read the result page (Section 4).",
    ])
    figure(shots.get("checking"), "Right after you choose a photo, the leaf check runs (1). The Analyse button appears only when it finishes.")
    figure(shots.get("preview"), "The preview panel once the leaf check has passed.")
    legend([
        (1, "Selected image", "File name of the photo you chose."),
        (2, "Remove photo", "Clears the photo so you can start again."),
        (3, "Preview", "The exact image that will be analysed."),
        (4, "Location card", "Place name, coordinates and a small map when location was captured (from device GPS or photo GPS data)."),
        (5, "Use my location", "Retry the GPS lookup if the location is missing or wrong."),
        (6, "Choose a different image", "Go back to the file picker without losing your place."),
        (7, "Analyse Leaf", "Starts the AI analysis. Only visible after the leaf check passes."),
    ])
    figure(shots.get("analysing"), "While the AI works, an 'Analysing leaf…' notice (1) is shown. Wait for the result page.")
    note("If you leave the home page before pressing Analyse, your selected photo is kept as a draft and restored when "
         "you come back. Use the <b>New Scan</b> button on other pages to discard it.", "tip")
    P("3.3 When the leaf check is not sure", h2)
    P("If the photo does not clearly show a rice leaf, a yellow warning appears under the preview. You may still press "
      "Analyse Leaf, but the result will be less reliable. A better option is to retake the photo closer, in good light, "
      "with one leaf filling the frame.", body)
    figure(shots.get("layer0_warning"), "Leaf check warning (1). The Analyse button (3) is still available; (4) lets you choose a better photo.")
    P("3.4 Unsupported files", h2)
    P("Files that are not images, or that are larger than 10 MB, are refused immediately with a red notice. "
      "Close the notice with the × button and pick a JPG, PNG or WEBP photo.", body)
    figure(shots.get("bad_file"), "The upload error notice (1) with its close button (2).")
    story.append(PageBreak())

    # ── 4. Result page ───────────────────────────────────────────────────
    P("4. Understanding the Result Page", h1)
    P("After analysis you are taken to the result page. Every scan has its own page and its own number, so you can "
      "return to it later from History or Recent Scans.", body)
    P("4.1 Page overview", h2)
    figure(shots.get("result_diseased") or shots.get("result_fresh"), "A complete result page for a diseased leaf.")
    legend([
        (1, "Toolbar", "New Scan, History and Print buttons plus the scan number and date."),
        (2, "Verdict banner", "The main answer: Healthy, Unhealthy (with the disease name), Not a Rice Leaf, or Unknown."),
        (3, "Your leaf photo", "The analysed photo and the place it was taken. Tap the photo to enlarge it."),
        (4, "At a glance", "Result, disease, scan time and location in one compact card."),
        (5, "Recommendation", "What this means and what to do next: immediate actions and the complete treatment procedure."),
        (6, "More guidance", "Expandable sections: symptoms, cultural management, pesticide options, spray guidelines, prevention, resistant varieties."),
        (7, "Print result", "Prints or saves the page as a clean report."),
    ])
    P("4.2 The verdict colours", h2)
    badge_table()
    figure(shots.get("result_hero"), "Verdict banner: (1) status icon, (2) verdict, (3) disease name.")
    P("4.3 Toolbar", h2)
    figure(shots.get("result_toolbar"), "Toolbar: (1) New Scan, (2) History, (3) Print, (4) scan number and date.")
    P("4.4 Your leaf photo and At a glance", h2)
    figure(shots.get("result_photo"), "Your leaf photo (1) with the capture place and an 'Open map' link (2).")
    figure(shots.get("result_glance"), "At a glance: (1) Result, (2) Disease, (3) Scan time, (4) Location.")
    figure(shots.get("photo_modal"), "Tapping the photo opens a full-screen view (1). Press Escape or tap outside to close.")
    P("4.5 Recommendation and treatment guide", h2)
    P("The recommendation card explains what the verdict means and lists the immediate actions and the complete "
      "treatment procedure for the detected disease. Below it, <b>More guidance</b> holds detailed sections you can "
      "open one at a time.", body)
    figure(shots.get("result_treatment"), "The Recommendation card with immediate actions and the complete procedure.")
    figure(shots.get("result_pesticides"), "More guidance with the Pesticide options section open.")
    legend([
        (1, "Symptoms to confirm", "Check these signs on the plant before treating."),
        (2, "Cultural management", "Non-chemical practices: water, fertiliser, sanitation, spacing."),
        (3, "Pesticide options", "FPA-registered products. Each numbered box is a different product; choose one available locally."),
        (4, "Product details", "Active ingredient, timing, rate, wet-field and dry-field instructions, and mixing / spray steps."),
        (5, "Prevention", "How to avoid the disease next season."),
        (6, "Resistant varieties", "Varieties to ask about at PhilRice or your MAO."),
    ])
    note("Rates in the guide are typical references. Always follow the label of the product you buy, respect the "
         "pre-harvest interval (PHI), and wear protective clothing when spraying.", "warn")
    P("4.6 Printing a result", h2)
    P("Press <b>Print</b> in the toolbar (or at the bottom of the page). The browser's print dialog opens with a clean "
      "report layout that includes the site name, scan number, date and the recommendation. Choose a printer or "
      "'Save as PDF'.", body)
    figure(shots.get("print_view"), "The print layout with its report header (1).")
    P("4.7 Other result types", h2)
    figure(shots.get("result_healthy"), "A healthy leaf: green verdict (1), care reminders (2) and the healthy-crop guide (3).")
    figure(shots.get("result_not_rice"), "Not a rice leaf: yellow verdict (1), the original photo (2), and advice to retake the photo (3). No disease analysis is made.")
    figure(shots.get("result_warned"), "A leaf-check warning (1) means the photo was analysed but the system was not fully confident it was a rice leaf. Treat the result with care.")
    story.append(PageBreak())

    # ── 5. History ───────────────────────────────────────────────────────
    P("5. Scan History", h1)
    P("Click <b>History</b> in the navigation bar to see every scan stored on the server, newest first.", body)
    figure(shots.get("history"), "The Scan History page.")
    legend([
        (1, "New Scan", "Go back to the home page and start a fresh scan."),
        (2, "Counters", "Total scans, Healthy, Unhealthy, Unknown, and scans with a saved location. Click a counter to filter."),
        (3, "Filter bar", "Quick filters: All, Healthy, Unhealthy, Unknown (the number in brackets is the count)."),
        (4, "Scan card", "Photo, verdict badge, disease, date and place. Click the card to open the result page."),
        (5, "Pagination", "Ten scans per page. Use Previous / Next to move between pages."),
    ])
    figure(shots.get("history_filter"), "Filter bar: (1) All, (2) Healthy, (3) Unhealthy, (4) Unknown.")
    figure(shots.get("history_card"), "A scan card: (1) the photo, (2) the verdict badge.")
    figure(shots.get("history_diseased"), "History filtered to Unhealthy scans. The active filter button (1) is highlighted.")
    story.append(PageBreak())

    # ── 6. Locations ─────────────────────────────────────────────────────
    P("6. Captured Locations", h1)
    P("When a scan was made with location on, RiceGuard records where the leaf was photographed. The <b>Locations</b> "
      "page turns those records into a field map and simple charts, which helps technicians spot where diseases "
      "are appearing.", body)
    figure(shots.get("locations"), "The Locations page.")
    legend([
        (1, "Counters", "Total scans, scans with location, scans without, unique places, healthy and unhealthy counts."),
        (2, "Location map", "Interactive map. Circles are capture zones; dots are exact capture points, coloured by verdict. Click a marker for details."),
        (3, "Barangays with most scans", "Bar chart of scan volume per barangay, with healthy vs diseased counts and the top disease."),
        (4, "Scan & disease timeline", "Line chart of scans and diseased detections over time (by hour, day or week depending on the span)."),
        (5, "Most captured places", "Ranked list of places. Click a place to see only its scans."),
        (6, "Common diseases", "Which diseases appear most often among located scans."),
        (7, "How location was captured", "From device GPS versus from the photo's own GPS data (EXIF)."),
    ])
    figure(shots.get("locations_map"), "The location map (1). Zoom with the + / − buttons or the mouse wheel; drag to pan.")
    figure(shots.get("locations_charts"), "Barangays with most scans.")
    figure(shots.get("locations_timeline"), "Scan and disease timeline.")
    figure(shots.get("locations_rank"), "Most captured places. Click a place name (1) to open its detail view.")
    figure(shots.get("locations_place"), "A single place selected: the map, timeline and scan cards are limited to that place. Use 'All places' to go back.")
    note("Location is only recorded when you allow the browser's location prompt, or when the photo already contains "
         "GPS data. Scans without location still appear in History.", "info")
    story.append(PageBreak())

    # ── 7. Mobile ────────────────────────────────────────────────────────
    P("7. Using RiceGuard on a Phone", h1)
    P("The site adapts to small screens. On a phone the menu collapses behind a three-line menu button, the camera opens in full "
      "screen, and the location hint is shown under the form.", body)
    side_by_side([shots.get("mobile_home"), shots.get("mobile_menu")],
                 "Home page on a phone (left) and the expanded menu after tapping the three-line button (right).")
    legend([
        (1, "Menu button (three lines)", "Opens the navigation links and the language menu."),
        (2, "Drop zone", "Tap to browse for a photo from the gallery."),
        (3, "Choose image", "On most phones this offers 'Camera' or 'Files'."),
        (4, "Use my location", "Allow the location prompt so the scan is mapped."),
        (5, "Use Camera", "Opens the rear camera inside the page; centre the leaf in the guide box and tap Capture."),
    ])
    note("For the camera and location to work, the site must be opened over HTTPS or on the same computer "
         "(localhost). Browsers block camera and GPS on plain HTTP addresses from other devices.", "warn")
    if shots.get("mobile_result") and Path(shots["mobile_result"]).exists():
        side_by_side(split_tall(Path(shots["mobile_result"]), max_ratio=2.5),
                     "A result page on a phone: the panels stack vertically (read left to right).")
    story.append(PageBreak())

    # ── 8. Admin ─────────────────────────────────────────────────────────
    P("8. Administrator Guide", h1)
    P("The administration panel is for staff accounts only. It lets you change the home page text and review, edit "
      "or delete scan records.", body)
    P("8.1 Logging in", h2)
    steps([
        "Click <b>Admin</b> in the navigation bar, or open <b>/admin/</b> on the server address.",
        "Enter your staff <b>username</b> (1) and <b>password</b> (2).",
        "Press <b>Log in</b> (3).",
    ])
    figure(shots.get("admin_login"), "The administrator login page.")
    note("Accounts are created by the system owner with the command <b>python manage.py createsuperuser</b>. "
         "Never share your password. Use <b>Log out</b> (top right) when you finish.", "info")
    P("8.2 The dashboard", h2)
    figure(shots.get("admin_dashboard"), "The admin dashboard.")
    legend([
        (1, "Detection app", "The two things you can manage: Leaf Scans and Site settings."),
        (2, "Leaf Scans", "Open the list of all scans. 'Add' creates a record manually (rarely needed)."),
        (3, "Site settings", "Change the headline, tagline and site name shown on the home page."),
        (4, "User tools", "Your username, 'View site' (opens the public pages), change password, and Log out."),
        (5, "Recent actions", "Your latest edits in the admin, for quick access."),
    ])
    P("8.3 Site settings (home page text)", h2)
    steps([
        "On the dashboard click <b>Site settings</b>.",
        "Edit the <b>Hero title</b> (1), <b>Hero tagline</b> (2) and <b>Site brand</b> (3).",
        "Press <b>Save</b> (4). The public home page shows the new text immediately (English interface only; Tagalog and Ilocano use their built-in translations).",
    ])
    figure(shots.get("admin_settings"), "Editing the site settings.")
    P("8.4 Managing leaf scans", h2)
    figure(shots.get("admin_scans"), "The Leaf Scans list.")
    legend([
        (1, "Search", "Search by disease name, recommendation text or place name."),
        (2, "Filters", "Filter by result (Healthy / Unhealthy / Unknown) or by upload date."),
        (3, "Scan list", "One row per scan: ID, thumbnail, result, disease, confidence, place and date. Click the ID to open it."),
        (4, "Actions", "Tick rows, choose 'Delete selected Leaf Scans', then press Go to delete several scans at once."),
        (5, "Paging", "Move between pages when the list is long."),
    ])
    P("8.5 Editing or deleting one scan", h2)
    figure(shots.get("admin_scan_detail"), "A single scan in the admin.")
    legend([
        (1, "Preview", "The uploaded photo (read-only)."),
        (2, "Result health", "Healthy / Unhealthy / Unknown. Correct it here if a technician confirms a different verdict."),
        (3, "Disease name", "Free text, e.g. 'brown spot', 'blast', 'bacterial blight'."),
        (4, "Recommendation", "The stored recommendation text shown on the public result page."),
        (5, "Delete", "Removes the scan and its photo permanently. A confirmation page appears first."),
        (6, "Save", "Save changes. 'Save and continue editing' keeps the page open."),
    ])
    note("Deleting a scan cannot be undone. Export or note down anything you need before deleting.", "warn")
    story.append(PageBreak())

    # ── 9. API ───────────────────────────────────────────────────────────
    P("9. REST API (for developers)", h1)
    P("Other software can use RiceGuard through a small JSON API. The <b>API</b> link in the navigation bar opens a "
      "browsable version of the scan list where you can try requests from the browser.", body)
    figure(shots.get("api"), "The browsable API page for the scan list.")
    table(["Method and path", "Purpose"], [
        ["<b>GET</b> /api/scans/", "List all scans as JSON. Add <b>?health=healthy</b>, <b>diseased</b> or <b>unknown</b> to filter."],
        ["<b>GET</b> /api/scans/&lt;id&gt;/", "One scan: verdict, disease, confidence, probabilities, recommendation, location and image URL."],
        ["<b>POST</b> /api/scans/ (or /api/scan/)", "Upload a photo as multipart form field <b>image</b>; optional <b>latitude</b> / <b>longitude</b>. Returns the analysed scan."],
        ["<b>POST</b> /api/validate-layer0/", "Run only the rice-leaf check on an image and get a warning if it does not look like a rice leaf."],
        ["<b>POST</b> /api/locate/", "Turn GPS coordinates (or a photo's EXIF data) into a place name."],
        ["<b>GET</b> /lang/&lt;en|fil|ilo&gt;/", "Set the interface language cookie."],
    ], [6.2 * cm, PAGE_W - 6.2 * cm])
    P("Example with curl:", body)
    story.append(Paragraph(
        'curl -X POST http://&lt;server&gt;:8000/api/scans/ -F "image=@leaf.jpg" -F "latitude=17.11" -F "longitude=120.43"',
        ParagraphStyle("code", parent=small, fontName="Courier", backColor=colors.HexColor("#f1f3f5"), borderPadding=6, leftIndent=6)))
    story.append(Spacer(1, 12))

    # ── 10. Troubleshooting ──────────────────────────────────────────────
    P("10. Troubleshooting and FAQ", h1)
    table(["What you see", "Why it happens", "What to do"], [
        ["<b>Unsupported file format</b>", "The file is not JPG, PNG or WEBP (for example HEIC, PDF or a document).", "Convert or re-save the photo as JPG or PNG, then upload again."],
        ["<b>File is too large</b>", "The image is over 10 MB.", "Use a smaller photo or lower the camera resolution."],
        ["<b>This photo may not show a clear rice leaf</b>", "The Layer 0 leaf check is not confident.", "Retake: one leaf, close up, plain background, good light. You may still analyse, but expect a less reliable result."],
        ["<b>Not a Rice Leaf</b> verdict", "The photo shows another plant, a whole field, or an unrelated object.", "Photograph a single rice leaf blade and scan again."],
        ["<b>Unknown</b> verdict", "The AI could not decide confidently, or the top two diseases were too close.", "Take a sharper, closer photo of the affected area."],
        ["<b>Location: Not captured</b>", "Location permission was refused, GPS is off, or the page is not on HTTPS.", "Turn on GPS, allow the browser prompt, and press 'Use my location' again."],
        ["<b>Camera not available</b>", "The browser has no camera permission, or the page is on plain HTTP.", "Allow the camera in the browser settings, or choose a photo from the gallery instead."],
        ["<b>Analysis could not finish</b>", "The connection dropped while the server was working.", "Check the network and press Analyse Leaf again."],
        ["<b>Scan not found</b>", "The scan was deleted by an administrator or the link is wrong.", "Open History and pick the scan from the list."],
        ["Page shows Ilocano text", "No language was chosen yet; Ilocano is the default.", "Use the language menu to switch to English or Tagalog."],
    ], [4.3 * cm, 5.6 * cm, PAGE_W - 9.9 * cm])
    figure(shots.get("not_found"), "The 'Scan not found' page offers links to scan a new leaf or open History.")

    P("10.1 Glossary", h2)
    table(["Term", "Meaning"], [
        ["Layer 0 / leaf check", "The first stage that verifies the photo looks like a rice leaf before any diagnosis."],
        ["Confidence", "How sure the AI is about its answer, from 0% to 100%. Higher is better."],
        ["Verdict", "The main result: Healthy, Unhealthy, Not a Rice Leaf, or Unknown."],
        ["FPA", "Fertilizer and Pesticide Authority — only FPA-registered products are listed in the guide."],
        ["PHI", "Pre-harvest interval: the days you must wait between the last spray and harvest."],
        ["MAO", "Municipal Agriculture Office — your local source for confirmation and advice."],
        ["PhilRice", "Philippine Rice Research Institute, whose recommendations the treatment guides follow."],
        ["EXIF GPS", "Location information some cameras store inside the photo file."],
    ], [4.3 * cm, PAGE_W - 4.3 * cm])

    doc.multiBuild(story, onFirstPage=on_page, onLaterPages=on_page)
    return OUTPUT_PDF


# ═══════════════════════════════════════════════════════════════════════════

SHOTS_INDEX = SHOT_DIR / "_shots.json"


def main() -> None:
    pdf_only = "--pdf-only" in sys.argv
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    django_setup()
    global BRAND, OUTPUT_PDF
    from detection.models import SiteSettings
    BRAND = (SiteSettings.load().site_brand or "RiceGuard").strip()
    safe = "".join(ch for ch in BRAND if ch.isalnum() or ch in "-_") or "RiceGuard"
    OUTPUT_PDF = DOCS_DIR / f"{safe}_User_Manual.pdf"
    samples = pick_sample_scans()
    print("Sample scans:", samples)
    diseases = disease_reference()

    global RUN_STARTED
    from django.utils import timezone
    RUN_STARTED = timezone.now()
    if pdf_only and SHOTS_INDEX.exists():
        shots = {k: (Path(v) if v else None) for k, v in json.loads(SHOTS_INDEX.read_text()).items()}
        print("Reusing saved screenshots (--pdf-only)")
    else:
        print("Starting server…")
        proc = start_server()
        try:
            print("Capturing screenshots…")
            shots = capture_all(samples)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            cleanup_created_scans()
        SHOTS_INDEX.write_text(json.dumps({k: (str(v) if v else None) for k, v in shots.items()}, indent=1))

    print("Drawing diagrams…")
    workflow = create_workflow_diagram()
    tips = create_photo_tips_diagram()

    print("Building PDF…")
    out = build_pdf(shots, workflow, tips, diseases)
    for junk in list(SHOT_DIR.glob("*_raw.png")) + list(SHOT_DIR.glob("*_crop.png")):
        junk.unlink(missing_ok=True)
    print(f"Done: {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
