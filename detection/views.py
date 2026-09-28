import math
import mimetypes
import os
from collections import defaultdict

from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_http_methods
from django.urls import reverse
from django.http import JsonResponse, FileResponse, Http404
from django.contrib import messages
from django.utils import timezone
from datetime import timedelta
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import LeafScan
from .forms import LeafUploadForm, validate_leaf_upload
from .serializers import LeafScanSerializer
from .geo import resolve_request_location, resolve_scan_location
from .i18n import (
    DIALECT_COOKIE,
    LANG_COOKIE,
    format_gate_warning,
    format_timeline_label,
    get_lang,
    get_strings,
    health_label,
    is_filipino_lang,
    localize_processing_error,
    localize_recommendation_text,
    normalize_lang,
)
from .training_samples import (
    get_most_similar_training_sample,
    get_training_samples,
    resolve_training_image,
)
from .recommendation_display import parse_recommendation
from .disease_guide import (
    get_application_guide,
    get_farmer_guide,
    localize_farmer_guide,
    PHILRICE_SOURCE,
    PHILRICE_SOURCE_URL,
)


def _layer0_recommendation_key(reason: str) -> str:
    if reason in ("not_rice", "wrong_shape"):
        return "not_rice"
    return "low_confidence"


def _layer0_check(upload) -> tuple[bool, str]:
    """Run Layer 0 heuristics on an upload or path. Returns (accepted, reason)."""
    from ml.layer0_validator import is_rice_leaf

    if isinstance(upload, str):
        return is_rice_leaf(upload)
    from PIL import Image

    with Image.open(upload) as img:
        return is_rice_leaf(img)


def _layer0_rejection_result(reason: str) -> dict:
    from ml.layer3_recommendation import get_recommendation

    key = _layer0_recommendation_key(reason)
    return {
        "health": "unknown",
        "disease_name": "",
        "confidence": 0.0,
        "recommendation": get_recommendation(key),
        "rejection_reason": reason,
    }


def _how_it_works(lang: str):
    t = get_strings(lang)
    return [
        (1, "image", t["howit_1_title"], t["howit_1_desc"]),
        (2, "cpu-fill", t["howit_2_title"], t["howit_2_desc"]),
        (3, "journal-medical", t["howit_3_title"], t["howit_3_desc"]),
    ]


def set_language(request, lang):
    lang = normalize_lang(lang)
    next_url = request.GET.get("next") or request.META.get("HTTP_REFERER") or "/"
    from django.utils.http import url_has_allowed_host_and_scheme

    if not url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
        next_url = "/"

    response = redirect(next_url)
    max_age = 365 * 24 * 60 * 60
    response.set_cookie(LANG_COOKIE, lang, max_age=max_age, samesite="Lax")
    if is_filipino_lang(lang):
        response.set_cookie(DIALECT_COOKIE, lang, max_age=max_age, samesite="Lax")
    return response


@never_cache
def index(request):
    from ml.training_report import METRIC_ROWS, get_validation_summary

    lang = get_lang(request)
    form = LeafUploadForm(lang=get_lang(request))
    recent_scans = LeafScan.objects.all()[:5]
    validation = get_validation_summary()
    validation_table: list[dict] = []
    if validation:
        for key, label in METRIC_ROWS:
            h_val = validation["health"].get(key)
            d_val = validation["disease"].get(key)
            validation_table.append({
                "label": label,
                "health": _format_metric(key, h_val),
                "disease": _format_metric(key, d_val),
            })
    return render(request, "detection/index.html", {
        "form": form,
        "recent_scans": recent_scans,
        "howit": _how_it_works(lang),
        "validation": validation,
        "validation_table": validation_table,
    })


def _format_metric(key: str, value) -> str:
    if value is None:
        return "—"
    if key == "accuracy":
        return f"{float(value):.2%}"
    return f"{float(value):.4f}"


@require_http_methods(["POST"])
def upload_leaf(request):
    form = LeafUploadForm(request.POST, request.FILES, lang=get_lang(request))
    if not form.is_valid():
        t = get_strings(get_lang(request))
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"error": t["invalid_form"], "details": form.errors}, status=400)
        messages.error(request, t["please_upload_valid"])
        return redirect("index")

    scan = LeafScan(image=form.cleaned_data["image"])
    scan.save()
    resolve_scan_location(scan, request=request, lang=get_lang(request))

    try:
        # Lenient Layer 0: the gate check lives inside run_pipeline now. A
        # flagged image is still analysed (with a warning); it is only rejected
        # outright when the gate says not_rice AND Layer 1 is unsure too.
        from ml.pipeline import run_pipeline

        result = run_pipeline(scan.image.path)

        scan.result_health = result.get("health", "unknown")
        scan.disease_name = result.get("disease_name", "")
        scan.confidence = result.get("confidence", 0.0)
        scan.all_disease_probs = result.get("all_disease_probs", {})
        scan.recommendation = result.get("recommendation", "")
        scan.gate_warning = result.get("gate_warning", "")
        scan.rejection_reason = result.get("rejection_reason", "")
    except Exception as exc:
        scan.result_health = "unknown"
        scan.processing_error = str(exc)

    scan.save()

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        serializer = LeafScanSerializer(scan, context={"request": request})
        return JsonResponse({
            "redirect_url": reverse("scan_detail", kwargs={"pk": scan.pk}),
            "scan": serializer.data,
        })

    return redirect("scan_detail", pk=scan.pk)


def _upload_requirements_response(upload, lang="en"):
    ok, code, message = validate_leaf_upload(upload, lang=lang)
    if ok:
        return None
    return JsonResponse(
        {"accepted": False, "code": code, "error": message, "message": message},
        status=400,
    )


@require_http_methods(["POST"])
def validate_layer0(request):
    """
    Pre-upload Layer 0 check — now advisory rather than blocking.

    A gate rejection no longer stops the upload: the response is accepted=True
    with a "warning" so the UI can flag it while still allowing analysis. The
    full pipeline makes the final call (reject only when the gate says not_rice
    AND Layer 1 confidence < 40%). Only malformed/oversized files are refused.
    """
    if "image" not in request.FILES:
        return JsonResponse({"error": get_strings(get_lang(request))["no_image_provided"]}, status=400)
    upload = request.FILES["image"]
    requirements_error = _upload_requirements_response(upload, lang=get_lang(request))
    if requirements_error:
        return requirements_error
    try:
        if hasattr(upload, "seek"):
            upload.seek(0)
        accepted, reason = _layer0_check(upload)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    lang = get_lang(request)
    location = resolve_request_location(request, upload=upload, lang=lang)
    payload = {"accepted": True}
    if not accepted:
        payload["reason"] = reason
        payload["warning"] = get_strings(lang)["layer0_warning"]
    if location:
        payload["location"] = location
    return JsonResponse(payload)


@require_http_methods(["POST"])
def locate_capture(request):
    """Resolve a place name from device GPS and/or photo EXIF."""
    upload = request.FILES.get("image")
    location = resolve_request_location(request, upload=upload, lang=get_lang(request))
    if not location:
        return JsonResponse({"found": False})
    return JsonResponse({"found": True, "location": location})


@never_cache
def scan_detail(request, pk):
    try:
        scan = LeafScan.objects.get(pk=pk)
    except LeafScan.DoesNotExist:
        t = get_strings(get_lang(request))
        return render(
            request,
            "detection/scan_not_found.html",
            {"scan_id": pk, "not_found_msg": t["scan_not_found_msg"].format(id=pk)},
            status=404,
        )
    # A photo the rice gate flagged as not_rice is presented as "Not a Rice
    # Leaf" — no disease analysis. Legacy scans from the lenient-gate era only
    # stored the flag inside gate_warning; LeafScan.is_not_rice covers both.
    is_not_rice = scan.is_not_rice

    training_samples = [] if is_not_rice else get_training_samples(
        scan.result_health, scan.disease_name
    )
    best_training_match = None
    if not is_not_rice and scan.image and os.path.exists(scan.image.path):
        best_training_match = get_most_similar_training_sample(
            scan.image.path,
            scan.result_health,
            scan.disease_name,
        )
    # A conclusive verdict means the image WAS analysed; only re-check the gate
    # for unknowns.
    layer0_passed = scan.result_health in ("healthy", "diseased") and not is_not_rice
    if (
        not layer0_passed
        and not is_not_rice
        and scan.image
        and os.path.exists(scan.image.path)
    ):
        layer0_passed, _ = _layer0_check(scan.image.path)

    lang = get_lang(request)
    t = get_strings(lang)
    if not scan.has_location:
        if resolve_scan_location(scan, lang=lang):
            scan.save(update_fields=["latitude", "longitude", "location_name", "location_source"])
    farmer_guide = None
    if is_not_rice:
        pass  # no treatment guide for a photo that is not a rice leaf
    elif scan.result_health == "diseased" and scan.disease_name:
        farmer_guide = get_farmer_guide(scan.disease_name)
    elif scan.result_health == "healthy":
        farmer_guide = get_farmer_guide("healthy")
    farmer_guide = localize_farmer_guide(farmer_guide, lang)

    if is_not_rice:
        recommendation_text = t["rec_not_rice"]
    else:
        recommendation_text = localize_recommendation_text(
            health=scan.result_health,
            disease_name=scan.disease_name or "",
            stored_recommendation=scan.recommendation or "",
            lang=lang,
        )

    return render(request, "detection/result.html", {
        "scan": scan,
        "is_not_rice": is_not_rice,
        "health_display": (
            t["health_not_rice"] if is_not_rice else health_label(scan.result_health, lang)
        ),
        "training_samples": training_samples,
        "best_training_match": best_training_match,
        "farmer_guide": farmer_guide,
        "recommendation_display": parse_recommendation(recommendation_text),
        "layer0_passed": layer0_passed,
        "philrice_source": PHILRICE_SOURCE,
        "philrice_source_url": PHILRICE_SOURCE_URL,
        "pesticide_disclaimer": t["pesticide_disclaimer"],
        "pesticide_application": get_application_guide(lang),
        "gate_warning_display": (
            "" if is_not_rice else format_gate_warning(scan.gate_warning, lang)
        ),
        "not_rice_note": t["not_rice_note"],
        "processing_error_display": localize_processing_error(scan.processing_error or "", lang),
    })


def training_sample_image(request, disease, filename):
    relpath = request.GET.get("relpath")
    path = resolve_training_image(disease, filename, relpath=relpath)
    if path is None:
        raise Http404("Training sample not found.")
    content_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(open(path, "rb"), content_type=content_type or "application/octet-stream")


@never_cache
def scan_history(request):
    health_filter = request.GET.get("health", "")
    if health_filter not in ("healthy", "diseased", "unknown"):
        health_filter = ""

    stats = LeafScan.objects.aggregate(
        total=Count("id"),
        healthy=Count("id", filter=Q(result_health="healthy")),
        diseased=Count("id", filter=Q(result_health="diseased")),
        located=Count(
            "id", filter=Q(latitude__isnull=False, longitude__isnull=False)
        ),
    )
    # Anything that is not a conclusive verdict (rejected images, pipeline
    # errors, legacy rows) counts as unknown so the cards add up to the total.
    unknown_count = stats["total"] - stats["healthy"] - stats["diseased"]

    scans = LeafScan.objects.all().order_by("-uploaded_at", "-id")
    if health_filter == "unknown":
        scans = scans.exclude(result_health__in=("healthy", "diseased"))
    elif health_filter:
        scans = scans.filter(result_health=health_filter)

    paginator = Paginator(scans, 10)
    page_obj = paginator.get_page(request.GET.get("page"))
    t = get_strings(get_lang(request))
    page_label = t["page_of"].format(
        current=page_obj.number,
        total=page_obj.paginator.num_pages,
    )
    return render(request, "detection/history.html", {
        "scans": page_obj,
        "page_obj": page_obj,
        "page_label": page_label,
        "health_filter": health_filter,
        "total": stats["total"],
        "healthy_count": stats["healthy"],
        "diseased_count": stats["diseased"],
        "unknown_count": unknown_count,
        "located_count": stats["located"],
        "filtered_count": paginator.count,
        "showing_label": t["showing_scans"].format(
            shown=paginator.count, total=stats["total"]
        ),
    })


def _normalize_place_name(name: str) -> str:
    """Collapse common language variants so the same field groups together."""
    text = " ".join((name or "").strip().split())
    if not text:
        return ""
    lowered = text.casefold()
    # Nominatim may return country in English or Filipino depending on language.
    for src, dst in (
        (", pilipinas", ", philippines"),
        (", philippines", ", philippines"),
    ):
        if lowered.endswith(src):
            return (text[: len(text) - len(src)] + dst).casefold()
    return lowered


def _location_key(scan: LeafScan) -> str:
    name = (scan.location_name or "").strip()
    if name:
        return f"name:{_normalize_place_name(name)}"
    return f"coord:{scan.latitude:.3f},{scan.longitude:.3f}"


def _normalize_place_query(raw: str) -> str:
    """Normalize ?place= keys from old or language-variant URLs."""
    key = (raw or "").strip()
    if key.startswith("name:"):
        return f"name:{_normalize_place_name(key[5:])}"
    return key


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(d_lng / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(min(1.0, a)))


def _build_location_groups(scans: list[LeafScan]) -> list[dict]:
    groups: dict[str, dict] = {}
    for scan in scans:
        key = _location_key(scan)
        if key not in groups:
            groups[key] = {
                "key": key,
                "name": (scan.location_name or "").strip() or scan.coordinates_display,
                "count": 0,
                "healthy": 0,
                "diseased": 0,
                "unknown": 0,
                "lat_sum": 0.0,
                "lng_sum": 0.0,
                "diseases": defaultdict(int),
                "latest": scan,
                "points": [],
            }
        group = groups[key]
        group["count"] += 1
        health = scan.result_health if scan.result_health in ("healthy", "diseased") else "unknown"
        group[health] += 1
        lat = float(scan.latitude)
        lng = float(scan.longitude)
        group["lat_sum"] += lat
        group["lng_sum"] += lng
        group["points"].append({
            "id": scan.pk,
            "lat": lat,
            "lng": lng,
            "status": health,
            "disease": scan.disease_name or "",
            "url": f"/result/{scan.pk}/",
        })
        if scan.disease_name:
            group["diseases"][scan.disease_name] += 1

    ranked = []
    for group in groups.values():
        lat = group["lat_sum"] / group["count"]
        lng = group["lng_sum"] / group["count"]
        # Cover every exact capture with a padded zone around the centroid.
        max_dist = max(
            (_haversine_m(lat, lng, point["lat"], point["lng"]) for point in group["points"]),
            default=0.0,
        )
        radius_m = max(45.0, max_dist * 1.35 + 25.0)
        diseases = sorted(group["diseases"].items(), key=lambda item: (-item[1], item[0]))
        if group["diseased"] > 0 and group["diseased"] >= group["healthy"]:
            status = "diseased"
        elif group["healthy"] > 0:
            status = "healthy"
        else:
            status = "unknown"
        ranked.append({
            "key": group["key"],
            "name": group["name"],
            "count": group["count"],
            "healthy": group["healthy"],
            "diseased": group["diseased"],
            "unknown": group["unknown"],
            "lat": lat,
            "lng": lng,
            "radius_m": round(radius_m, 1),
            "points": group["points"],
            "top_disease": diseases[0][0] if diseases else "",
            "diseases": diseases[:3],
            "latest": group["latest"],
            "status": status,
            "status_badge": {
                "healthy": "success",
                "diseased": "danger",
                "unknown": "secondary",
            }.get(status, "secondary"),
            "diseased_pct": round(100 * group["diseased"] / group["count"]) if group["count"] else 0,
            "maps_url": (
                f"https://www.openstreetmap.org/?mlat={lat}&mlon={lng}"
                f"#map=16/{lat}/{lng}"
            ),
        })
    ranked.sort(key=lambda item: (-item["count"], item["name"].casefold()))
    return ranked


def _barangay_label(place_name: str, lang: str = "en") -> str:
    """Use the first comma-separated part as the barangay/locality label."""
    parts = [part.strip() for part in (place_name or "").split(",") if part.strip()]
    if parts:
        return parts[0]
    return place_name or get_strings(lang)["unknown_place"]


def _build_barangay_chart(groups: list[dict], limit: int = 8, lang: str = "en") -> dict:
    """Bar-chart payload: top barangays by scan count with disease breakdown."""
    rows = []
    for group in groups[:limit]:
        diseases = [
            {"name": name, "count": count}
            for name, count in (group.get("diseases") or [])
        ]
        rows.append({
            "label": _barangay_label(group["name"], lang=lang),
            "full_name": group["name"],
            "key": group["key"],
            "scans": group["count"],
            "healthy": group["healthy"],
            "diseased": group["diseased"],
            "unknown": group["unknown"],
            "top_disease": group.get("top_disease") or "",
            "diseases": diseases,
        })
    return {
        "labels": [row["label"] for row in rows],
        "keys": [row["key"] for row in rows],
        "full_names": [row["full_name"] for row in rows],
        "scans": [row["scans"] for row in rows],
        "healthy": [row["healthy"] for row in rows],
        "diseased": [row["diseased"] for row in rows],
        "unknown": [row["unknown"] for row in rows],
        "top_diseases": [row["top_disease"] for row in rows],
        "diseases": [row["diseases"] for row in rows],
    }


def _build_timeline_chart(scans: list[LeafScan], lang: str = "en") -> dict:
    """Line-chart payload: scans and diseased counts over time."""
    empty = {
        "labels": [],
        "scans": [],
        "diseased": [],
        "healthy": [],
        "cumulative_scans": [],
        "cumulative_diseased": [],
        "grain": "day",
        "peak_scans_label": "",
        "peak_diseased_label": "",
    }
    dated = [scan for scan in scans if scan.uploaded_at]
    if not dated:
        return empty

    stamps = [scan.uploaded_at for scan in dated]
    first, last = min(stamps), max(stamps)
    span_hours = max((last - first).total_seconds() / 3600.0, 0.0)

    # Same-day activity → hourly buckets so the line has shape.
    if span_hours <= 36:
        grain = "hour"
        def bucket_key(dt):
            local = timezone.localtime(dt)
            return local.replace(minute=0, second=0, microsecond=0)

        def label_for(dt):
            return format_timeline_label(dt, "hour", lang)
    elif span_hours <= 24 * 60:
        grain = "day"
        def bucket_key(dt):
            local = timezone.localtime(dt)
            return local.replace(hour=0, minute=0, second=0, microsecond=0)

        def label_for(dt):
            return format_timeline_label(dt, "day", lang)
    else:
        grain = "week"
        def bucket_key(dt):
            local = timezone.localtime(dt).replace(hour=0, minute=0, second=0, microsecond=0)
            return local - timedelta(days=local.weekday())

        def label_for(dt):
            return format_timeline_label(dt, "week", lang)
    buckets: dict = {}
    for scan in dated:
        key = bucket_key(scan.uploaded_at)
        if key not in buckets:
            buckets[key] = {"scans": 0, "diseased": 0, "healthy": 0}
        buckets[key]["scans"] += 1
        health = scan.result_health if scan.result_health in ("healthy", "diseased") else "unknown"
        if health in ("healthy", "diseased"):
            buckets[key][health] += 1

    ordered = sorted(buckets.items(), key=lambda item: item[0])
    labels = [label_for(key) for key, _ in ordered]
    scans_series = [row["scans"] for _, row in ordered]
    diseased_series = [row["diseased"] for _, row in ordered]
    healthy_series = [row["healthy"] for _, row in ordered]

    cum_scans = []
    cum_diseased = []
    running_scans = 0
    running_diseased = 0
    for total, diseased in zip(scans_series, diseased_series):
        running_scans += total
        running_diseased += diseased
        cum_scans.append(running_scans)
        cum_diseased.append(running_diseased)

    peak_scans_label = ""
    peak_diseased_label = ""
    if scans_series:
        peak_i = max(range(len(scans_series)), key=lambda i: (scans_series[i], -i))
        peak_scans_label = labels[peak_i]
    if any(diseased_series):
        peak_d = max(range(len(diseased_series)), key=lambda i: (diseased_series[i], -i))
        peak_diseased_label = labels[peak_d]

    return {
        "labels": labels,
        "scans": scans_series,
        "diseased": diseased_series,
        "healthy": healthy_series,
        "cumulative_scans": cum_scans,
        "cumulative_diseased": cum_diseased,
        "grain": grain,
        "peak_scans_label": peak_scans_label,
        "peak_diseased_label": peak_diseased_label,
    }


@never_cache
def scan_locations(request):
    lang = get_lang(request)
    located = list(
        LeafScan.objects.filter(latitude__isnull=False, longitude__isnull=False)
        .order_by("-uploaded_at")
    )
    groups = _build_location_groups(located)
    selected_key = _normalize_place_query((request.GET.get("place") or "").strip())
    selected = next((group for group in groups if group["key"] == selected_key), None)
    place_scans = [
        scan for scan in located if selected and _location_key(scan) == selected["key"]
    ]
    timeline_scans = place_scans if selected else located
    timeline_chart = _build_timeline_chart(timeline_scans, lang=lang)

    total = LeafScan.objects.count()
    located_count = len(located)
    diseases = (
        LeafScan.objects.filter(latitude__isnull=False, longitude__isnull=False)
        .exclude(disease_name="")
        .values("disease_name")
        .annotate(count=Count("id"))
        .order_by("-count", "disease_name")[:8]
    )
    disease_max = diseases[0]["count"] if diseases else 1
    barangay_chart = _build_barangay_chart(groups, lang=lang)
    barangay_rows = [
        {
            "label": label,
            "key": key,
            "full_name": full_name,
            "scans": scans,
            "healthy": healthy,
            "diseased": diseased,
            "top_disease": top_disease,
            "diseases": disease_list,
        }
        for label, key, full_name, scans, healthy, diseased, top_disease, disease_list in zip(
            barangay_chart["labels"],
            barangay_chart["keys"],
            barangay_chart["full_names"],
            barangay_chart["scans"],
            barangay_chart["healthy"],
            barangay_chart["diseased"],
            barangay_chart["top_diseases"],
            barangay_chart["diseases"],
        )
    ]

    markers = [
        {
            "key": group["key"],
            "name": group["name"],
            "lat": group["lat"],
            "lng": group["lng"],
            "count": group["count"],
            "healthy": group["healthy"],
            "diseased": group["diseased"],
            "unknown": group["unknown"],
            "top_disease": group["top_disease"],
            "status": group["status"],
            "radius_m": group["radius_m"],
            "points": group["points"],
        }
        for group in groups
    ]

    return render(request, "detection/locations.html", {
        "location_groups": groups,
        "selected": selected,
        "place_scans": place_scans,
        "recent_located": located[:8],
        "total": total,
        "located_count": located_count,
        "unlocated_count": max(total - located_count, 0),
        "unique_places": len(groups),
        "healthy_count": sum(1 for scan in located if scan.result_health == "healthy"),
        "diseased_count": sum(1 for scan in located if scan.result_health == "diseased"),
        "diseases": diseases,
        "disease_max": disease_max,
        "markers": markers,
        "barangay_chart": barangay_chart,
        "barangay_rows": barangay_rows,
        "timeline_chart": timeline_chart,
        "source_exif": sum(1 for scan in located if scan.location_source == "exif"),
        "source_device": sum(1 for scan in located if scan.location_source == "device"),
        "max_count": groups[0]["count"] if groups else 1,
        "selected_key": selected["key"] if selected else "",
    })


def _run_scan_pipeline(scan: LeafScan) -> None:
    """Populate a LeafScan from its uploaded image (lenient Layer 0 + pipeline)."""
    try:
        from ml.pipeline import run_pipeline

        result = run_pipeline(scan.image.path)
        scan.result_health = result.get("health", "unknown")
        scan.disease_name = result.get("disease_name", "")
        scan.confidence = result.get("confidence", 0.0)
        scan.all_disease_probs = result.get("all_disease_probs", {})
        scan.recommendation = result.get("recommendation", "")
        scan.gate_warning = result.get("gate_warning", "")
        scan.rejection_reason = result.get("rejection_reason", "")
    except Exception as exc:
        scan.result_health = "unknown"
        scan.processing_error = str(exc)
    scan.save()


def _create_scan_from_upload(request):
    if "image" not in request.FILES:
        return Response(
            {"error": get_strings(get_lang(request))["no_image_api"]},
            status=status.HTTP_400_BAD_REQUEST,
        )

    upload = request.FILES["image"]
    ok, code, message = validate_leaf_upload(upload, lang=get_lang(request))
    if not ok:
        return Response(
            {"accepted": False, "code": code, "error": message, "message": message},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if hasattr(upload, "seek"):
        upload.seek(0)

    scan = LeafScan(image=upload)
    scan.save()
    resolve_scan_location(scan, request=request, lang=get_lang(request))
    _run_scan_pipeline(scan)
    serializer = LeafScanSerializer(scan, context={"request": request})
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(["POST"])
def api_scan(request):
    """Create a scan (alias of POST /api/scans/)."""
    return _create_scan_from_upload(request)


@api_view(["GET", "POST"])
def api_scan_list(request):
    """List all scans (GET) or upload and analyze a leaf image (POST)."""
    if request.method == "POST":
        return _create_scan_from_upload(request)

    scans = LeafScan.objects.all()
    health_filter = request.query_params.get("health", "")
    if health_filter in ("healthy", "diseased", "unknown"):
        scans = scans.filter(result_health=health_filter)

    serializer = LeafScanSerializer(scans, many=True, context={"request": request})
    return Response(serializer.data)


@api_view(["GET"])
def api_scan_detail(request, pk):
    scan = get_object_or_404(LeafScan, pk=pk)
    serializer = LeafScanSerializer(scan, context={"request": request})
    return Response(serializer.data)
