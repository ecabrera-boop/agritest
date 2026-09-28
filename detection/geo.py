"""Capture-site GPS: EXIF fallback, device coords, and reverse geocoding."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from PIL import Image
from PIL.ExifTags import GPSTAGS, IFD

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "RiceGuard/1.0 (rice leaf disease detector; local research app)"
GEOCODE_TIMEOUT = 4


def parse_coordinates(latitude, longitude) -> Optional[tuple[float, float]]:
    """Return a valid (lat, lng) pair or None."""
    try:
        lat = float(latitude)
        lng = float(longitude)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    if lat == 0.0 and lng == 0.0:
        return None
    return lat, lng


def coordinates_from_request(request) -> Optional[tuple[float, float]]:
    data = getattr(request, "POST", None) or {}
    return parse_coordinates(data.get("latitude"), data.get("longitude"))


def _to_degrees(values) -> Optional[float]:
    try:
        deg, minutes, seconds = values[:3]
        return float(deg) + float(minutes) / 60.0 + float(seconds) / 3600.0
    except (TypeError, ValueError, IndexError):
        return None


def extract_gps_from_image(source) -> Optional[tuple[float, float]]:
    """Read GPS from JPEG/TIFF EXIF. Canvas camera captures have none."""
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        with Image.open(source) as img:
            exif = img.getexif()
            if not exif:
                return None
            gps_ifd = exif.get_ifd(IFD.GPSInfo)
            if not gps_ifd:
                return None
            gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
    except Exception:
        logger.debug("Could not read EXIF GPS from image", exc_info=True)
        return None
    finally:
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass

    lat = _to_degrees(gps.get("GPSLatitude") or ())
    lng = _to_degrees(gps.get("GPSLongitude") or ())
    if lat is None or lng is None:
        return None
    if str(gps.get("GPSLatitudeRef", "N")).upper().startswith("S"):
        lat = -lat
    if str(gps.get("GPSLongitudeRef", "E")).upper().startswith("W"):
        lng = -lng
    return parse_coordinates(lat, lng)


def _short_place_name(payload: dict) -> str:
    address = payload.get("address") or {}
    parts: list[str] = []

    for key in (
        "hamlet",
        "village",
        "suburb",
        "neighbourhood",
        "quarter",
        "city_district",
    ):
        value = (address.get(key) or "").strip()
        if value:
            parts.append(value)
            break

    for key in ("city", "town", "municipality", "county"):
        value = (address.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
            break

    for key in ("state", "province", "region"):
        value = (address.get(key) or "").strip()
        if value and value not in parts:
            parts.append(value)
            break

    country = (address.get("country") or "").strip()
    if country and country not in parts:
        parts.append(country)

    if parts:
        return ", ".join(parts)

    display = (payload.get("display_name") or "").strip()
    return display[:255] if display else ""


def reverse_geocode(latitude: float, longitude: float, lang: str = "en") -> str:
    """Turn coordinates into a short place name via OpenStreetMap Nominatim."""
    if lang == "ilo":
        accept_lang = "ilo,tl,fil,en"
    elif lang == "fil":
        accept_lang = "tl,fil,en"
    else:
        accept_lang = "en"
    query = urllib.parse.urlencode(
        {
            "lat": f"{latitude:.7f}",
            "lon": f"{longitude:.7f}",
            "format": "jsonv2",
            "zoom": 16,
            "addressdetails": 1,
        }
    )
    request = urllib.request.Request(
        f"{NOMINATIM_URL}?{query}",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": accept_lang,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=GEOCODE_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        logger.info("Reverse geocode failed for %s, %s", latitude, longitude)
        return ""
    if not isinstance(payload, dict):
        return ""
    return _short_place_name(payload)[:255]


def location_dict(latitude: float, longitude: float, name: str = "") -> dict:
    name = (name or "").strip()
    coords = f"{latitude:.5f}, {longitude:.5f}"
    delta = 0.012
    return {
        "latitude": latitude,
        "longitude": longitude,
        "location_name": name,
        "location_display": name or coords,
        "coordinates_display": coords,
        "maps_url": (
            f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}"
            f"#map=17/{latitude}/{longitude}"
        ),
        "map_embed_url": (
            "https://www.openstreetmap.org/export/embed.html"
            f"?bbox={longitude - delta},{latitude - delta},{longitude + delta},{latitude + delta}"
            f"&layer=mapnik&marker={latitude},{longitude}"
        ),
    }


def resolve_request_location(request, upload=None, lang: str = "en") -> Optional[dict]:
    """EXIF GPS from the upload first, then device coordinates from the form."""
    coords = extract_gps_from_image(upload) if upload is not None else None
    source = "exif" if coords else ""
    if coords is None:
        coords = coordinates_from_request(request)
        if coords:
            source = "device"
    if coords is None:
        return None
    lat, lng = coords
    payload = location_dict(lat, lng, reverse_geocode(lat, lng, lang=lang))
    payload["location_source"] = source
    return payload


def resolve_scan_location(scan, request=None, lang: str = "en") -> bool:
    """
    Fill latitude/longitude/location_name on a saved LeafScan.

    Prefer EXIF GPS from the photo, then device coordinates from the request.
    Returns True when coordinates were applied.
    """
    source = ""
    coords = None
    image_path = getattr(getattr(scan, "image", None), "path", "") or ""
    if image_path:
        coords = extract_gps_from_image(image_path)
        if coords:
            source = "exif"

    if coords is None and request is not None:
        coords = coordinates_from_request(request)
        if coords:
            source = "device"

    if coords is None:
        return False

    lat, lng = coords
    scan.latitude = lat
    scan.longitude = lng
    scan.location_source = source
    if not (scan.location_name or "").strip():
        scan.location_name = reverse_geocode(lat, lng, lang=lang)
    return True
