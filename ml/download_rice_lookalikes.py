"""
Download photos of rice look-alike plants and ingest them as rice-gate negatives.

Pulls research-grade, openly-licensed observation photos from iNaturalist for
each species that is commonly mistaken for rice (barnyardgrass, watergrass,
rice cutgrass, wild/red rice, wheat, barley, ...), stages them under
downloads/rice_lookalikes/<species>/, then (with --ingest) copies them into
leaf/not_rice/ using the same MD5 + dHash dedup as ml/add_not_rice.py.

Examples
--------
  # download ~80 photos per species and ingest into leaf/not_rice/:
  python ml/download_rice_lookalikes.py --ingest

  # smaller test run, staging only:
  python ml/download_rice_lookalikes.py --per-species 10

  # afterwards retrain so the gate learns the new negatives:
  python ml/train_models.py
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

API_URL = "https://api.inaturalist.org/v1/observations"
STAGING_DIR = _PROJECT_ROOT / "downloads" / "rice_lookalikes"
HEADERS = {"User-Agent": "rice-disease-project/1.0 (training-data collection)"}
# Open licenses only, so the images are safe to keep in the training set.
PHOTO_LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa"

RICE_LOOKALIKES = [
    {
        "id": 1,
        "scientific_name": "Echinochloa crus-galli",
        "common_name": "Common barnyardgrass",
        "similarity": "Very high",
    },
    {
        "id": 2,
        "scientific_name": "Echinochloa oryzicola",
        "common_name": "Early watergrass",
        "similarity": "Extremely high",
    },
    {
        "id": 3,
        "scientific_name": "Leersia oryzoides",
        "common_name": "Rice cutgrass",
        "similarity": "High",
    },
    {
        "id": 4,
        "scientific_name": "Oryza rufipogon",
        "common_name": "Red rice",
        "similarity": "Very high",
    },
    {
        "id": 5,
        "scientific_name": "Oryza longistaminata",
        "common_name": "Long-stamen rice",
        "similarity": "High",
    },
    {
        "id": 6,
        "scientific_name": "Zizania",
        "common_name": "Wild rice",
        "similarity": "Moderate",
    },
    {
        "id": 7,
        "scientific_name": "Leptochloa chinensis",
        "common_name": "Chinese sprangletop",
        "similarity": "Moderate",
    },
    {
        "id": 8,
        "scientific_name": "Triticum aestivum",
        "common_name": "Wheat",
        "similarity": "Moderate (early seedling only)",
    },
    {
        "id": 9,
        "scientific_name": "Hordeum vulgare",
        "common_name": "Barley",
        "similarity": "Moderate (early seedling only)",
    },
    {
        "id": 10,
        "scientific_name": "Avena sativa",
        "common_name": "Oat",
        "similarity": "Low to moderate (early stage)",
    },
    {
        "id": 11,
        "scientific_name": "Digitaria",
        "common_name": "Crabgrass",
        "similarity": "Low",
    },
    {
        "id": 12,
        "scientific_name": "Setaria",
        "common_name": "Foxtail grass",
        "similarity": "Low",
    },
]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _photo_url(photo: dict) -> str | None:
    """Turn the API's thumbnail URL into the 500px 'medium' version."""
    url = photo.get("url")
    if not url:
        return None
    return url.replace("square", "medium")


def _download_species(
    session: requests.Session,
    species: dict,
    dest: Path,
    limit: int,
) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in dest.glob("*.jpg")}
    saved = len(existing)
    if saved >= limit:
        print(f"  already have {saved} photos, skipping download")
        return 0

    new_count = 0
    page = 1
    # Sparse taxa (e.g. Echinochloa oryzicola) have almost no openly-licensed
    # photos; fall back to any license since these stay local as training data.
    license_filter: str | None = PHOTO_LICENSES
    while saved < limit and page <= 10:
        params = {
            "taxon_name": species["scientific_name"],
            "quality_grade": "research",
            "photos": "true",
            "per_page": 200,
            "page": page,
            "order_by": "votes",
        }
        if license_filter:
            params["photo_license"] = license_filter
        try:
            resp = session.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  API error on page {page}: {exc}")
            break

        results = resp.json().get("results", [])
        if not results:
            if page == 1 and license_filter:
                print("  no openly-licensed photos; retrying without license filter")
                license_filter = None
                continue
            break

        for obs in results:
            if saved >= limit:
                break
            photos = obs.get("photos") or []
            if not photos:
                continue
            # one photo per observation keeps the negatives diverse
            url = _photo_url(photos[0])
            if url is None:
                continue
            name = f"obs{obs['id']}_p{photos[0]['id']}"
            if name in existing:
                continue
            try:
                img = session.get(url, timeout=30)
                img.raise_for_status()
            except requests.RequestException:
                continue
            if len(img.content) < 2048:  # placeholder / broken image
                continue
            (dest / f"{name}.jpg").write_bytes(img.content)
            existing.add(name)
            saved += 1
            new_count += 1
            time.sleep(0.1)

        page += 1
        time.sleep(1.0)  # be polite to the iNaturalist API

    return new_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download rice look-alike photos from iNaturalist for the rice gate."
    )
    parser.add_argument(
        "--per-species",
        type=int,
        default=80,
        help="Target number of photos per species (default: 80).",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="After downloading, copy into leaf/not_rice/ with dedup.",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    total_new = 0
    for species in RICE_LOOKALIKES:
        label = f"{species['scientific_name']} ({species['common_name']})"
        print(f"\n[{species['id']:>2}/12] {label} — similarity: {species['similarity']}")
        dest = STAGING_DIR / f"{species['id']:02d}_{_slug(species['scientific_name'])}"
        added = _download_species(session, species, dest, args.per_species)
        have = len(list(dest.glob("*.jpg")))
        total_new += added
        print(f"  downloaded {added} new photo(s); {have} staged in {dest.relative_to(_PROJECT_ROOT)}")

    print(f"\nDone. {total_new} new photo(s) downloaded to {STAGING_DIR}")

    if args.ingest:
        print("\nIngesting staged photos into leaf/not_rice/ (with dedup) ...")
        from ml.add_not_rice import add_not_rice

        add_not_rice([str(STAGING_DIR)])
        print(
            "\nNext step — retrain so the gate learns them:\n"
            "  python ml/train_models.py      (both models)\n"
            "  python ml/train_cnn.py         (CNN only, faster)"
        )


if __name__ == "__main__":
    main()
