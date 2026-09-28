"""
Add non-rice (look-alike) images into leaf/not_rice/ so the rice gate learns them.

Point it at a folder (or individual images) of plants/objects that are being
wrongly recognised as rice. It copies the images into the training negatives
folder, skipping any that already exist there (byte-identical MD5 or re-encoded
dHash duplicates), then optionally retrains so the change takes effect.

Examples
--------
  # add every image under a folder, then retrain the CNN gate:
  python ml/add_not_rice.py "C:/Users/me/Downloads/other_plants" --retrain

  # just stage the images (retrain later with: python ml/train_models.py):
  python ml/add_not_rice.py img1.jpg img2.jpg some_folder/
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ML_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ML_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.dataset_io import (  # noqa: E402
    IMAGE_SUFFIXES,
    _content_digest,
    _perceptual_digest,
    default_training_root,
    resolve_not_rice_dir,
)


def _iter_images(sources: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            out += [q for q in sorted(p.rglob("*"))
                    if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES]
        elif p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
        else:
            print(f"  (skipped, not an image or missing): {s}")
    return out


def _existing_digests(dest: Path) -> tuple[set[str], set[bytes]]:
    """MD5 + dHash of images already in the not_rice folder."""
    md5s: set[str] = set()
    dhashes: set[bytes] = set()
    for q in dest.rglob("*"):
        if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES:
            try:
                md5s.add(_content_digest(q))
            except OSError:
                continue
            ph = _perceptual_digest(q)
            if ph is not None:
                dhashes.add(ph)
    return md5s, dhashes


def _unique_dest_name(dest: Path, src: Path, md5: str) -> Path:
    """Collision-proof filename: <stem>_<md5[:8]><suffix> under dest."""
    stem = src.stem[:40]
    candidate = dest / f"{stem}_{md5[:8]}{src.suffix.lower()}"
    n = 1
    while candidate.exists():
        candidate = dest / f"{stem}_{md5[:8]}_{n}{src.suffix.lower()}"
        n += 1
    return candidate


def add_not_rice(sources: list[str]) -> int:
    dest = resolve_not_rice_dir(default_training_root())
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()

    candidates = _iter_images(sources)
    if not candidates:
        print("No images found to add.")
        return 0

    print(f"Destination: {dest_resolved}")
    print(f"Scanning {len(candidates)} candidate image(s); indexing existing negatives ...")
    md5s, dhashes = _existing_digests(dest)

    added = exact_dupes = near_dupes = 0
    for src in candidates:
        if src.resolve().parent == dest_resolved:
            continue  # already in the folder
        try:
            md5 = _content_digest(src)
        except OSError:
            print(f"  (unreadable, skipped): {src}")
            continue
        if md5 in md5s:
            exact_dupes += 1
            continue
        ph = _perceptual_digest(src)
        if ph is not None and ph in dhashes:
            near_dupes += 1
            continue

        target = _unique_dest_name(dest, src, md5)
        shutil.copy2(src, target)
        md5s.add(md5)
        if ph is not None:
            dhashes.add(ph)
        added += 1

    total_now = sum(
        1 for q in dest.rglob("*") if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES
    )
    print(
        f"\nAdded {added} new not_rice image(s); "
        f"skipped {exact_dupes} identical + {near_dupes} re-encoded duplicate(s)."
    )
    print(f"leaf/not_rice/ now holds {total_now} negative image(s).")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest non-rice look-alike images into leaf/not_rice/ (with dedup)."
    )
    parser.add_argument("sources", nargs="+", help="Image files and/or folders to add.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="After adding, retrain the CNN gate (fast; embeddings are cached).",
    )
    args = parser.parse_args()

    added = add_not_rice(args.sources)

    if args.retrain and added:
        print("\nRetraining the neural-network gate on the new negatives ...")
        from ml.train_cnn import main as train_cnn_main

        train_cnn_main()
        print(
            "\nCNN gate updated. To also refresh the RF fallback gate, run:\n"
            "  python ml/train_models.py"
        )
    elif added:
        print(
            "\nNext step — retrain so the gate learns them:\n"
            "  python ml/train_models.py      (both models)\n"
            "  python ml/train_cnn.py         (CNN only, faster)"
        )


if __name__ == "__main__":
    main()
