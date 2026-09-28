"""
fil_ilocano_dict
=================

A small Filipino <-> Ilocano word-translation library.

Basic usage:

    from fil_ilocano_dict import Dictionary

    d = Dictionary()
    d.to_ilocano("salamat")          -> "agyamanak"
    d.to_filipino("wen")             -> "oo"
    d.search("water")                -> matches on notes/definitions
    d.categories()                   -> list of word categories
    d.add_word("bagong salita", "baro a sao", category="misc")
    d.all_words()                    -> full list of entries
"""

import json
import os
from difflib import get_close_matches

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "words.json")


class Dictionary:
    def __init__(self, data_path: str = _DATA_PATH):
        self._data_path = data_path
        self._entries = []          # flat list of dicts
        self._fil_index = {}        # filipino (lower) -> entry
        self._ilo_index = {}        # ilocano (lower) -> entry
        self._load()

    def _load(self):
        with open(self._data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._entries = []
        self._fil_index = {}
        self._ilo_index = {}

        for category, words in raw.items():
            for entry in words:
                item = {
                    "filipino": entry["filipino"],
                    "ilocano": entry["ilocano"],
                    "notes": entry.get("notes", ""),
                    "category": category,
                }
                self._entries.append(item)
                self._fil_index[item["filipino"].lower()] = item
                # some ilocano fields list multiple variants separated by " / "
                for variant in item["ilocano"].split(" / "):
                    self._ilo_index.setdefault(variant.strip().lower(), item)

    # -- lookups -----------------------------------------------------

    def to_ilocano(self, filipino_word: str):
        """Return the Ilocano translation of a Filipino word, or None."""
        entry = self._fil_index.get(filipino_word.strip().lower())
        return entry["ilocano"] if entry else None

    def to_filipino(self, ilocano_word: str):
        """Return the Filipino translation of an Ilocano word, or None."""
        entry = self._ilo_index.get(ilocano_word.strip().lower())
        return entry["filipino"] if entry else None

    def suggest(self, word: str, n: int = 3):
        """Fuzzy-match suggestions when an exact lookup fails."""
        pool = list(self._fil_index.keys()) + list(self._ilo_index.keys())
        return get_close_matches(word.strip().lower(), pool, n=n, cutoff=0.6)

    def search(self, query: str):
        """Search across filipino, ilocano, and notes fields."""
        q = query.strip().lower()
        return [
            e for e in self._entries
            if q in e["filipino"].lower()
            or q in e["ilocano"].lower()
            or q in e["notes"].lower()
        ]

    # -- browsing ------------------------------------------------------

    def categories(self):
        return sorted(set(e["category"] for e in self._entries))

    def by_category(self, category: str):
        return [e for e in self._entries if e["category"] == category]

    def all_words(self):
        return list(self._entries)

    def __len__(self):
        return len(self._entries)

    # -- extending -------------------------------------------------------

    def add_word(self, filipino: str, ilocano: str, notes: str = "", category: str = "misc"):
        """Add a new word pair at runtime (in-memory only, unless save() is called)."""
        item = {"filipino": filipino, "ilocano": ilocano, "notes": notes, "category": category}
        self._entries.append(item)
        self._fil_index[filipino.lower()] = item
        self._ilo_index[ilocano.lower()] = item
        return item

    def save(self, path: str = None):
        """Persist the current in-memory word list back to a JSON file,
        grouped by category (same shape as the original data file)."""
        target = path or self._data_path
        grouped = {}
        for e in self._entries:
            grouped.setdefault(e["category"], []).append({
                "filipino": e["filipino"],
                "ilocano": e["ilocano"],
                "notes": e["notes"],
            })
        with open(target, "w", encoding="utf-8") as f:
            json.dump(grouped, f, ensure_ascii=False, indent=2)
