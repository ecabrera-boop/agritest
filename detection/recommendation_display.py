"""Parse stored recommendation text into structured display data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RecommendationDisplay:
    kind: str  # "treatment", "notice", "plain"
    title: str = ""
    subtitle: str = ""
    intro: str = ""
    steps: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    text: str = ""


_TITLE_RE = re.compile(r"^(.+?):\s*(.+)$", re.DOTALL)
_STEP_SPLIT_RE = re.compile(r"\s*(?=\d+\.\s)")
_STEP_PREFIX_RE = re.compile(r"^\d+\.\s*")


def parse_recommendation(text: str) -> RecommendationDisplay:
    if not text or not text.strip():
        return RecommendationDisplay(kind="plain", text="")

    text = text.strip()

    title_match = _TITLE_RE.match(text)
    if title_match:
        title_part = title_match.group(1).strip()
        body = title_match.group(2).strip()
        raw_steps = [part.strip() for part in _STEP_SPLIT_RE.split(body) if part.strip()]
        steps = [_STEP_PREFIX_RE.sub("", step).strip() for step in raw_steps]
        if len(steps) >= 2 or (len(steps) == 1 and re.match(r"^\d+\.", body)):
            title, subtitle = _split_title(title_part)
            return RecommendationDisplay(
                kind="treatment",
                title=title,
                subtitle=subtitle,
                steps=steps,
            )

    if "\n" in text or "•" in text:
        paragraphs: list[str] = []
        bullets: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("•"):
                bullets.append(line.lstrip("•").strip())
            else:
                paragraphs.append(line)

        intro = "\n\n".join(paragraphs)
        if bullets:
            return RecommendationDisplay(
                kind="notice",
                intro=intro,
                bullets=bullets,
            )
        return RecommendationDisplay(kind="plain", text=intro)

    return RecommendationDisplay(kind="plain", text=text)


def _split_title(title_part: str) -> tuple[str, str]:
    paren_match = re.match(r"^(.+?)\s(\([^)]+\))$", title_part)
    if paren_match:
        return paren_match.group(1).strip(), paren_match.group(2).strip()
    return title_part, ""
