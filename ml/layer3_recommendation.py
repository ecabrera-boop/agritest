"""
Layer 3 — Treatment recommendation engine.

Disease-specific text is built from detection.disease_guide (PhilRice-aligned).
Non-disease keys (upload errors, healthy notice) stay in RECOMMENDATIONS.
"""

from detection.disease_guide import get_farmer_guide

RECOMMENDATIONS: dict[str, str] = {
    "not_rice": (
        "This does not appear to be a rice leaf. RiceGuard is trained only on narrow "
        "rice leaf images from the UCI Rice Leaf Diseases dataset.\n\n"
        "Please upload a clear photo of a rice leaf (long, grass-like blade) taken "
        "against a plain background. Broad or heart-shaped leaves from other plants "
        "cannot be diagnosed by this model."
    ),
    "wrong_shape": (
        "The main green region in this photo does not look like a narrow rice leaf blade "
        "(long and thin). RiceGuard expects a single elongated rice leaf, not a round blob, "
        "broad leaf, or scattered green patches.\n\n"
        "Please upload a clear photo of one rice leaf blade against a plain background."
    ),
    "low_confidence": (
        "The model could not confidently match this image to any class in its training "
        "database. No disease diagnosis is shown.\n\n"
        "Try again with:\n"
        "• A closer, well-lit photo of a single rice leaf\n"
        "• Minimal background clutter\n"
        "• The full leaf visible and in focus\n\n"
        "If the leaf is not rice, this tool cannot classify it."
    ),
    "unknown": (
        "Unable to classify this image. It may not be a rice leaf, or the photo quality "
        "may be too low for a reliable match. Please upload a clear rice leaf image."
    ),
    "healthy": (
        "Healthy Rice Plant (No pathogen detected): "
        "1. Maintain proper water depth and drainage suited to your growth stage. "
        "2. Apply fertiliser based on soil test and PhilRice Nutrient Manager guidance. "
        "3. Scout twice weekly during the rainy season for early signs of BLB, blast, and tungro. "
        "4. Keep field bunds clean and remove weeds that host pests and diseases."
    ),
}

_DEFAULT_RECOMMENDATION = (
    "Disease detected but no specific recommendation is available for '{disease}'. "
    "Please consult your local agricultural extension officer or plant pathologist "
    "for tailored advice. In the meantime, avoid spreading material from this plant "
    "to healthy areas and monitor nearby plants closely."
)


def get_recommendation(disease_name: str) -> str:
    """
    Look up the treatment recommendation for a given disease name.

    Parameters
    ----------
    disease_name : str
        Disease label returned by Layer 2 (or "healthy" from Layer 1).

    Returns
    -------
    str
        Human-readable treatment recommendation.
    """
    key = disease_name.strip().lower()

    guide = get_farmer_guide(disease_name)
    if guide and guide.key != "healthy":
        return guide.to_recommendation_text()

    if key in RECOMMENDATIONS:
        return RECOMMENDATIONS[key]

    for known_key, text in RECOMMENDATIONS.items():
        if known_key in key or key in known_key:
            return text

    return _DEFAULT_RECOMMENDATION.format(disease=disease_name)
