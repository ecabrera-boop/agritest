"""English / Tagalog / Ilocano UI strings. Language is stored in the riceguard_lang cookie.

Filipino in the UI is a family choice: Tagalog (`fil`) or Ilocano (`ilo`).
"""

from __future__ import annotations

import json
import re

from django.http import HttpRequest
from django.utils.safestring import mark_safe
from fil_ilocano_dict import Dictionary

LANG_COOKIE = "riceguard_lang"
DIALECT_COOKIE = "riceguard_fil_dialect"
SUPPORTED = ("en", "fil", "ilo")
FILIPINO_LANGS = ("fil", "ilo")
DEFAULT_LANG = "ilo"
DEFAULT_FILIPINO_LANG = "fil"

ILOCANO_DICTIONARY = Dictionary()


def ilocano_word(filipino_word: str, fallback: str = "") -> str:
    """Return the glossary's preferred Ilocano word without exposing variants.

    Full UI sentences remain human-written because word-for-word replacement
    does not preserve Ilocano grammar. The fallback supports agricultural
    terms that are outside the bundled general-purpose glossary.
    """
    translated = ILOCANO_DICTIONARY.to_ilocano(filipino_word)
    if not translated:
        return fallback
    return translated.split(" / ", 1)[0].strip()


_REASON_RE = re.compile(r"reason:\s*(\w+)", re.I)

JS_KEYS = (
    "unsupported_format_title",
    "unsupported_format_msg",
    "file_too_large_title",
    "file_too_large_msg",
    "not_rice_leaf_title",
    "not_rice_leaf_msg",
    "could_not_check_title",
    "could_not_check_msg",
    "upload_problem",
    "checking_image",
    "cannot_analyse",
    "not_passed_check",
    "choose_image_first",
    "analysing_ellipsis",
    "analysis_could_not_finish",
    "connection_timeout",
    "upload_failed",
    "no_result_url",
    "camera_not_available",
    "capturing",
    "could_not_validate",
    "location_captured",
    "location_unavailable",
    "finding_location",
    "open_map",
    "location_denied",
    "allow_location_hint",
    "use_my_location",
    "getting_location",
)


UI = {
    "en": {
        "html_lang": "en",
        "home": "Home",
        "history": "History",
        "locations": "Locations",
        "api": "API",
        "admin": "Admin",
        "lang_label": "Language",
        "lang_dialect_label": "Filipino dialect",
        "lang_en": "English",
        "lang_fil": "Filipino",
        "lang_tag": "Tagalog",
        "lang_ilo": "Ilocano",
        "hero_title": "Rice Leaf Disease Detector",
        "hero_tagline": (
            "Upload a photo of a rice leaf — our 3-layer AI pipeline will identify "
            "any disease and recommend treatment in seconds."
        ),
        "upload_title": "Upload a Leaf Image",
        "upload_help": (
            "After you choose a photo, Layer 0 checks that it looks like a rice leaf. "
            "Only then can you run analysis on the original image."
        ),
        "accepted": "Accepted:",
        "accepted_hint": "JPG, PNG, or WEBP · max 10 MB · clear photo of a rice leaf blade",
        "drop_title": "Drag & drop your image here",
        "drop_sub": "or click to browse / use camera below",
        "choose_image": "Choose image",
        "file_types": "JPG, PNG, WEBP — max 10 MB",
        "selected_image": "Selected image",
        "remove_photo": "Remove photo",
        "checking_image": "Checking image…",
        "choose_different": "Choose a different image",
        "analyse_leaf": "Analyse Leaf",
        "use_camera": "Use Camera",
        "capture": "Capture",
        "camera_hint": "Hold steady — center the leaf and keep it in focus",
        "capturing": "Capturing…",
        "close_camera": "Close camera",
        "analysing": "Analysing leaf… this may take a few seconds.",
        "recent_scans": "Recent Scans",
        "ago": "ago",
        "view_all_history": "View All History",
        "view_all_locations": "View All Locations",
        "no_scans_yet": "No scans yet. Upload your first leaf!",
        "how_it_works": "How it works",
        "howit_1_title": "Upload a leaf photo",
        "howit_1_desc": (
            "Upload a clear photo of a rice leaf blade — Layer 0 checks it before the AI runs."
        ),
        "howit_2_title": "AI pipeline analyses it",
        "howit_2_desc": (
            "Layer 0 validates the image, then three Random Forest layers detect health, "
            "identify the disease, and measure confidence."
        ),
        "howit_3_title": "Get a recommendation",
        "howit_3_desc": (
            "Receive a clear, actionable treatment recommendation matched to the specific disease found."
        ),
        "scan_history": "Scan History",
        "new_scan": "New Scan",
        "total_scans": "Total Scans",
        "health_healthy": "Healthy",
        "health_unhealthy": "Unhealthy",
        "health_unknown": "Unknown",
        "health_not_rice": "Not a Rice Leaf",
        "not_rice_note": (
            "This photo does not look like a rice leaf, so no disease analysis was made. "
            "Upload a clear photo of a single rice leaf blade."
        ),
        "filter": "Filter:",
        "filter_all": "All",
        "showing_scans": "Showing {shown} of {total} scans",
        "no_scans_found": "No scans found.",
        "no_scans_found_filter": "No scans found for this filter.",
        "previous": "Previous",
        "next": "Next",
        "page_of": "Page {current} of {total}",
        "upload_first_leaf": "Upload First Leaf",
        "scan_not_found": "Scan not found",
        "scan_not_found_msg": (
            "Scan #{id} is no longer available. It may have been removed or this link is outdated."
        ),
        "scan_new_leaf": "Scan a new leaf",
        "view_history": "View history",
        "print": "Print",
        "scan_report": "Rice Leaf Disease Detection — Scan Report",
        "scan_number": "Scan #",
        "date": "Date:",
        "printed": "Printed:",
        "analysis_result": "Analysis result",
        "processing_warning": "Processing warning",
        "leaf_check_warning": "Leaf check warning",
        "your_leaf_photo": "Your leaf photo",
        "tap_to_enlarge": "Tap to enlarge",
        "close_photo_preview": "Close photo",
        "analysed_from_upload": "Analysed from your original upload.",
        "did_not_pass_leaf_check": "This image did not pass the leaf check and was not analysed.",
        "at_a_glance": "At a glance",
        "result": "Result",
        "disease": "Disease",
        "scan_time": "Scan time",
        "location": "Location",
        "open_map": "Open map",
        "location_captured": "Capture location saved",
        "location_unavailable": "Not captured",
        "finding_location": "Finding location…",
        "location_denied": "Location permission is off",
        "allow_location_hint": "Allow location so the scan can show where the leaf was photographed.",
        "use_my_location": "Use my location",
        "getting_location": "Getting device location…",
        "locations_title": "Captured Locations",
        "locations_subtitle": "Places where leaf scans were taken, ranked by how often they were captured.",
        "most_captured": "Most captured places",
        "with_location": "With location",
        "without_location": "Not captured",
        "unique_places": "Unique places",
        "scans_here": "scans",
        "last_scan": "Last scan",
        "view_scans": "View scans",
        "no_locations_yet": "No capture locations yet. Scan a leaf with location on to map your fields.",
        "common_diseases": "Common diseases",
        "recent_located": "Recent scans with location",
        "location_map": "Location map",
        "capture_zone": "Capture zone",
        "exact_captures": "Exact capture points",
        "barangay_chart_title": "Barangays with most scans",
        "barangay_chart_subtitle": "Scan volume by barangay, with healthy vs diseased counts and top diseases.",
        "timeline_chart_title": "Scan & disease timeline",
        "timeline_chart_subtitle": "When scans rise, and when diseased leaves are detected over time.",
        "timeline_scans": "Total scans",
        "timeline_diseased": "Diseased scans",
        "timeline_healthy": "Healthy scans",
        "timeline_cum_scans": "Cumulative scans",
        "timeline_cum_diseased": "Cumulative diseased",
        "timeline_peak_scans": "Peak scans",
        "timeline_peak_diseased": "Peak diseased",
        "top_disease_label": "Top disease",
        "no_disease_short": "No disease",
        "photo_gps": "From photo GPS",
        "device_gps": "From device GPS",
        "all_places": "All places",
        "scans_at_place": "Scans at this place",
        "diseased_share": "Unhealthy share",
        "capture_source": "How location was captured",
        "top_disease": "Top disease",
        "no_disease_yet": "No disease recorded yet",
        "place_rank": "Rank",
        "na": "N/A",
        "what_this_means": "What this means",
        "what_to_do_next": "What to do next",
        "recommendation": "Recommendation",
        "immediate_actions": "Immediate actions",
        "complete_procedure": "Complete treatment procedure",
        "tip": "Tip:",
        "more_guidance": "More guidance",
        "symptoms": "Symptoms to confirm",
        "cultural": "Cultural management",
        "pesticide_options": "Pesticide options",
        "spray_guidelines": "Spray guidelines",
        "prevention": "Prevention",
        "resistant_varieties": "Resistant varieties",
        "based_on": "Based on",
        "confirm_products": "Confirm products with your MAO or PhilRice technologist.",
        "reference_samples": "Reference samples",
        "best_training_match": "Best training match",
        "similarity": "Similarity:",
        "best_match_note": "This image is selected from the same predicted class as your result.",
        "more_sample_photos": "More sample photos for comparison",
        "scan_another": "Scan another leaf",
        "print_result": "Print result",
        "print_footer": (
            "Generated by {brand} — AI-assisted recommendation. Confirm with your local "
            "agricultural office (MAO) or PhilRice technologist before applying pesticides."
        ),
        "wet_or_dry": "Wet or dry rice field?",
        "field_mode_note": (
            "Choose the method that matches your field water. Foliar sprays need "
            "<strong>dry leaves</strong>; granules work best in <strong>standing water</strong>."
        ),
        "how_to_apply": "How to apply properly",
        "fpa_products": "FPA-registered product options",
        "numbered_boxes_note": (
            "Each numbered box is a <strong>different product</strong>. Choose one that is "
            "available locally. Do not mix unless the label allows it."
        ),
        "next_pesticide": "Next pesticide",
        "pesticide_n_of": "Pesticide {n} of {total}",
        "active": "Active:",
        "best_wet": "Best in wet field",
        "spray_dry": "Spray on dry leaves",
        "before_sowing": "Before sowing",
        "when": "When",
        "rate": "Rate",
        "wet_field": "Wet field (basa):",
        "dry_field": "Dry field (tuyo):",
        "show_mixing": "Show mixing / spray steps for this product",
        "gate_warning": (
            "The rice-leaf check was not fully confident about this photo "
            "(reason: {reason}). The analysis below still ran, but treat the result "
            "with extra care — retake the photo closer, in good light, with one leaf "
            "filling the frame if it looks wrong."
        ),
        "layer0_warning": (
            "This photo may not show a clear rice leaf. You can still analyse it, but "
            "the result may be less reliable — a close, well-lit photo of a single "
            "leaf blade works best."
        ),
        "unsupported_format_title": "Unsupported file format",
        "unsupported_format_msg": (
            "This file type cannot be used. Please upload a photo saved as JPG, PNG, or WEBP."
        ),
        "file_too_large_title": "File is too large",
        "file_too_large_msg": (
            "The image must be 10 MB or smaller. Try a smaller photo or lower the camera quality."
        ),
        "file_too_large_server": "This image is {size} MB. The maximum allowed size is 10 MB.",
        "not_rice_leaf_title": "Not a rice leaf",
        "not_rice_leaf_msg": (
            "This image does not look like a rice leaf. Upload a clear photo of a single rice leaf blade."
        ),
        "could_not_check_title": "Could not check image",
        "could_not_check_msg": "Something went wrong while validating your photo. Please try again.",
        "upload_problem": "Upload problem",
        "cannot_analyse": "This image cannot be analysed.",
        "not_passed_check": (
            "This image has not passed the rice-leaf check yet. Choose a clear photo of a rice leaf blade."
        ),
        "choose_image_first": "Please choose an image before analysing.",
        "analysing_ellipsis": "Analysing…",
        "analysis_could_not_finish": "Analysis could not finish",
        "connection_timeout": "The connection may have timed out. Check your network and try again.",
        "upload_failed": "Upload failed. Please try again.",
        "no_result_url": "No result URL returned from the server.",
        "camera_not_available": "Camera not available: ",
        "could_not_validate": "Could not validate image.",
        "please_choose_image": "Please choose an image file.",
        "only_jpg_png_webp": "Only JPG, PNG, and WEBP images are accepted.",
        "invalid_form": "Invalid form data.",
        "please_upload_valid": "Please upload a valid image file.",
        "no_image_provided": "No image provided.",
        "pesticide_disclaimer": (
            "Use only FPA-registered products for rice. Follow the label, PHI (pre-harvest interval), "
            "and advice from your municipal agriculturist (MAO). Rates below are typical references — "
            "always use the label on your purchased product."
        ),
        "rec_not_rice": (
            "This does not appear to be a rice leaf. RiceGuard is trained only on narrow "
            "rice leaf images from the UCI Rice Leaf Diseases dataset.\n\n"
            "Please upload a clear photo of a rice leaf (long, grass-like blade) taken "
            "against a plain background. Broad or heart-shaped leaves from other plants "
            "cannot be diagnosed by this model."
        ),
        "rec_wrong_shape": (
            "The main green region in this photo does not look like a narrow rice leaf blade "
            "(long and thin). RiceGuard expects a single elongated rice leaf, not a round blob, "
            "broad leaf, or scattered green patches.\n\n"
            "Please upload a clear photo of one rice leaf blade against a plain background."
        ),
        "rec_low_confidence": (
            "The model could not confidently match this image to any class in its training "
            "database. No disease diagnosis is shown.\n\n"
            "Try again with:\n"
            "• A closer, well-lit photo of a single rice leaf\n"
            "• Minimal background clutter\n"
            "• The full leaf visible and in focus\n\n"
            "If the leaf is not rice, this tool cannot classify it."
        ),
        "rec_unknown": (
            "Unable to classify this image. It may not be a rice leaf, or the photo quality "
            "may be too low for a reliable match. Please upload a clear rice leaf image."
        ),
        "rec_default": (
            "Disease detected but no specific recommendation is available for '{disease}'. "
            "Please consult your local agricultural extension officer or plant pathologist "
            "for tailored advice. In the meantime, avoid spreading material from this plant "
            "to healthy areas and monitor nearby plants closely."
        ),
        "alt_leaf_scan": "Leaf scan",
        "alt_uploaded_leaf": "Uploaded leaf photo",
        "alt_original_upload": "Original upload",
        "alt_best_training_match": "Best matching training sample for {disease}",
        "alt_reference_sample": "Reference: {disease}",
        "history_pagination": "History pagination",
        "week_of": "Week of {date}",
        "unknown_place": "Unknown",
        "no_image_api": "No image provided. Send 'image' as multipart/form-data.",
        "form_image_label": "Leaf Image",
        "form_image_help": "Upload a clear photo of the rice leaf (JPG, PNG, WEBP — max 10 MB).",
        "processing_error_generic": "Analysis failed. Please try again with a clearer rice leaf photo.",
    },
    "fil": {
        "html_lang": "fil",
        "home": "Unang Pahina",
        "history": "Talaan",
        "locations": "Mga Lokasyon",
        "api": "API",
        "admin": "Admin",
        "lang_label": "Wika",
        "lang_dialect_label": "Diyalekto ng Filipino",
        "lang_en": "English",
        "lang_fil": "Filipino",
        "lang_tag": "Tagalog",
        "lang_ilo": "Ilokano",
        "hero_title": "Detektor ng Sakit sa Dahon ng Palay",
        "hero_tagline": (
            "Mag-upload ng larawan ng dahon ng palay — tutukuyin ng 3-layer AI pipeline "
            "ang sakit at magbibigay ng rekomendasyon sa loob ng ilang segundo."
        ),
        "upload_title": "Mag-upload ng Larawan ng Dahon",
        "upload_help": (
            "Pagkatapos mong pumili ng larawan, tinitingnan ng Layer 0 kung dahon ng palay ito. "
            "Saka lang magagawa ang pagsusuri sa orihinal na larawan."
        ),
        "accepted": "Tinatanggap:",
        "accepted_hint": "JPG, PNG, o WEBP · hanggang 10 MB · malinaw na larawan ng dahon ng palay",
        "drop_title": "I-drag at i-drop ang larawan dito",
        "drop_sub": "o i-click para pumili / gamitin ang camera sa ibaba",
        "choose_image": "Pumili ng larawan",
        "file_types": "JPG, PNG, WEBP — hanggang 10 MB",
        "selected_image": "Piniling larawan",
        "remove_photo": "Tanggalin",
        "checking_image": "Tinitingnan ang larawan…",
        "choose_different": "Pumili ng ibang larawan",
        "analyse_leaf": "Suriin ang Dahon",
        "use_camera": "Gamitin ang Camera",
        "capture": "Kunan",
        "camera_hint": "Huwag gumalaw — igitna ang dahon at panatilihing naka-focus",
        "capturing": "Kinukuha…",
        "close_camera": "Isara ang camera",
        "analysing": "Sinusuri ang dahon… maaaring tumagal ng ilang segundo.",
        "recent_scans": "Mga Kamakailang Scan",
        "ago": "ang nakalipas",
        "view_all_history": "Tingnan ang Buong Talaan",
        "view_all_locations": "Tingnan ang Lahat ng Lokasyon",
        "no_scans_yet": "Wala pang scan. Mag-upload ng unang dahon!",
        "how_it_works": "Paano ito gumagana",
        "howit_1_title": "Mag-upload ng larawan ng dahon",
        "howit_1_desc": (
            "Mag-upload ng malinaw na larawan ng dahon ng palay — tinitingnan muna ito ng Layer 0 "
            "bago tumakbo ang AI."
        ),
        "howit_2_title": "Sinusuri ito ng AI pipeline",
        "howit_2_desc": (
            "Tinitiyak ng Layer 0 ang larawan, pagkatapos ay tinutukoy ng tatlong Random Forest layer "
            "ang kalusugan, ang sakit, at ang kumpiyansa ng resulta."
        ),
        "howit_3_title": "Makakuha ng rekomendasyon",
        "howit_3_desc": (
            "Makakatanggap ka ng malinaw na gabay sa paggamot ayon sa nakitang sakit."
        ),
        "scan_history": "Talaan ng mga Scan",
        "new_scan": "Bagong Scan",
        "total_scans": "Kabuuang Scan",
        "health_healthy": "Malusog",
        "health_unhealthy": "May Sakit",
        "health_unknown": "Hindi Natukoy",
        "health_not_rice": "Hindi Dahon ng Palay",
        "not_rice_note": (
            "Mukhang hindi dahon ng palay ang larawang ito, kaya walang ginawang pagsusuri "
            "ng sakit. Mag-upload ng malinaw na larawan ng iisang dahon ng palay."
        ),
        "filter": "Salain:",
        "filter_all": "Lahat",
        "showing_scans": "Ipinapakita ang {shown} sa {total} na scan",
        "no_scans_found": "Walang nakitang scan.",
        "no_scans_found_filter": "Walang nakitang scan para sa filter na ito.",
        "previous": "Nakaraan",
        "next": "Susunod",
        "page_of": "Pahina {current} ng {total}",
        "upload_first_leaf": "Mag-upload ng Unang Dahon",
        "scan_not_found": "Hindi nahanap ang scan",
        "scan_not_found_msg": (
            "Hindi na available ang Scan #{id}. Maaaring natanggal na ito, o luma na ang link."
        ),
        "scan_new_leaf": "Mag-scan ng bagong dahon",
        "view_history": "Tingnan ang talaan",
        "print": "I-print",
        "scan_report": "Pagtukoy ng Sakit sa Dahon ng Palay — Ulat ng Scan",
        "scan_number": "Scan #",
        "date": "Petsa:",
        "printed": "Na-print:",
        "analysis_result": "Resulta ng pagsusuri",
        "processing_warning": "Babala sa pagproseso",
        "leaf_check_warning": "Babala sa pagsusuri ng dahon",
        "your_leaf_photo": "Larawan ng iyong dahon",
        "tap_to_enlarge": "I-tap para palakihin",
        "close_photo_preview": "Isara ang larawan",
        "analysed_from_upload": "Sinuri mula sa orihinal mong upload.",
        "did_not_pass_leaf_check": (
            "Hindi pumasa ang larawang ito sa pagsusuri ng dahon at hindi ito sinuri."
        ),
        "at_a_glance": "Sa madaling sabi",
        "result": "Resulta",
        "disease": "Sakit",
        "scan_time": "Oras ng scan",
        "location": "Lokasyon",
        "open_map": "Buksan ang mapa",
        "location_captured": "Nakuha ang lokasyon ng kuha",
        "location_unavailable": "Hindi nakuha",
        "finding_location": "Hinahanap ang lokasyon…",
        "location_denied": "Naka-off ang pahintulot sa lokasyon",
        "allow_location_hint": "Payagan ang lokasyon para makita kung saan kinunan ang dahon.",
        "use_my_location": "Gamitin ang lokasyon ko",
        "getting_location": "Kinukuha ang lokasyon ng device…",
        "locations_title": "Mga Nakuhaang Lokasyon",
        "locations_subtitle": "Mga lugar kung saan kinuha ang scan ng dahon, ayon sa kung gaano kadalas silang nakuha.",
        "most_captured": "Pinakamadalas kunan",
        "with_location": "May lokasyon",
        "without_location": "Hindi nakuha",
        "unique_places": "Iba't ibang lugar",
        "scans_here": "mga scan",
        "last_scan": "Huling scan",
        "view_scans": "Tingnan ang mga scan",
        "no_locations_yet": "Wala pang nakuhaang lokasyon. Mag-scan ng dahon na may lokasyon para makita ang mga bukid.",
        "common_diseases": "Mga karaniwang sakit",
        "recent_located": "Mga kamakailang scan na may lokasyon",
        "location_map": "Mapa ng mga lokasyon",
        "capture_zone": "Sona ng pagkuha",
        "exact_captures": "Eksaktong mga punto ng pagkuha",
        "barangay_chart_title": "Mga barangay na may pinakamaraming scan",
        "barangay_chart_subtitle": "Dami ng scan bawat barangay, kasama ang malusog vs may sakit at pangunahing sakit.",
        "timeline_chart_title": "Timeline ng scan at sakit",
        "timeline_chart_subtitle": "Kailan tumataas ang mga scan, at kailan may nadedetect na may sakit.",
        "timeline_scans": "Kabuuang scan",
        "timeline_diseased": "Mga scan na may sakit",
        "timeline_healthy": "Mga malusog na scan",
        "timeline_cum_scans": "Pinagsama-samang scan",
        "timeline_cum_diseased": "Pinagsama-samang may sakit",
        "timeline_peak_scans": "Pinakamataas na scan",
        "timeline_peak_diseased": "Pinakamataas na may sakit",
        "top_disease_label": "Pangunahing sakit",
        "no_disease_short": "Walang sakit",
        "photo_gps": "Mula sa GPS ng larawan",
        "device_gps": "Mula sa GPS ng device",
        "all_places": "Lahat ng lugar",
        "scans_at_place": "Mga scan sa lugar na ito",
        "diseased_share": "Bahaging may sakit",
        "capture_source": "Paano nakuha ang lokasyon",
        "top_disease": "Pinakamadalas na sakit",
        "no_disease_yet": "Wala pang naitalang sakit",
        "place_rank": "Ranggo",
        "na": "Wala",
        "what_this_means": "Ano ang ibig sabihin nito",
        "what_to_do_next": "Ano ang susunod na gawin",
        "recommendation": "Rekomendasyon",
        "immediate_actions": "Mga agarang hakbang",
        "complete_procedure": "Buong paraan ng paggamot",
        "tip": "Payo:",
        "more_guidance": "Karagdagang gabay",
        "symptoms": "Mga sintomas para kumpirmahin",
        "cultural": "Pamamahala sa bukid",
        "pesticide_options": "Mga opsyon sa pestisidyo",
        "spray_guidelines": "Gabay sa pag-spray",
        "prevention": "Pag-iwas",
        "resistant_varieties": "Mga lumalaban na barayti",
        "based_on": "Batay sa",
        "confirm_products": "Kumpirmahin ang mga produkto sa iyong MAO o teknolohista ng PhilRice.",
        "reference_samples": "Mga sample na sanggunian",
        "best_training_match": "Pinakamalapit na sample sa training",
        "similarity": "Pagkakatulad:",
        "best_match_note": "Ang larawang ito ay kinuha mula sa parehong klase ng iyong resulta.",
        "more_sample_photos": "Higit pang sample na larawan para ikumpara",
        "scan_another": "Mag-scan ng isa pang dahon",
        "print_result": "I-print ang resulta",
        "print_footer": (
            "Ginawa ng {brand} — Rekomendasyong tinulungan ng AI. Kumpirmahin sa lokal na "
            "tanggapan ng agrikultura (MAO) o teknolohista ng PhilRice bago maglagay ng pestisidyo."
        ),
        "wet_or_dry": "Basa o tuyong palayan?",
        "field_mode_note": (
            "Piliin ang paraan ayon sa tubig sa bukid. Kailangan ng foliar spray ang "
            "<strong>tuyong dahon</strong>; mas epektibo ang granules sa <strong>nakatayong tubig</strong>."
        ),
        "how_to_apply": "Paano maglagay nang tama",
        "fpa_products": "Mga produktong rehistrado sa FPA",
        "numbered_boxes_note": (
            "Bawat naka-numerong kahon ay <strong>ibang produkto</strong>. Pumili ng available "
            "sa inyong lugar. Huwag ihalo maliban kung pinapayagan ng label."
        ),
        "next_pesticide": "Susunod na pestisidyo",
        "pesticide_n_of": "Pestisidyo {n} sa {total}",
        "active": "Aktibong sangkap:",
        "best_wet": "Pinakamainam sa basang bukid",
        "spray_dry": "I-spray sa tuyong dahon",
        "before_sowing": "Bago maghasik",
        "when": "Kailan",
        "rate": "Dosis",
        "wet_field": "Basang bukid (basa):",
        "dry_field": "Tuyong bukid (tuyo):",
        "show_mixing": "Ipakita ang hakbang sa paghalo / pag-spray ng produktong ito",
        "gate_warning": (
            "Hindi lubos na sigurado ang pagsusuri ng dahon ng palay sa larawang ito "
            "(dahilan: {reason}). Tumuloy pa rin ang pagsusuri sa ibaba, ngunit ingatan "
            "ang resulta — kumuha ulit ng mas malapit, maliwanag na larawan ng isang dahon "
            "kung mukhang mali ito."
        ),
        "layer0_warning": (
            "Maaaring hindi malinaw na dahon ng palay ang larawang ito. Maaari mo pa rin itong "
            "suriin, ngunit maaaring mas hindi sigurado ang resulta — mas mainam ang malapit, "
            "maliwanag na larawan ng iisang dahon."
        ),
        "unsupported_format_title": "Hindi suportadong format",
        "unsupported_format_msg": (
            "Hindi magagamit ang uri ng file na ito. Mag-upload ng larawang naka-save bilang JPG, PNG, o WEBP."
        ),
        "file_too_large_title": "Masyadong malaki ang file",
        "file_too_large_msg": (
            "Dapat 10 MB o mas maliit ang larawan. Subukan ang mas maliit na litrato o ibaba ang quality ng camera."
        ),
        "file_too_large_server": "Ang larawang ito ay {size} MB. Ang pinakamalaking sukat ay 10 MB.",
        "not_rice_leaf_title": "Hindi dahon ng palay",
        "not_rice_leaf_msg": (
            "Hindi mukhang dahon ng palay ang larawang ito. Mag-upload ng malinaw na larawan ng iisang dahon ng palay."
        ),
        "could_not_check_title": "Hindi masuri ang larawan",
        "could_not_check_msg": "May naganap na problema sa pagsusuri ng iyong litrato. Subukan ulit.",
        "upload_problem": "Problema sa pag-upload",
        "cannot_analyse": "Hindi masuri ang larawang ito.",
        "not_passed_check": (
            "Hindi pa pumasa ang larawang ito sa pagsusuri ng dahon ng palay. Pumili ng malinaw na larawan ng dahon."
        ),
        "choose_image_first": "Pumili muna ng larawan bago suriin.",
        "analysing_ellipsis": "Sinusuri…",
        "analysis_could_not_finish": "Hindi natapos ang pagsusuri",
        "connection_timeout": "Maaaring na-timeout ang koneksyon. Tingnan ang network at subukan ulit.",
        "upload_failed": "Hindi natuloy ang pag-upload. Subukan ulit.",
        "no_result_url": "Walang resultang URL na ibinalik ng server.",
        "camera_not_available": "Hindi available ang camera: ",
        "could_not_validate": "Hindi ma-validate ang larawan.",
        "please_choose_image": "Pumili ng file na larawan.",
        "only_jpg_png_webp": "JPG, PNG, at WEBP lamang ang tinatanggap.",
        "invalid_form": "Hindi wasto ang datos ng form.",
        "please_upload_valid": "Mag-upload ng wastong file na larawan.",
        "no_image_provided": "Walang ibinigay na larawan.",
        "pesticide_disclaimer": (
            "Gumamit lamang ng produktong rehistrado sa FPA para sa palay. Sundin ang label, PHI "
            "(pre-harvest interval), at payo ng municipal agriculturist (MAO). Ang mga dosis sa ibaba "
            "ay karaniwang sanggunian — laging sundin ang label ng binili mong produkto."
        ),
        "rec_not_rice": (
            "Mukhang hindi ito dahon ng palay. Sinanay ang RiceGuard sa makitid na dahon ng palay "
            "mula sa UCI Rice Leaf Diseases dataset.\n\n"
            "Mag-upload ng malinaw na larawan ng dahon ng palay (mahaba, parang damo) na kuha "
            "sa simpleng background. Hindi masusuri ng modelong ito ang malapad o hugis-pusong dahon "
            "mula sa ibang halaman."
        ),
        "rec_wrong_shape": (
            "Ang berdeng bahagi sa larawan ay hindi mukhang makitid na dahon ng palay "
            "(mahaba at manipis). Inaasahan ng RiceGuard ang iisang pahabang dahon ng palay, "
            "hindi bilog, malapad, o magkakahiwalay na berdeng bahagi.\n\n"
            "Mag-upload ng malinaw na larawan ng iisang dahon ng palay sa simpleng background."
        ),
        "rec_low_confidence": (
            "Hindi sigurado ang modelo na itugma ang larawang ito sa anumang klase sa training "
            "database. Walang diagnosis ng sakit na ipinapakita.\n\n"
            "Subukan ulit gamit ang:\n"
            "• Mas malapit at maliwanag na larawan ng iisang dahon ng palay\n"
            "• Kaunting kalat sa background\n"
            "• Buong dahon na kita at naka-focus\n\n"
            "Kung hindi palay ang dahon, hindi ito makikilala ng tool na ito."
        ),
        "rec_unknown": (
            "Hindi makilala ang larawang ito. Maaaring hindi ito dahon ng palay, o masyadong "
            "mababa ang kalidad ng litrato. Mag-upload ng malinaw na larawan ng dahon ng palay."
        ),
        "rec_default": (
            "May nakitang sakit ngunit walang tiyak na rekomendasyon para sa '{disease}'. "
            "Kumonsulta sa lokal na agricultural extension officer o plant pathologist. "
            "Sa ngayon, iwasang ikalat ang materyal mula sa halaman na ito sa malulusog na "
            "bahagi at bantayan ang mga kalapit na halaman."
        ),
        "alt_leaf_scan": "Scan ng dahon",
        "alt_uploaded_leaf": "Na-upload na larawan ng dahon",
        "alt_original_upload": "Orihinal na upload",
        "alt_best_training_match": "Pinakamalapit na sample sa training para sa {disease}",
        "alt_reference_sample": "Sanggunian: {disease}",
        "history_pagination": "Paginasyon ng talaan",
        "week_of": "Linggo ng {date}",
        "unknown_place": "Hindi alam",
        "no_image_api": "Walang ibinigay na larawan. Ipadala ang 'image' bilang multipart/form-data.",
        "form_image_label": "Larawan ng Dahon",
        "form_image_help": "Mag-upload ng malinaw na larawan ng dahon ng palay (JPG, PNG, WEBP — hanggang 10 MB).",
        "processing_error_generic": "Hindi natapos ang pagsusuri. Subukan ulit gamit ang mas malinaw na larawan ng dahon ng palay.",
    },
}

# Ilocano UI — based on Tagalog copy, rewritten for Ilocano speakers.
_ILO_OVERRIDES = {
    "html_lang": "ilo",
    "home": "Umuna a Panid",
    "history": "Listaan",
    "locations": "Dagiti Lokasion",
    "lang_label": "Pagsasao",
    "lang_dialect_label": "Dialekto ti Filipino",
    "lang_en": "English",
    "lang_fil": "Filipino",
    "lang_tag": "Tagalog",
    "lang_ilo": "Ilokano",
    "hero_title": "Detektor ti Sakit iti Bulong ti Pagay",
    "hero_tagline": (
        "I-upload ti ladawan ti bulong ti pagay — ti 3-layer AI pipeline ket "
        "mangilasin iti sakit ken mangted rekomendasion iti bassit laeng a segundo."
    ),
    "upload_title": "I-upload ti Ladawan ti Bulong",
    "upload_help": (
        "Kalpasan a piliem ti ladawan, kitaen ti Layer 0 no bulong ti pagay daytoy. "
        "Kalpasanna laeng a mabalin ti panagusig iti orihinal a ladawan."
    ),
    "accepted": "Maawat:",
    "accepted_hint": "JPG, PNG, wenno WEBP · agingga 10 MB · nalawag a ladawan ti bulong ti pagay",
    "drop_title": "I-drag ken i-drop ti ladawan ditoy",
    "drop_sub": "wenno i-click tapno agpili / usaren ti camera iti baba",
    "choose_image": "Agpili iti ladawan",
    "file_types": "JPG, PNG, WEBP — agingga 10 MB",
    "selected_image": "Napili a ladawan",
    "remove_photo": "Ikkatem",
    "checking_image": "Kitaen ti ladawan…",
    "choose_different": "Agpili iti sabali a ladawan",
    "analyse_leaf": "Usigen ti Bulong",
    "use_camera": "Usaren ti Camera",
    "capture": "Alaen",
    "camera_hint": "Agtalna — itengnga ti bulong ken ipanunot a naka-focus",
    "capturing": "Agal-ala…",
    "close_camera": "Serraan ti camera",
    "analysing": "Us-usigen ti bulong… mabalin a makastrek iti bassit a segundo.",
    "recent_scans": "Dagiti Kaudian a Scan",
    "ago": "ti napalabas",
    "view_all_history": "Kitaen ti Amin a Listaan",
    "view_all_locations": "Kitaen amin a Lokasion",
    "no_scans_yet": "Awan pay ti scan. I-upload ti umuna a bulong!",
    "how_it_works": "Kasano a agtrabaho",
    "howit_1_title": "I-upload ti ladawan ti bulong",
    "howit_1_desc": (
        "I-upload ti nalawag a ladawan ti bulong ti pagay — kitaen nga umuna ti Layer 0 "
        "sakbay a tumaray ti AI."
    ),
    "howit_2_title": "Usigen daytoy ti AI pipeline",
    "howit_2_desc": (
        "Patien ti Layer 0 ti ladawan, kalpasanna ti tallo a Random Forest layer "
        "mangilasin iti salun-at, ti sakit, ken ti panagtalek iti resulta."
    ),
    "howit_3_title": "Mangala iti rekomendasion",
    "howit_3_desc": (
        "Umawatka iti nalawag a panangiwaragawag iti pananggagamod segun iti nasarakan a sakit."
    ),
    "scan_history": "Listaan dagiti Scan",
    "new_scan": f"{ilocano_word('bago', 'baro').capitalize()} a Scan",
    "total_scans": "Dagup a Scan",
    "health_healthy": "Nasalun-at",
    "health_unhealthy": "Adda Sakit",
    "health_unknown": "Saan a Nailasin",
    "health_not_rice": "Saan a Bulong ti Pagay",
    "not_rice_note": (
        "Kasla saan a bulong ti pagay daytoy a ladawan, isu nga awan ti naaramid a "
        "panagusig iti sakit. I-upload ti nalawag a ladawan ti maymaysa a bulong ti pagay."
    ),
    "filter": "Sagat:",
    "filter_all": "Amin",
    "showing_scans": "Ipabpakita ti {shown} kadagiti {total} a scan",
    "no_scans_found": "Awan ti nasarakan a scan.",
    "no_scans_found_filter": "Awan ti nasarakan a scan para iti daytoy a sagat.",
    "previous": "Napalabas",
    "next": "Sumaruno",
    "page_of": "Pahina {current} ti {total}",
    "upload_first_leaf": "I-upload ti Umuna a Bulong",
    "scan_not_found": "Saan a nasarakan ti scan",
    "scan_not_found_msg": (
        "Saanen a available ti Scan #{id}. Mabalin a naikkat daytoy, wenno daan ti link."
    ),
    "scan_new_leaf": "I-scan ti baro a bulong",
    "view_history": "Kitaen ti listaan",
    "print": "I-print",
    "scan_report": "Panangilasin ti Sakit iti Bulong ti Pagay — Report ti Scan",
    "scan_number": "Scan #",
    "date": "Petsa:",
    "printed": "Na-print:",
    "analysis_result": "Resulta ti panagusig",
    "processing_warning": "Ballaag iti panangproseso",
    "leaf_check_warning": "Ballaag iti panangkita ti bulong",
    "your_leaf_photo": "Ladawan ti bulongmo",
    "tap_to_enlarge": "I-tap tapno palakken",
    "close_photo_preview": "Irikep ti ladawan",
    "analysed_from_upload": "Nausig manipud iti orihinal a na-uploadmo.",
    "did_not_pass_leaf_check": (
        "Saan a nakaipasa daytoy a ladawan iti panangkita ti bulong ket saan a nausig."
    ),
    "at_a_glance": "Iti ababa",
    "result": "Resulta",
    "disease": "Sakit",
    "scan_time": "Oras ti scan",
    "location": "Lokasion",
    "open_map": "Lukatan ti mapa",
    "location_captured": "Nalaan ti lokasion ti kuha",
    "location_unavailable": "Saan a nalaan",
    "finding_location": "Birbiruken ti lokasion…",
    "location_denied": "Naka-off ti pammalubos iti lokasion",
    "allow_location_hint": "Palubosan ti lokasion tapno makita no sadino a naikua ti bulong.",
    "use_my_location": "Usaren ti lokasionko",
    "getting_location": "Alaen ti lokasion ti device…",
    "locations_title": "Dagiti Nalaan a Lokasion",
    "locations_subtitle": "Dagiti lugar a nakuha ti scan ti bulong, segun no kasano kadalasda a nakuha.",
    "most_captured": "Kapinakanakuaan",
    "with_location": "Adda lokasion",
    "without_location": "Saan a nalaan",
    "unique_places": "Nadumaduma a lugar",
    "scans_here": "dagiti scan",
    "last_scan": "Maudi a scan",
    "view_scans": "Kitaen dagiti scan",
    "no_locations_yet": "Awan pay ti nalaan a lokasion. I-scan ti bulong nga adda lokasion tapno makita dagiti talon.",
    "common_diseases": "Dagiti kadawyan a sakit",
    "recent_located": "Dagiti kaudian a scan nga adda lokasion",
    "location_map": "Mapa dagiti lokasion",
    "capture_zone": "Sona ti panagkuwa",
    "exact_captures": "Eksakto a puntos ti panagkuwa",
    "barangay_chart_title": "Dagiti barangay nga adu ti scan",
    "barangay_chart_subtitle": "Bilang ti scan kada barangay, agraman nasalun-at vs adda sakit ken kangrunaan a sakit.",
    "timeline_chart_title": "Timeline ti scan ken sakit",
    "timeline_chart_subtitle": "No kaano a dumakkel ti scan, ken no kaano a madetect ti adda sakit.",
    "timeline_scans": "Dagup a scan",
    "timeline_diseased": "Dagiti scan nga adda sakit",
    "timeline_healthy": "Dagiti nasalun-at a scan",
    "timeline_cum_scans": "Naipon a scan",
    "timeline_cum_diseased": "Naipon nga adda sakit",
    "timeline_peak_scans": "Kangatoan a scan",
    "timeline_peak_diseased": "Kangatoan nga adda sakit",
    "top_disease_label": "Kangrunaan a sakit",
    "no_disease_short": "Awan sakit",
    "photo_gps": "Manipud GPS ti ladawan",
    "device_gps": "Manipud GPS ti device",
    "all_places": "Amin a lugar",
    "scans_at_place": "Dagiti scan iti daytoy a lugar",
    "diseased_share": "Paset nga adda sakit",
    "capture_source": "Kasano a nalaan ti lokasion",
    "top_disease": "Kapinakanakit a sakit",
    "no_disease_yet": "Awan pay ti naitala a sakit",
    "place_rank": "Ranggo",
    "na": "Awan",
    "what_this_means": "Ania ti kaipapananna",
    "what_to_do_next": "Ania ti sumaruno a maaramid",
    "recommendation": "Rekomendasion",
    "immediate_actions": "Dagiti dagus a tarken",
    "complete_procedure": "Kompleto a wagas ti pananggagamod",
    "tip": "Balakad:",
    "more_guidance": "Ad-adu a panangiwaragawag",
    "symptoms": "Dagiti sintomas tapno makonfirma",
    "cultural": "Panangimaton iti talon",
    "pesticide_options": "Dagiti opsion ti pestisidio",
    "spray_guidelines": "Panangiwaragawag iti panagspray",
    "prevention": "Panangilag",
    "resistant_varieties": "Dagiti makasagrap a bariedad",
    "based_on": "Naibatay iti",
    "confirm_products": "Konfirmaen dagiti produkto iti MAO wenno teknolohista ti PhilRice.",
    "reference_samples": "Dagiti sample a reperensia",
    "best_training_match": "Kasikasingpet a sample iti training",
    "similarity": "Panagpapada:",
    "best_match_note": "Nalaan daytoy a ladawan manipud iti isu met la a klase ti resultam.",
    "more_sample_photos": "Ad-adu a sample a ladawan tapno maidilig",
    "scan_another": "I-scan ti sabali pay a bulong",
    "print_result": "I-print ti resulta",
    "print_footer": (
        "Inaramid ti {brand} — Rekomendasion a tinulongan ti AI. Konfirmaen iti lokal "
        "nga opisina ti agrikultura (MAO) wenno teknolohista ti PhilRice sakbay nga agikabil iti pestisidio."
    ),
    "wet_or_dry": "Nabasa wenno namaga a pagayan?",
    "field_mode_note": (
        "Pilien ti wagas segun iti danum iti talon. Masapul ti foliar spray ti "
        "<strong>namaga a bulong</strong>; nasaysayaat dagiti granule no adda "
        "<strong>agtaktakder a danum</strong>."
    ),
    "how_to_apply": "Kasano nga umiso nga ikabil",
    "fpa_products": "Dagiti produkto a nairihistro iti FPA",
    "numbered_boxes_note": (
        "Ti tunggal numero a kahon ket <strong>sabali a produkto</strong>. Pilien ti available "
        "iti lugaryo. Saan nga ihalo malaksid no mabalin iti label."
    ),
    "next_pesticide": "Sumaruno a pestisidio",
    "pesticide_n_of": "Pestisidio {n} iti {total}",
    "active": "Aktibo a sangkap:",
    "best_wet": "Kasayaatan iti nabasa a talon",
    "spray_dry": "I-spray iti namaga a bulong",
    "before_sowing": "Sakbay nga aghasik",
    "when": ilocano_word("kailan", "kaano").capitalize(),
    "rate": "Dosis",
    "wet_field": "Nabasa a talon (nabasa):",
    "dry_field": "Namaga a talon (namaga):",
    "show_mixing": "Ipakita ti addang iti pananghalo / panagspray daytoy a produkto",
    "gate_warning": (
        "Saan a napudno a sigurado ti panangkita ti bulong ti pagay iti daytoy a ladawan "
        "(gapu: {reason}). Nagtultuloy latta ti panagusig iti baba, ngem agannadka "
        "iti resulta — agala manen iti asideg, nalawag a ladawan ti maysa a bulong "
        "no kasla biddut daytoy."
    ),
    "layer0_warning": (
        "Mabalin a saan a nalawag a bulong ti pagay daytoy a ladawan. Mabalinmo latta nga "
        "usigen, ngem mabalin a saan a nasigurado ti resulta — nasaysayaat ti asideg, "
        "nalawag a ladawan ti maymaysa a bulong."
    ),
    "unsupported_format_title": "Saan a suportado a format",
    "unsupported_format_msg": (
        "Saan a mausar daytoy a kita ti file. I-upload ti ladawan a na-save a kas JPG, PNG, wenno WEBP."
    ),
    "file_too_large_title": "Nakarukod a dakkel ti file",
    "file_too_large_msg": (
        "Masapul a 10 MB wenno bassit ti ladawan. Padasem ti bassit a ladawan wenno babaen ti quality ti camera."
    ),
    "file_too_large_server": "Daytoy a ladawan ket {size} MB. Ti kadakkelan a sukat ket 10 MB.",
    "not_rice_leaf_title": "Saan a bulong ti pagay",
    "not_rice_leaf_msg": (
        "Saan a kasla bulong ti pagay daytoy a ladawan. I-upload ti nalawag a ladawan ti maymaysa a bulong ti pagay."
    ),
    "could_not_check_title": "Saan a mausig ti ladawan",
    "could_not_check_msg": "Adda problema a napasamak iti panagusig ti ladawanmo. Padasem manen.",
    "upload_problem": "Problema iti panang-upload",
    "cannot_analyse": "Saan a mausig daytoy a ladawan.",
    "not_passed_check": (
        "Saan pay a nakaipasa daytoy a ladawan iti panangkita ti bulong ti pagay. Agpili iti nalawag a ladawan ti bulong."
    ),
    "choose_image_first": "Agpili nga umuna iti ladawan sakbay nga agusig.",
    "analysing_ellipsis": "Us-usigen…",
    "analysis_could_not_finish": "Saan a nalpas ti panagusig",
    "connection_timeout": "Mabalin a na-timeout ti koneksion. Kitaen ti network ken padasem manen.",
    "upload_failed": "Saan a nagballigi ti panang-upload. Padasem manen.",
    "no_result_url": "Awan ti result a URL a naisubli ti server.",
    "camera_not_available": "Saan a available ti camera: ",
    "could_not_validate": "Saan a ma-validate ti ladawan.",
    "please_choose_image": "Agpili iti file a ladawan.",
    "only_jpg_png_webp": "JPG, PNG, ken WEBP laeng ti maawat.",
    "invalid_form": "Saan a umiso ti datos ti form.",
    "please_upload_valid": "I-upload ti umiso a file a ladawan.",
    "no_image_provided": "Awan ti naited a ladawan.",
    "pesticide_disclaimer": (
        "Usaren laeng ti produkto a nairihistro iti FPA para iti pagay. Suruten ti label, PHI "
        "(pre-harvest interval), ken balakad ti municipal agriculturist (MAO). Dagiti dosis iti baba "
        "ket kadawyan a reperensia — kanayon a suruten ti label ti ginatangmo a produkto."
    ),
    "rec_not_rice": (
        "Kasla saan daytoy a bulong ti pagay. Nasanay ti RiceGuard kadagiti naingpis a bulong ti pagay "
        "manipud iti UCI Rice Leaf Diseases dataset.\n\n"
        "I-upload ti nalawag a ladawan ti bulong ti pagay (atiddog, kasla ruot) a naikua "
        "iti simple a background. Saan a mausig daytoy a modelo ti nalawa wenno heart-shaped a bulong "
        "manipud iti sabali a mula."
    ),
    "rec_wrong_shape": (
        "Ti berde a paset iti ladawan ket saan a kasla naingpis a bulong ti pagay "
        "(atiddog ken napinpintog). Namnamaen ti RiceGuard ti maymaysa nga atiddog a bulong ti pagay, "
        "saan a nagtimbukel, nalawa, wenno naisina-sina a berde a paset.\n\n"
        "I-upload ti nalawag a ladawan ti maymaysa a bulong ti pagay iti simple a background."
    ),
    "rec_low_confidence": (
        "Saan a sigurado ti modelo a pagpapadaan daytoy a ladawan iti ania man a klase iti training "
        "database. Awan ti diagnosis ti sakit a maipakita.\n\n"
        "Padasem manen nga usaren ti:\n"
        "• Asideg ken nalawag a ladawan ti maymaysa a bulong ti pagay\n"
        "• Bassit laeng a kalat iti background\n"
        "• Kompleto a bulong a makita ken naka-focus\n\n"
        "No saan a pagay ti bulong, saan a mailasin daytoy a tool."
    ),
    "rec_unknown": (
        "Saan a mailasin daytoy a ladawan. Mabalin a saan daytoy a bulong ti pagay, wenno nakarukod "
        "a baba ti kalidad ti ladawan. I-upload ti nalawag a ladawan ti bulong ti pagay."
    ),
    "rec_default": (
        "Adda nasarakan a sakit ngem awan ti spesifik a rekomendasion para iti '{disease}'. "
        "Konsultaen ti lokal nga agricultural extension officer wenno plant pathologist. "
        "Ita, liklikan ti panangiwaras iti material manipud iti daytoy a mula kadagiti nasalun-at "
        "a paset ken bantayan dagiti kaarruba a mula."
    ),
    "alt_leaf_scan": "Scan ti bulong",
    "alt_uploaded_leaf": "Na-upload a ladawan ti bulong",
    "alt_original_upload": "Orihinal a na-upload",
    "alt_best_training_match": "Kasikasingpet a sample iti training para iti {disease}",
    "alt_reference_sample": "Reperensia: {disease}",
    "history_pagination": "Paginasion ti listaan",
    "week_of": "Lawas ti {date}",
    "unknown_place": "Saan a naammuan",
    "no_image_api": "Awan ti naited a ladawan. Ipatulod ti 'image' a kas multipart/form-data.",
    "form_image_label": "Ladawan ti Bulong",
    "form_image_help": "I-upload ti nalawag a ladawan ti bulong ti pagay (JPG, PNG, WEBP — agingga 10 MB).",
    "processing_error_generic": "Saan a nalpas ti panagusig. Padasem manen nga usaren ti nalawag a ladawan ti bulong ti pagay.",
}

UI["ilo"] = {**UI["fil"], **_ILO_OVERRIDES}


def normalize_lang(value: str | None) -> str:
    if value in SUPPORTED:
        return value
    # Legacy / friendly aliases
    if value in ("tl", "tag", "tagalog"):
        return "fil"
    if value in ("ilocano", "ilokano"):
        return "ilo"
    return DEFAULT_LANG


def is_filipino_lang(lang: str | None) -> bool:
    return normalize_lang(lang) in FILIPINO_LANGS


def get_lang(request: HttpRequest | None) -> str:
    if request is None:
        return DEFAULT_LANG
    cookie = getattr(request, "COOKIES", {}).get(LANG_COOKIE)
    if cookie in SUPPORTED:
        return cookie
    return normalize_lang(getattr(request, "lang", None))


def get_strings(lang: str | None = None) -> dict[str, str]:
    return UI[normalize_lang(lang)]


def js_strings(lang: str) -> dict[str, str]:
    t = get_strings(lang)
    return {key: t[key] for key in JS_KEYS}


def js_strings_json(lang: str) -> str:
    return mark_safe(json.dumps(js_strings(lang), ensure_ascii=False))


def health_label(health: str, lang: str) -> str:
    t = get_strings(lang)
    return {
        "healthy": t["health_healthy"],
        "diseased": t["health_unhealthy"],
        "unknown": t["health_unknown"],
        "not_rice": t["health_not_rice"],
    }.get(health, t["health_unknown"])


def format_gate_warning(stored: str, lang: str) -> str:
    if not stored:
        return ""
    match = _REASON_RE.search(stored)
    reason = match.group(1) if match else "—"
    return get_strings(lang)["gate_warning"].format(reason=reason)


def unknown_recommendation(stored: str, lang: str) -> str:
    t = get_strings(lang)
    text = (stored or "").lower()
    if "does not appear to be a rice leaf" in text or "not_rice" in text:
        return t["rec_not_rice"]
    if "does not look like a narrow rice leaf" in text or "wrong_shape" in text:
        return t["rec_wrong_shape"]
    if "could not confidently" in text or "low_confidence" in text:
        return t["rec_low_confidence"]
    if stored:
        return t["rec_unknown"]
    return t["rec_unknown"]


_MONTH_ABBR = {
    "en": ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
    "fil": ("Ene", "Peb", "Mar", "Abr", "May", "Hun", "Hul", "Ago", "Set", "Okt", "Nob", "Dis"),
    "ilo": ("Ene", "Peb", "Mar", "Abr", "May", "Hun", "Hul", "Ago", "Set", "Okt", "Nob", "Dis"),
}


def format_month_day(dt, lang: str | None = None) -> str:
    """Short month + day label without relying on missing Django locales."""
    from django.utils import timezone as dj_tz

    local = dj_tz.localtime(dt)
    months = _MONTH_ABBR[normalize_lang(lang)]
    return f"{months[local.month - 1]} {local.day:02d}"


def format_timeline_label(dt, grain: str, lang: str | None = None) -> str:
    from django.utils import timezone as dj_tz

    code = normalize_lang(lang)
    local = dj_tz.localtime(dt)
    day = format_month_day(local, code)
    if grain == "hour":
        return f"{day} · {local.strftime('%H:%M')}"
    if grain == "week":
        return get_strings(code)["week_of"].format(date=day)
    return day


def localize_recommendation_text(
    *,
    health: str,
    disease_name: str,
    stored_recommendation: str,
    lang: str,
) -> str:
    """Build display recommendation text for the active UI language."""
    from .disease_guide import get_farmer_guide, localize_farmer_guide

    t = get_strings(lang)
    guide = None
    if health == "diseased" and disease_name:
        guide = localize_farmer_guide(get_farmer_guide(disease_name), lang)
    elif health == "healthy":
        guide = localize_farmer_guide(get_farmer_guide("healthy"), lang)
    if guide:
        return guide.to_recommendation_text()
    if health == "unknown":
        return unknown_recommendation(stored_recommendation, lang)
    if health == "diseased" and disease_name:
        return t["rec_default"].format(disease=disease_name)
    return stored_recommendation or t["rec_unknown"]


def localize_processing_error(stored: str, lang: str) -> str:
    if not stored:
        return ""
    # Prefer a clear localized message; keep technical detail only in English logs/API raw field.
    return get_strings(lang)["processing_error_generic"]


def localize_gate_warning_display(stored: str, lang: str) -> str:
    return format_gate_warning(stored, lang)