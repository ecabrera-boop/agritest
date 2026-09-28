"""Localization completeness and language-routing regression tests."""

from __future__ import annotations

from django.test import RequestFactory, SimpleTestCase, TestCase, override_settings

from detection.disease_guide import (
    APPLICATION_GUIDE,
    APPLICATION_GUIDE_FIL,
    APPLICATION_GUIDE_ILO,
    GUIDES,
    get_application_guide,
    get_farmer_guide,
    localize_farmer_guide,
)
from detection.forms import LeafUploadForm
from detection.i18n import (
    UI,
    format_timeline_label,
    get_strings,
    localize_recommendation_text,
    normalize_lang,
)
from detection.i18n_guides import GUIDE_TEXT_FIL
from detection.i18n_guides_ilo import GUIDE_TEXT_ILO
from detection.i18n_pesticides import PESTICIDE_TEXT_FIL, PESTICIDE_TEXT_ILO
from detection.models import LeafScan
from detection.serializers import LeafScanSerializer


REQUIRED_GUIDE_FIELDS = (
    "name",
    "summary",
    "procedure",
    "farmer_tip",
)


class GuideLocalizationTests(SimpleTestCase):
    def test_guide_key_parity(self):
        self.assertEqual(set(GUIDES), set(GUIDE_TEXT_FIL))
        self.assertEqual(set(GUIDES), set(GUIDE_TEXT_ILO))

    def test_guide_field_shapes(self):
        for key in GUIDES:
            fil_entry = GUIDE_TEXT_FIL[key]
            ilo_entry = GUIDE_TEXT_ILO[key]
            self.assertEqual(set(fil_entry), set(ilo_entry), msg=f"field mismatch for {key}")
            for field in REQUIRED_GUIDE_FIELDS:
                self.assertIn(field, ilo_entry, msg=f"{key}.{field}")
            self.assertIsInstance(ilo_entry["procedure"], list)
            self.assertTrue(ilo_entry["procedure"], msg=f"{key} procedure empty")
            if "symptoms" in ilo_entry:
                self.assertIsInstance(ilo_entry["symptoms"], list)
            for field, value in fil_entry.items():
                if isinstance(value, list):
                    self.assertEqual(
                        len(value),
                        len(ilo_entry[field]),
                        msg=f"{key}.{field} length mismatch",
                    )
    def test_ilocano_differs_from_filipino_and_english(self):
        english = get_farmer_guide("blast")
        fil = localize_farmer_guide(english, "fil")
        ilo = localize_farmer_guide(english, "ilo")
        self.assertNotEqual(ilo.summary, fil.summary)
        self.assertNotEqual(ilo.summary, english.summary)
        self.assertEqual(ilo.name, GUIDE_TEXT_ILO["blast"]["name"])
        self.assertEqual(fil.name, GUIDE_TEXT_FIL["blast"]["name"])

    def test_pesticide_technical_fields_unchanged(self):
        english = get_farmer_guide("blast")
        ilo = localize_farmer_guide(english, "ilo")
        self.assertTrue(english.pesticides)
        for src, loc in zip(english.pesticides, ilo.pesticides):
            self.assertEqual(src.product, loc.product)
            self.assertEqual(src.active_ingredient, loc.active_ingredient)
            self.assertEqual(src.form, loc.form)
            self.assertEqual(src.rate, loc.rate)

    def test_method_labels_and_application_guide_by_lang(self):
        ilo = localize_farmer_guide(get_farmer_guide("blast"), "ilo")
        fil = localize_farmer_guide(get_farmer_guide("blast"), "fil")
        self.assertEqual(ilo.pesticides[0].method_label, "I-spray iti bulong")
        self.assertEqual(fil.pesticides[0].method_label, "Spray sa dahon")
        self.assertEqual(
            get_application_guide("ilo")["field_modes"][0]["title"],
            APPLICATION_GUIDE_ILO["field_modes"][0]["title"],
        )
        self.assertEqual(
            get_application_guide("fil")["field_modes"][0]["title"],
            APPLICATION_GUIDE_FIL["field_modes"][0]["title"],
        )
        self.assertEqual(
            get_application_guide("en")["field_modes"][0]["title"],
            APPLICATION_GUIDE["field_modes"][0]["title"],
        )

    def test_pesticide_timing_note_coverage(self):
        timings = {px.timing for g in GUIDES.values() for px in g.pesticides}
        notes = {px.note for g in GUIDES.values() for px in g.pesticides}
        self.assertEqual(timings, set(PESTICIDE_TEXT_ILO["__timings__"]))
        self.assertEqual(notes, set(PESTICIDE_TEXT_ILO["__notes__"]))
        self.assertEqual(timings, set(PESTICIDE_TEXT_FIL["__timings__"]))
        self.assertEqual(notes, set(PESTICIDE_TEXT_FIL["__notes__"]))

    def test_recommendation_text_ilocano(self):
        text = localize_recommendation_text(
            health="diseased",
            disease_name="blast",
            stored_recommendation="English fallback",
            lang="ilo",
        )
        self.assertIn(GUIDE_TEXT_ILO["blast"]["name"], text)
        self.assertIn("1.", text)
        self.assertIn("2.", text)


class UiStringTests(SimpleTestCase):
    def test_ui_key_parity(self):
        self.assertEqual(set(UI["en"]), set(UI["fil"]))
        self.assertEqual(set(UI["en"]), set(UI["ilo"]))

    def test_normalize_lang_aliases(self):
        self.assertEqual(normalize_lang("ilokano"), "ilo")
        self.assertEqual(normalize_lang("tagalog"), "fil")
        self.assertEqual(normalize_lang("ilo"), "ilo")

    def test_timeline_week_label_ilocano(self):
        from datetime import datetime
        from django.utils import timezone

        dt = timezone.make_aware(datetime(2026, 8, 15, 10, 0, 0))
        label = format_timeline_label(dt, "week", "ilo")
        self.assertTrue(label.startswith(get_strings("ilo")["week_of"].split("{")[0]))

    def test_form_labels_follow_lang(self):
        form = LeafUploadForm(lang="ilo")
        self.assertEqual(form.fields["image"].label, get_strings("ilo")["form_image_label"])


@override_settings(ALLOWED_HOSTS=["testserver"])
class ApiLocalizationTests(TestCase):
    def test_serializer_display_fields_use_request_lang(self):
        scan = LeafScan.objects.create(
            result_health="diseased",
            disease_name="blast",
            confidence=0.9,
            recommendation="English stored recommendation",
            gate_warning="Gate warning reason: blurry",
            processing_error="RuntimeError: boom",
        )
        request = RequestFactory().get("/api/scan/1/")
        request.COOKIES["riceguard_lang"] = "ilo"
        data = LeafScanSerializer(scan, context={"request": request}).data
        self.assertEqual(data["recommendation"], "English stored recommendation")
        self.assertIn(GUIDE_TEXT_ILO["blast"]["name"], data["recommendation_display"])
        self.assertEqual(
            data["processing_error_display"],
            get_strings("ilo")["processing_error_generic"],
        )
        self.assertIn("blurry", data["gate_warning_display"])
        self.assertNotEqual(data["gate_warning_display"], scan.gate_warning)
