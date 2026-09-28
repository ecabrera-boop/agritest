from django.db import models


class SiteSettings(models.Model):
    """Singleton homepage copy editable at /admin/."""

    hero_title = models.CharField(
        max_length=200,
        default="Rice Leaf Disease Detector",
        help_text="Main headline on the home page.",
    )
    hero_tagline = models.TextField(
        default=(
            "Upload a photo of a rice leaf — our 3-layer AI pipeline will identify "
            "any disease and recommend treatment in seconds."
        ),
        help_text="Subtitle shown under the headline.",
    )
    site_brand = models.CharField(
        max_length=80,
        default="RiceGuard",
        help_text="Site name in the navbar and browser tab.",
    )

    class Meta:
        verbose_name = "Site settings"
        verbose_name_plural = "Site settings"

    def __str__(self):
        return self.hero_title

    def save(self, *args, **kwargs):
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def load(cls):
        obj, _ = cls.objects.get_or_create(
            pk=1,
            defaults={
                "hero_title": "Rice Leaf Disease Detector",
                "hero_tagline": (
                    "Upload a photo of a rice leaf — our 3-layer AI pipeline will identify "
                    "any disease and recommend treatment in seconds."
                ),
                "site_brand": "RiceGuard",
            },
        )
        return obj


class LeafScan(models.Model):
    HEALTH_CHOICES = [
        ("healthy", "Healthy"),
        ("diseased", "Unhealthy"),
        ("unknown", "Unknown"),
    ]

    image = models.ImageField(upload_to="leaf_uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    result_health = models.CharField(max_length=20, choices=HEALTH_CHOICES, default="unknown")
    disease_name = models.CharField(max_length=100, blank=True, default="")
    confidence = models.FloatField(default=0.0)
    all_disease_probs = models.JSONField(blank=True, default=dict)
    recommendation = models.TextField(blank=True, default="")
    processing_error = models.TextField(blank=True, default="")
    # Non-empty when Layer 0 flagged the photo but the pipeline analysed it
    # anyway (lenient gate) — shown as a caution banner on the result page.
    gate_warning = models.TextField(blank=True, default="")
    # Why the pipeline returned no verdict: "not_rice", "wrong_shape",
    # "low_confidence", or "" for a normal result.
    rejection_reason = models.CharField(max_length=32, blank=True, default="")
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    location_name = models.CharField(max_length=255, blank=True, default="")
    location_source = models.CharField(max_length=16, blank=True, default="")

    class Meta:
        ordering = ["-uploaded_at"]
        verbose_name = "Leaf Scan"
        verbose_name_plural = "Leaf Scans"

    def __str__(self):
        label = self.disease_name if self.disease_name else self.result_health
        return f"Scan #{self.pk} — {label} ({self.uploaded_at:%Y-%m-%d %H:%M})"

    @property
    def has_location(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def coordinates_display(self) -> str:
        if not self.has_location:
            return ""
        return f"{self.latitude:.5f}, {self.longitude:.5f}"

    @property
    def location_display(self) -> str:
        name = (self.location_name or "").strip()
        if name:
            return name
        return self.coordinates_display

    @property
    def maps_url(self) -> str:
        if not self.has_location:
            return ""
        return (
            f"https://www.openstreetmap.org/?mlat={self.latitude}&mlon={self.longitude}"
            f"#map=17/{self.latitude}/{self.longitude}"
        )

    @property
    def map_embed_url(self) -> str:
        if not self.has_location:
            return ""
        delta = 0.012
        west = self.longitude - delta
        south = self.latitude - delta
        east = self.longitude + delta
        north = self.latitude + delta
        return (
            "https://www.openstreetmap.org/export/embed.html"
            f"?bbox={west},{south},{east},{north}&layer=mapnik"
            f"&marker={self.latitude},{self.longitude}"
        )

    @property
    def confidence_percent(self):
        return round(self.confidence * 100, 1)

    @property
    def is_not_rice(self) -> bool:
        """True when the rice-leaf gate rejected this photo (incl. legacy scans)."""
        reason = (self.rejection_reason or "").strip()
        if reason in ("not_rice", "wrong_shape"):
            return True
        warning = self.gate_warning or ""
        return "not_rice" in warning or "wrong_shape" in warning

    @property
    def status_badge(self):
        if self.is_not_rice:
            return "warning"
        return {
            "healthy": "success",
            "diseased": "danger",
            "unknown": "secondary",
        }.get(self.result_health, "secondary")
