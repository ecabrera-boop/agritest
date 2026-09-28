from rest_framework import serializers

from .i18n import (
    format_gate_warning,
    get_lang,
    localize_processing_error,
    localize_recommendation_text,
)
from .models import LeafScan


class LeafScanSerializer(serializers.ModelSerializer):
    confidence_percent = serializers.ReadOnlyField()
    image_url = serializers.SerializerMethodField()
    location_display = serializers.ReadOnlyField()
    maps_url = serializers.ReadOnlyField()
    recommendation_display = serializers.SerializerMethodField()
    processing_error_display = serializers.SerializerMethodField()
    gate_warning_display = serializers.SerializerMethodField()

    class Meta:
        model = LeafScan
        fields = [
            "id",
            "image",
            "image_url",
            "uploaded_at",
            "result_health",
            "disease_name",
            "confidence",
            "confidence_percent",
            "recommendation",
            "recommendation_display",
            "processing_error",
            "processing_error_display",
            "gate_warning",
            "gate_warning_display",
            "rejection_reason",
            "latitude",
            "longitude",
            "location_name",
            "location_display",
            "maps_url",
        ]
        read_only_fields = [
            "id",
            "uploaded_at",
            "result_health",
            "disease_name",
            "confidence",
            "confidence_percent",
            "recommendation",
            "recommendation_display",
            "processing_error",
            "processing_error_display",
            "gate_warning",
            "gate_warning_display",
            "rejection_reason",
            "location_display",
            "maps_url",
        ]

    def _lang(self) -> str:
        request = self.context.get("request")
        return get_lang(request)

    def get_image_url(self, obj):
        request = self.context.get("request")
        if obj.image and request:
            return request.build_absolute_uri(obj.image.url)
        return None

    def get_recommendation_display(self, obj):
        return localize_recommendation_text(
            health=obj.result_health,
            disease_name=obj.disease_name or "",
            stored_recommendation=obj.recommendation or "",
            lang=self._lang(),
        )

    def get_processing_error_display(self, obj):
        return localize_processing_error(obj.processing_error or "", self._lang())

    def get_gate_warning_display(self, obj):
        return format_gate_warning(obj.gate_warning or "", self._lang())
