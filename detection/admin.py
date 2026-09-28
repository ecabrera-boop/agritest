from django.contrib import admin
from django.utils.html import format_html
from .models import LeafScan, SiteSettings


@admin.register(SiteSettings)
class SiteSettingsAdmin(admin.ModelAdmin):
    fieldsets = (
        (None, {"fields": ("hero_title", "hero_tagline", "site_brand")}),
    )

    def has_add_permission(self, request):
        return not SiteSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(LeafScan)
class LeafScanAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "thumbnail",
        "result_health",
        "disease_name",
        "confidence_display",
        "location_name",
        "uploaded_at",
    ]
    list_filter = ["result_health", "uploaded_at"]
    search_fields = ["disease_name", "recommendation", "location_name"]
    readonly_fields = ["uploaded_at", "thumbnail_large", "confidence_display"]
    ordering = ["-uploaded_at"]

    def thumbnail(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="height:48px;border-radius:4px;" />',
                obj.image.url,
            )
        return "—"
    thumbnail.short_description = "Image"

    def thumbnail_large(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" style="max-height:300px;border-radius:8px;" />',
                obj.image.url,
            )
        return "—"
    thumbnail_large.short_description = "Preview"

    def confidence_display(self, obj):
        return f"{obj.confidence_percent}%"
    confidence_display.short_description = "Confidence"
