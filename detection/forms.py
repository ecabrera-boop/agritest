import os

from django import forms

from .i18n import get_strings

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def validate_leaf_upload(image, lang="en") -> tuple[bool, str, str]:
    """Return (ok, code, message) for an uploaded file-like object."""
    t = get_strings(lang)
    if not image:
        return False, "invalid_format", t["please_choose_image"]
    if image.size > MAX_UPLOAD_BYTES:
        size_mb = image.size / (1024 * 1024)
        return (
            False,
            "file_too_large",
            t["file_too_large_server"].format(size=f"{size_mb:.1f}"),
        )
    ext = os.path.splitext(getattr(image, "name", "") or "")[1].lower()
    content_type = (getattr(image, "content_type", "") or "").lower()
    ext_ok = ext in ALLOWED_EXTENSIONS
    type_ok = content_type in ALLOWED_CONTENT_TYPES
    if not ext_ok and not type_ok:
        return (
            False,
            "invalid_format",
            t["only_jpg_png_webp"],
        )
    return True, "", ""


class LeafUploadForm(forms.Form):
    image = forms.ImageField(
        widget=forms.ClearableFileInput(
            attrs={
                "accept": ".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp",
                "capture": "environment",
            }
        ),
    )

    def __init__(self, *args, lang="en", **kwargs):
        self.lang = lang
        super().__init__(*args, **kwargs)
        t = get_strings(self.lang)
        self.fields["image"].label = t["form_image_label"]
        self.fields["image"].help_text = t["form_image_help"]

    def clean_image(self):
        image = self.cleaned_data.get("image")
        if image:
            ok, code, message = validate_leaf_upload(image, lang=self.lang)
            if not ok:
                raise forms.ValidationError(message)
        return image
