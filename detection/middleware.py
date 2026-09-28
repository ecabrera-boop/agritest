from django.utils import translation

from .i18n import LANG_COOKIE, is_filipino_lang, normalize_lang


class RiceGuardLanguageMiddleware:
    """Read the language cookie and activate Django locale for dates/timesince."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        lang = normalize_lang(request.COOKIES.get(LANG_COOKIE))
        request.lang = lang
        # Django has no built-in ilo locale; use fil for PH date/time formatting.
        django_lang = "fil" if is_filipino_lang(lang) else "en"
        translation.activate(django_lang)
        request.LANGUAGE_CODE = django_lang
        try:
            return self.get_response(request)
        finally:
            translation.deactivate()
