from .i18n import get_lang, get_strings, is_filipino_lang, js_strings
from .models import SiteSettings


def site_settings(request):
    return {"site": SiteSettings.load()}


def ui_language(request):
    lang = get_lang(request)
    t = dict(get_strings(lang))
    brand = SiteSettings.load().site_brand or "RiceGuard"
    t["print_footer"] = t["print_footer"].format(brand=brand)
    return {
        "lang": lang,
        "is_filipino": is_filipino_lang(lang),
        "t": t,
        "js_i18n": js_strings(lang),
    }
