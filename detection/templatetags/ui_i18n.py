from django import template
from django.utils.safestring import mark_safe

from detection.i18n import get_lang, health_label as _health_label

register = template.Library()


@register.simple_tag(takes_context=True)
def health_label(context, health_or_scan):
    """Render a health badge label.

    Pass either a health string (``healthy`` / ``diseased`` / ``unknown``) or a
    ``LeafScan`` — scans rejected by the rice gate show as "Not a Rice Leaf".
    """
    request = context.get("request")
    lang = context.get("lang") or get_lang(request)
    if hasattr(health_or_scan, "is_not_rice"):
        if health_or_scan.is_not_rice:
            return _health_label("not_rice", lang)
        return _health_label(health_or_scan.result_health or "", lang)
    return _health_label(health_or_scan or "", lang)


@register.filter(name="fmt")
def fmt(template_str, arg):
    """Format ``template_str`` with comma-separated positional args or a mapping.

    Usage:
      {{ t.pesticide_n_of|fmt:forloop.counter|fmt_kw }}  — not used
      Prefer: {% format_str t.pesticide_n_of n=forloop.counter total=count %}
    """
    if template_str is None:
        return ""
    try:
        if isinstance(arg, (list, tuple)):
            return template_str.format(*arg)
        return template_str.format(arg)
    except (IndexError, KeyError, ValueError):
        return template_str


@register.simple_tag
def format_str(template_str, **kwargs):
    if not template_str:
        return ""
    try:
        return mark_safe(template_str.format(**kwargs))
    except (KeyError, ValueError):
        return template_str
