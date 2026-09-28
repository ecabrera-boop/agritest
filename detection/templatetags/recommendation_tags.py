from django import template

from detection.recommendation_display import parse_recommendation

register = template.Library()


@register.filter
def parse_recommendation_text(text):
    return parse_recommendation(text or "")
