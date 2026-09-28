from django.contrib import admin

admin.site.site_header = "Rice Leaf Disease Detector"
admin.site.site_title = "RiceGuard Admin"
admin.site.index_title = "Site & scan management"
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),
    path(
        "favicon.ico",
        RedirectView.as_view(url="/static/favicon.svg", permanent=False),
        name="favicon",
    ),
    path("", include("detection.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else settings.STATIC_ROOT
    urlpatterns += static(settings.STATIC_URL, document_root=static_dir)
