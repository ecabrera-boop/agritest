from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("lang/<str:lang>/", views.set_language, name="set_language"),
    path("upload/", views.upload_leaf, name="upload_leaf"),
    path("api/validate-layer0/", views.validate_layer0, name="validate_layer0"),
    path("api/locate/", views.locate_capture, name="locate_capture"),
    path("history/", views.scan_history, name="scan_history"),
    path("locations/", views.scan_locations, name="scan_locations"),
    path("result/<int:pk>/", views.scan_detail, name="scan_detail"),
    path(
        "training-sample/<str:disease>/<str:filename>",
        views.training_sample_image,
        name="training_sample_image",
    ),
    path("api/scan/", views.api_scan, name="api_scan"),
    path("api/scans/", views.api_scan_list, name="api_scan_list"),
    path("api/scans/<int:pk>/", views.api_scan_detail, name="api_scan_detail"),
]
