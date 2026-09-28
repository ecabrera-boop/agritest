import os
import sys

from django.apps import AppConfig


def _should_print_training_on_startup() -> bool:
    if "runserver" not in sys.argv:
        return False
    if "--noreload" in sys.argv:
        return True
    return os.environ.get("RUN_MAIN") == "true"


class DetectionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detection"

    def ready(self) -> None:
        if not _should_print_training_on_startup():
            return
        from ml.training_report import print_validation_summary

        print_validation_summary()
