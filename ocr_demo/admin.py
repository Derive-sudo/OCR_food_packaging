from django.contrib import admin
from .models import OCRRecord, TrainingDataset, TrainingImage, TrainingJob

@admin.register(OCRRecord)
class OCRRecordAdmin(admin.ModelAdmin):
    list_display = ("user", "image", "result_text", "confidence", "is_compliant", "created_at", "updated_at")
    list_filter = ("user", "is_compliant", "created_at")
    search_fields = ("user__username", "result_text")
    readonly_fields = ("created_at", "updated_at")

@admin.register(TrainingDataset)
class TrainingDatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "dataset_type", "created_by", "is_active", "created_at")
    list_filter = ("dataset_type", "is_active", "created_at")
    search_fields = ("name", "description")
    readonly_fields = ("created_at", "updated_at")


@admin.register(TrainingImage)
class TrainingImageAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "annotation_status", "annotated_by", "created_at")
    list_filter = ("dataset", "annotation_status", "created_at")
    search_fields = ("dataset__name", "annotation_text")
    readonly_fields = ("created_at",)


@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ("name", "dataset", "status", "progress", "epochs", "created_by", "created_at")
    list_filter = ("status", "dataset__dataset_type", "created_at")
    search_fields = ("name", "dataset__name")
    readonly_fields = ("created_at", "started_at", "completed_at")