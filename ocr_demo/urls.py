from django.urls import path
from . import views

urlpatterns = [
    #OCR检测
    path("", views.ocr_home, name="ocr_home"),
    #检测记录
    path("record/", views.ocr_records, name="ocr_records"),
    #管理员记录
    path("records/", views.admin_records, name="admin_records"),
    #导出记录（CSV）
    path("records/export/", views.admin_export_records, name="admin_export_records"),
    #样本管理
    path("samples/", views.samples_manage, name="samples_manage"),
    #扩展功能：图像优化
    path("image-optimize/", views.image_optimize, name="image_optimize"),
    path("image-optimize/detect/", views.image_optimize_detect, name="image_optimize_detect"),
    #导出样本（CSV）
    path("samples/export/", views.samples_export, name="samples_export"),
    #管理员：标签数据管理 / 模型配置 / 标签模板管理
    path("admin/tag-data/", views.admin_tag_data, name="admin_tag_data"),
    path("admin/model-config/", views.admin_model_config, name="admin_model_config"),
    path("admin/label-templates/", views.admin_label_templates, name="admin_label_templates"),
    #管理员：模型评估
    path("admin/model-evaluate/", views.admin_model_evaluate, name="admin_model_evaluate"),
    path("admin/model-evaluate/<int:run_id>/", views.admin_model_evaluate_detail, name="admin_model_evaluate_detail"),
    #模型微调
    path("finetune/", views.finetune_model, name="finetune_model"),
    path("finetune/model/", views.finetune_set_model, name="finetune_set_model"),
    # 训练数据集管理
    path("finetune/datasets/", views.finetune_datasets, name="finetune_datasets"),
    path("finetune/datasets/create/", views.finetune_dataset_create, name="finetune_dataset_create"),
    path("finetune/datasets/<int:dataset_id>/", views.finetune_dataset_detail, name="finetune_dataset_detail"),
    path("finetune/datasets/<int:dataset_id>/upload/", views.finetune_image_upload, name="finetune_image_upload"),
    path("finetune/datasets/<int:dataset_id>/annotate/<int:image_id>/", views.finetune_image_annotate, name="finetune_image_annotate"),
    path("finetune/datasets/<int:dataset_id>/prelabel/", views.finetune_dataset_prelabel, name="finetune_dataset_prelabel"),
    path("finetune/datasets/<int:dataset_id>/export/", views.finetune_dataset_export, name="finetune_dataset_export"),
    # 训练任务管理
    path("finetune/jobs/", views.finetune_jobs, name="finetune_jobs"),
    path("finetune/start/", views.finetune_start_training, name="finetune_start_training"),
    path("finetune/jobs/create/", views.finetune_job_create, name="finetune_job_create"),
    path("finetune/jobs/<int:job_id>/", views.finetune_job_detail, name="finetune_job_detail"),
    path("finetune/jobs/<int:job_id>/start/", views.finetune_job_start, name="finetune_job_start"),
    path("finetune/jobs/<int:job_id>/stop/", views.finetune_job_stop, name="finetune_job_stop"),
    path("finetune/jobs/<int:job_id>/logs/", views.finetune_job_logs, name="finetune_job_logs"),
]

