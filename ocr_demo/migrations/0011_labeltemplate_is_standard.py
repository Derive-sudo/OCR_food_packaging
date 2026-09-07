from django.db import migrations, models

# 标准食品标签必标字段名（与 ocr_demo/utils.py 的 DEFAULT_FIELDS 保持一致）。
STANDARD_FIELD_NAMES = {
    "产品名称",
    "产品类型",
    "配料",
    "净含量",
    "生产日期",
    "保质期",
    "产品标准代号",
    "生产许可证编号",
    "生产厂家",
    "地址",
    "产地",
}


def mark_standard_templates(apps, schema_editor):
    """回填 is_standard：把历史上由「一键导入 GB 7718」或系统种子创建的模板标为国标模板。

    判据：模板名称/描述提到 GB 7718，或其字段全部取自标准必标字段集（即种子模板，
    含管理员事后删减过字段的情况）。其余一律视为自定义模板。
    """
    LabelTemplate = apps.get_model("ocr_demo", "LabelTemplate")
    LabelField = apps.get_model("ocr_demo", "LabelField")

    for template in LabelTemplate.objects.all():
        text = f"{template.name}{template.description}".replace(" ", "")
        field_names = set(
            LabelField.objects.filter(template=template).values_list("name", flat=True)
        )
        by_name = "GB7718" in text.upper()
        by_fields = bool(field_names) and field_names <= STANDARD_FIELD_NAMES
        if by_name or by_fields:
            template.is_standard = True
            template.save(update_fields=["is_standard"])


def unmark_standard_templates(apps, schema_editor):
    """回滚：清除标记（字段本身由 RemoveField 删除，这里保持可逆）。"""
    LabelTemplate = apps.get_model("ocr_demo", "LabelTemplate")
    LabelTemplate.objects.update(is_standard=False)


class Migration(migrations.Migration):

    dependencies = [
        ("ocr_demo", "0010_trainingimage_structured_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="labeltemplate",
            name="is_standard",
            field=models.BooleanField(
                default=False,
                help_text="标记来自 GB 7718 标准的模板；检测演示页据此显示国标判据，自定义模板则显示模板自身要求",
                verbose_name="国标模板",
            ),
        ),
        migrations.RunPython(mark_standard_templates, unmark_standard_templates),
    ]
