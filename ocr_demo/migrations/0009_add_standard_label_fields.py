from django.db import migrations

# 标准食品标签必标字段（与 ocr_demo/utils.py 的 DEFAULT_FIELDS 保持一致），
# 用于给已存在的标签模板补齐缺失字段（GB 7718 要求标注的核心信息）。
STANDARD_FIELDS = {
    "产品名称": ["产品名称", "品名", "食品名称", "产品名"],
    "产品类型": ["产品类型", "产品类别", "品类"],
    "配料": ["配料", "配料表", "原料"],
    "净含量": ["净含量", "净重", "规格"],
    "生产日期": ["生产日期", "生产日期见"],
    "保质期": ["保质期", "保质日期", "保质期至"],
    "产品标准代号": ["产品标准代号", "产品标准号", "执行标准", "标准代号", "GB"],
    "生产许可证编号": ["生产许可证编号", "生产许可证号", "食品生产许可证", "许可证编号", "SC"],
    "生产厂家": ["生产厂家", "生产商", "制造商", "生产者", "厂家", "委托方"],
    "地址": ["地址", "厂址", "生产地址", "公司地址"],
    "产地": ["产地", "原产地"],
}


def add_standard_fields(apps, schema_editor):
    """给所有标签模板补齐标准字段（仅新增缺失项，不删除用户已有字段）。"""
    LabelTemplate = apps.get_model("ocr_demo", "LabelTemplate")
    LabelField = apps.get_model("ocr_demo", "LabelField")

    for template in LabelTemplate.objects.all():
        existing = set(
            LabelField.objects.filter(template=template).values_list("name", flat=True)
        )
        order = LabelField.objects.filter(template=template).count()
        for name, keywords in STANDARD_FIELDS.items():
            if name not in existing:
                LabelField.objects.create(
                    template=template,
                    name=name,
                    keywords=",".join(keywords),
                    order=order,
                )
                order += 1


def remove_standard_fields(apps, schema_editor):
    """回滚：删除刚补齐的标准字段（仅删与本迁移字段同名的字段）。"""
    LabelField = apps.get_model("ocr_demo", "LabelField")
    LabelField.objects.filter(name__in=STANDARD_FIELDS.keys()).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("ocr_demo", "0008_trainingjob_base_model_trainingjob_cosine_lr_and_more"),
    ]

    operations = [
        migrations.RunPython(add_standard_fields, remove_standard_fields),
    ]
