"""生成一张包含全部 11 项必标字段的示例食品标签图片（用于演示 OCR 识别与字段修正）。

用法：
    python manage.py generate_sample_label              # 只生成图片
    python manage.py generate_sample_label --seed       # 生成图片并写入示例数据集（含正确预标注）
"""
import os

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.management.base import BaseCommand

from PIL import Image, ImageDraw, ImageFont

from ocr_demo.models import TrainingDataset, TrainingImage
from ocr_demo.utils import DEFAULT_FIELDS

# 完整的 11 项必标字段示例值（顺序与 DEFAULT_FIELDS 一致）
SAMPLE_FIELDS = {
    "产品名称": "乐享牌原味苏打饼干",
    "产品类型": "发酵饼干",
    "配料": "小麦粉、白砂糖、食用植物油、全脂乳粉、食用盐、碳酸氢钠（膨松剂）、食用香精",
    "净含量": "400克",
    "生产日期": "2026年08月15日",
    "保质期": "12个月",
    "产品标准代号": "GB/T 20980",
    "生产许可证编号": "SC12411020800001",
    "生产厂家": "乐享食品（杭州）有限公司",
    "地址": "浙江省杭州市余杭区仓前街道科技大道99号",
    "产地": "浙江省杭州市",
}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simkai.ttf",
    "C:/Windows/Fonts/msyh.ttc",
]


def _load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_width):
    """按像素宽度将中文文本折行，返回行列表。"""
    if draw.textlength(text, font=font) <= max_width:
        return [text]
    lines, current = [], ""
    for ch in text:
        if draw.textlength(current + ch, font=font) > max_width and current:
            lines.append(current)
            current = ch
        else:
            current += ch
    if current:
        lines.append(current)
    return lines


class Command(BaseCommand):
    help = "生成示例食品标签图片（含 11 项必标字段），可选写入示例数据集"

    def add_arguments(self, parser):
        parser.add_argument("--seed", action="store_true", help="同时写入示例训练数据集（含正确预标注）")

    def handle(self, *args, **options):
        # ---------- 1. 生成图片 ----------
        width, margin, line_h = 1500, 70, 78
        title_font = _load_font(56)
        name_font = _load_font(48)
        body_font = _load_font(40)

        # 预计算各字段行（含折行），得到总高度
        probe = Image.new("RGB", (10, 10), "white")
        pd = ImageDraw.Draw(probe)
        lines = []  # [(text, font)]
        lines.append(("食品标签（示例）", title_font))
        lines.append(("产品名称：" + SAMPLE_FIELDS["产品名称"], name_font))
        for name in DEFAULT_FIELDS.keys():
            if name == "产品名称":
                continue
            full = f"{name}：{SAMPLE_FIELDS[name]}"
            for sub in _wrap_text(pd, full, body_font, width - 2 * margin):
                lines.append((sub, body_font))

        height = margin * 2 + line_h * len(lines) + 30
        img = Image.new("RGB", (width, height), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([12, 12, width - 12, height - 12], outline=(0, 105, 78), width=4)

        y = margin
        boxes = []  # 每个字段行的 (text, x0, y0, x1, y1)
        for text, font in lines:
            d.text((margin, y), text, font=font, fill=(20, 20, 20))
            tw = d.textlength(text, font=font)
            boxes.append((text, margin, y, margin + int(tw), y + line_h))
            y += line_h

        demo_dir = os.path.join(settings.MEDIA_ROOT, "demo_labels")
        os.makedirs(demo_dir, exist_ok=True)
        img_path = os.path.join(demo_dir, "食品标签示例.png")
        img.save(img_path, "PNG")
        self.stdout.write(self.style.SUCCESS(f"示例标签图片已生成：{img_path}"))

        # ---------- 2. 可选写入示例数据集 ----------
        if not options["seed"]:
            return

        User = get_user_model()
        owner = User.objects.filter(is_superuser=True).first() or User.objects.first()
        if owner is None:
            self.stdout.write(self.style.WARNING("系统中暂无用户，跳过数据集写入。"))
            return

        dataset, created = TrainingDataset.objects.get_or_create(
            name="食品标签演示集（示例）",
            created_by=owner,
            defaults={"description": "含全部 11 项必标字段的示例标签，用于演示预标注与字段修正", "dataset_type": "food_label"},
        )

        with open(img_path, "rb") as f:
            rel = default_storage.save(f"training_images/demo/{os.path.basename(img_path)}", ContentFile(f.read()))

        # 用字段行坐标构造逐行标注框（跳过标题行与产品名称行——产品名称单独作为一行）
        annotation_bbox = []
        for text, x0, y0, x1, y1 in boxes:
            annotation_bbox.append({
                "text": text,
                "confidence": 0.99,
                "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            })

        TrainingImage.objects.create(
            dataset=dataset,
            image=rel,
            annotation_text="\n".join(t for t, *_ in boxes),
            annotation_bbox=annotation_bbox,
            structured_fields=SAMPLE_FIELDS,
            annotation_status="annotated",
            annotated_by=owner,
        )
        self.stdout.write(self.style.SUCCESS(f"示例数据集「{dataset.name}」已就绪（id={dataset.id}，含 1 张已修正图片）。"))
