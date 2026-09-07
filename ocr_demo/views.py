import os
import io
import re
import sys
import uuid
import csv
import glob
import zipfile
from datetime import datetime
import threading
import subprocess
import time

os.environ["FLAGS_use_onednn"] = "0"  # 禁用OneDNN

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.paginator import Paginator
from django.db.models import Avg, Count, Q
from django.utils import timezone
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import ensure_csrf_cookie

from PIL import Image, ImageDraw

import cv2
import numpy as np

from paddleocr import PaddleOCR

from .models import OCRRecord, TrainingDataset, TrainingImage, TrainingJob, LabelTemplate, LabelField, EvaluationRun
from .utils import check_compliance, extract_fields, get_standard_fields, get_active_template, DEFAULT_FIELDS
from user_auth.models import SystemSettings, UserProfile, log_action


# 识别模型实例（懒加载 + 缓存：模型切换后按 key 变化自动重新加载）
_ocr_instance = None
_ocr_key = None


def get_ocr():
    """返回当前系统设置对应的 PaddleOCR 实例。

    识别模型可在「默认 OCR / 微调 / DB 检测 / CRNN 识别 / 自定义」间切换（存于 SystemSettings）。
    db_det / crnn_rec 使用官方权重（DB=检测、CRNN=识别本就是官方模型），
    由调用方决定只跑 det 或只跑 rec。模型切换后 key 变化，下次调用时自动重新加载。
    """
    global _ocr_instance, _ocr_key
    settings_obj = SystemSettings.load()
    mode = settings_obj.rec_model

    if mode == "finetuned" and settings_obj.finetuned_model_dir:
        key = ("finetuned", settings_obj.finetuned_model_dir)
    elif mode == "custom" and settings_obj.custom_model_dir:
        key = ("custom", settings_obj.custom_model_dir)
    else:
        key = ("official", None)

    if _ocr_instance is None or _ocr_key != key:
        kwargs = {"use_angle_cls": True, "lang": "ch"}
        if key[0] in ("finetuned", "custom"):
            kwargs["rec_model_dir"] = key[1]
        _ocr_instance = PaddleOCR(**kwargs)
        _ocr_key = key
    return _ocr_instance


def get_rec_mode():
    """返回当前识别模型模式（official / finetuned / db_det / crnn_rec / custom）。"""
    return SystemSettings.load().rec_model


def _run_prelabel(image_path):
    """对图片跑 PaddleOCR，返回文字行标注列表。

    返回: list[dict]，每项 {"text": str, "confidence": float, "points": [[x,y]*4]}
    无文字时返回空列表。
    """
    result = get_ocr().ocr(image_path)
    lines = []
    if result and result[0]:
        for line in result[0]:
            box = line[0]          # 4 个角点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text, conf = line[1]   # (文字, 置信度)
            lines.append({
                "text": text,
                "confidence": round(float(conf), 4),
                "points": [[int(p[0]), int(p[1])] for p in box],
            })
    return lines


def is_admin_user(user):
    """判断用户是否为管理员。"""
    profile = getattr(user, "profile", None)
    return profile is not None and profile.role == UserProfile.ROLE_ADMIN

def _extract_rec_text(result):
    """从 rec-only（det=False）结果中提取识别文本与置信度。"""
    if not result:
        return "", None
    page = result[0] if isinstance(result, (list, tuple)) else result
    if not isinstance(page, (list, tuple)) or not page:
        return "", None
    first = page[0]
    if isinstance(first, str):
        return first.strip(), None
    if isinstance(first, (list, tuple)) and first:
        text = str(first[0]).strip()
        conf = None
        if len(first) > 1 and isinstance(first[1], (int, float)):
            conf = float(first[1])
        return text, conf
    return "", None


def _draw_det_boxes(full_path, boxes):
    """在原图上绘制 DB 检测框，返回相对 media 路径（无框或失败返回 None）。"""
    if not boxes:
        return None
    try:
        with Image.open(full_path) as im:
            draw = ImageDraw.Draw(im)
            for box in boxes:
                pts = [(int(p[0]), int(p[1])) for p in box]
                if len(pts) < 4:
                    continue
                draw.polygon(pts, outline=(0, 105, 78), width=3)
            root, _ = os.path.splitext(full_path)
            annotated_path = root + "_det.jpg"
            im.convert("RGB").save(annotated_path, "JPEG")
        return os.path.relpath(annotated_path, settings.MEDIA_ROOT).replace("\\", "/")
    except Exception:
        return None


def _render_ocr_home(request, extra=None):
    """渲染检测演示页，并始终带上可选的图像处理方法。"""
    context = {
        "optimize_groups": optimize_method_groups(),
    }
    if extra:
        context.update(extra)
    return render(request, "ocr_demo/ocr_home.html", context)


@login_required
def ocr_home(request):
    """OCR 首页：上传图片并识别（默认/微调/DB检测/CRNN识别/自定义五种模式）。"""
    if request.method == "POST":
        if "image" not in request.FILES:
            messages.error(request, "请选择要上传的图片文件！")
            return _render_ocr_home(request)

        image_file = request.FILES["image"]

        # 检查文件类型
        if not image_file.content_type.startswith("image/"):
            messages.error(request, "请上传有效的图片文件（jpg/png等）！")
            return _render_ocr_home(request)

        try:
            # 1. 保存图片到 media 目录
            file_name = f"{uuid.uuid4().hex}_{image_file.name}"
            file_path = default_storage.save(f"ocr_images/{datetime.now().strftime('%Y/%m/%d')}/{file_name}", ContentFile(image_file.read()))
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)

            # 可选：识别前先做图像优化（先优化再识别）
            optimize_method = request.POST.get("optimize_method", "").strip()
            optimized = False
            if request.POST.get("optimize") == "on" and optimize_method in OPTIMIZE_METHODS:
                opt_rel, _ = _apply_image_optimize(full_path, optimize_method)
                full_path = os.path.join(settings.MEDIA_ROOT, opt_rel)
                file_path = opt_rel  # 后续分支统一使用优化后的图片
                optimized = True

            mode = get_rec_mode()
            mode_label = dict(SystemSettings.REC_MODEL_CHOICES).get(mode, mode)
            packaging_category = request.POST.get("packaging_category", "").strip()
            valid_categories = [c[0] for c in OCRRecord.PACKAGING_CHOICES]
            if packaging_category not in valid_categories:
                packaging_category = ""
            ocr = get_ocr()

            # 2. DB 文本检测：只框出文字区域
            if mode == "db_det":
                det_result = ocr.ocr(full_path, det=True, rec=False, cls=False)
                boxes = det_result[0] if det_result and det_result[0] else []
                annotated_rel = _draw_det_boxes(full_path, boxes)
                messages.success(request, f"DB 文本检测完成，共检测到 {len(boxes)} 个文字区域。")
                return _render_ocr_home(request, {
                    "result": f"检测到 {len(boxes)} 个文字区域",
                    "mode": mode,
                    "mode_label": mode_label,
                    "detected_count": len(boxes),
                    "boxes": boxes,
                    "annotated_image_url": (settings.MEDIA_URL + annotated_rel) if annotated_rel else None,
                    "image_url": settings.MEDIA_URL + file_path,
                })

            # 3. CRNN 文字识别：只识别文字
            if mode == "crnn_rec":
                rec_result = ocr.ocr(full_path, det=False, rec=True, cls=False)
                rec_text, rec_confidence = _extract_rec_text(rec_result)
                messages.success(request, "CRNN 文字识别完成。")
                return _render_ocr_home(request, {
                    "result": rec_text or "未识别到文字",
                    "mode": mode,
                    "mode_label": mode_label,
                    "rec_text": rec_text or "未识别到文字",
                    "rec_confidence": rec_confidence,
                    "image_url": settings.MEDIA_URL + file_path,
                })

            # 4. 完整识别（默认 / 微调 / 自定义）
            context = _recognize_full(request, full_path, file_path, packaging_category)
            if optimized:
                context.update({
                    "optimized": True,
                    "optimize_method_label": OPTIMIZE_METHODS.get(optimize_method, optimize_method),
                })
            return _render_ocr_home(request, context)

        except Exception as e:
            messages.error(request, f"OCR 识别失败：{str(e)}")
            return _render_ocr_home(request)

    return _render_ocr_home(request)


def _recognize_full(request, full_path, file_path, packaging_category=""):
    """对指定图片路径跑完整 OCR，建识别记录并返回渲染上下文（ocr_home 与图像优化共用）。"""
    ocr = get_ocr()
    result = ocr.ocr(full_path)
    if result and result[0]:
        text_lines = [line[1][0] for line in result[0]]
        full_text = "\n".join(text_lines)
        avg_confidence = sum(line[1][1] for line in result[0]) / len(result[0]) if result[0] else 0
    else:
        full_text = "未识别到任何文字"
        avg_confidence = 0

    # 识别端不再使用可编辑标签模板，改为固定按 GB 7718 必标字段做合规检测与字段展示
    fields = get_standard_fields()
    compliance = check_compliance(full_text, fields)
    extracted_fields = extract_fields(full_text, fields)
    mode = get_rec_mode()
    mode_label = dict(SystemSettings.REC_MODEL_CHOICES).get(mode, mode)

    record = OCRRecord.objects.create(
        user=request.user,
        image=file_path,
        result_text=full_text,
        confidence=avg_confidence if avg_confidence > 0 else None,
        is_compliant=compliance["is_compliant"],
        compliance_issues=compliance["issues"],
        missing_fields=compliance["missing_fields"],
        packaging_category=packaging_category,
    )

    messages.success(request, "OCR 识别成功！")
    log_action(request.user, "ocr_detect", f"识别「{packaging_category or '通用'}」外包装图片", request)
    return {
        "result": full_text,
        "mode": mode,
        "mode_label": mode_label,
        "record": record,
        "image_url": record.image.url if record.image else None,
        "extracted_fields": extracted_fields,
    }


# 图像优化方法标签（扩展功能）
OPTIMIZE_METHODS = {
    "gray": "灰度化",
    "denoise": "高斯降噪",
    "median": "中值降噪",
    "contrast": "对比度增强 (CLAHE)",
    "sharpen": "锐化",
    "binarize": "二值化 (Otsu)",
    "rotate": "旋转 (+8°)",
    "shear": "错切 (x 方向 0.2)",
    "cutout": "随机擦除 (Cutout)",
}

# 前一组改善画质、利于识别；后一组模拟拍摄畸变与遮挡，用于扩充训练样本。
# 分组只影响页面展示，校验与取名仍统一走 OPTIMIZE_METHODS。
OPTIMIZE_GROUPS = [
    ("图像优化（改善画质）", ["gray", "denoise", "median", "contrast", "sharpen", "binarize"]),
    ("图像增强（模拟畸变遮挡）", ["rotate", "shear", "cutout"]),
]


# 未指定方法时的默认项（原「综合增强」已移除）
DEFAULT_OPTIMIZE_METHOD = "contrast"

# 增强参数取固定值而非随机：优化前后对比与实训报告需要可复现的结果
ROTATE_ANGLE = 8.0      # 旋转角度（度，逆时针）
SHEAR_FACTOR = 0.2      # x 方向错切系数
CUTOUT_RATIO = 0.25     # 擦除方块边长占图片短边的比例


def optimize_method_groups():
    """返回 [(组名, [(value, label), ...]), ...]，供模板分组渲染。"""
    return [
        (group, [(v, OPTIMIZE_METHODS[v]) for v in values])
        for group, values in OPTIMIZE_GROUPS
    ]


def _imread_any(path):
    """中文路径安全的图片读取（cv2.imread 在 Windows 中文路径会失败）。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_any(path, img):
    """中文路径安全的图片写入。"""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("编码输出图片失败")
    buf.tofile(path)


def _apply_image_optimize(input_path, method):
    """对图片应用指定 OpenCV 处理，返回 (相对路径, 方法中文名)。

    含两类：改善画质的预处理（灰度/降噪/增强/锐化/二值化），以及模拟拍摄畸变与
    遮挡的图像增强（旋转/错切/擦除），后者用于扩充训练样本、检验识别鲁棒性。
    """
    img = _imread_any(input_path)
    if img is None:
        raise ValueError("无法读取图片（格式不支持或文件损坏）")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "gray":
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif method == "denoise":
        out = cv2.GaussianBlur(img, (5, 5), 0)
    elif method == "median":
        out = cv2.medianBlur(img, 5)
    elif method == "contrast":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    elif method == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        out = cv2.filter2D(img, -1, kernel)
    elif method == "binarize":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    elif method == "rotate":
        # 绕中心旋转固定角度，模拟拍摄时的倾斜；边缘复制填充，避免出现黑角干扰 OCR
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), ROTATE_ANGLE, 1.0)
        out = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    elif method == "shear":
        # x 方向错切，模拟从侧面拍摄包装造成的倾斜变形；画布加宽以容纳错开的部分
        h, w = img.shape[:2]
        matrix = np.float32([[1, SHEAR_FACTOR, 0], [0, 1, 0]])
        out = cv2.warpAffine(img, matrix, (w + int(h * SHEAR_FACTOR), h),
                             borderMode=cv2.BORDER_REPLICATE)
    elif method == "cutout":
        # 擦除一块矩形区域（填充灰色），模拟包装褶皱、反光或遮挡造成的信息缺失
        out = img.copy()
        h, w = out.shape[:2]
        box = max(1, int(min(h, w) * CUTOUT_RATIO))
        y0 = min(max(0, int(h * 0.4) - box // 2), max(0, h - box))
        x0 = min(max(0, int(w * 0.5) - box // 2), max(0, w - box))
        out[y0:y0 + box, x0:x0 + box] = 114  # 114 是常用的填充均值
    else:
        out = img

    ts = datetime.now().strftime("%Y/%m/%d")
    out_rel = f"image_optimize/{ts}/opt_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(settings.MEDIA_ROOT, out_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _imwrite_any(out_path, out)
    return out_rel, OPTIMIZE_METHODS.get(method, method)


def _ocr_full_text(path):
    """对图片跑完整 OCR，返回拼接文本（用于图像优化前后对比）。"""
    try:
        result = get_ocr().ocr(path)
        if result and result[0]:
            return "\n".join(line[1][0] for line in result[0])
    except Exception:
        pass
    return ""


@login_required
def image_optimize(request):
    """扩展功能：图像优化（OpenCV 预处理 + 前后对比 + 可选 OCR 识别）。"""
    context = {}
    if request.method == "POST":
        if "image" not in request.FILES:
            messages.error(request, "请先选择要优化的图片。")
            return redirect("image_optimize")

        image_file = request.FILES["image"]
        if not image_file.content_type.startswith("image/"):
            messages.error(request, "请上传有效的图片文件（jpg/png等）。")
            return redirect("image_optimize")

        method = request.POST.get("method", DEFAULT_OPTIMIZE_METHOD)
        if method not in OPTIMIZE_METHODS:
            method = DEFAULT_OPTIMIZE_METHOD
        run_ocr = request.POST.get("run_ocr") == "on"

        try:
            ts = datetime.now().strftime("%Y/%m/%d")
            orig_rel = default_storage.save(
                f"image_optimize/{ts}/orig_{uuid.uuid4().hex}_{image_file.name}",
                ContentFile(image_file.read()),
            )
            orig_path = os.path.join(settings.MEDIA_ROOT, orig_rel)

            opt_rel, method_label = _apply_image_optimize(orig_path, method)

            orig_text = opt_text = None
            if run_ocr:
                orig_text = _ocr_full_text(orig_path)
                opt_text = _ocr_full_text(os.path.join(settings.MEDIA_ROOT, opt_rel))

            context.update({
                "original_url": settings.MEDIA_URL + orig_rel,
                "optimized_url": settings.MEDIA_URL + opt_rel,
                "optimized_path": opt_rel,
                "method": method,
                "method_label": method_label,
                "orig_text": orig_text,
                "opt_text": opt_text,
            })
            log_action(request.user, "ocr_detect", f"图像优化：{method_label}", request)
            messages.success(request, f"已应用「{method_label}」，前后对比结果如下。")
        except Exception as e:
            messages.error(request, f"图像优化失败：{str(e)}")

    context["optimize_groups"] = optimize_method_groups()
    return render(request, "ocr_demo/image_optimize.html", context)


@login_required
@require_POST
def image_optimize_detect(request):
    """用优化后的图片直接进行 OCR 识别（跳到检测演示结果页）。"""
    image_path = request.POST.get("image_path", "").strip()
    full_path = os.path.join(settings.MEDIA_ROOT, image_path)
    if not image_path or not os.path.exists(full_path):
        messages.error(request, "优化图片不存在，请先完成图像优化。")
        return redirect("image_optimize")

    packaging_category = request.POST.get("packaging_category", "").strip()
    valid_categories = [c[0] for c in OCRRecord.PACKAGING_CHOICES]
    if packaging_category not in valid_categories:
        packaging_category = ""

    context = _recognize_full(request, full_path, image_path, packaging_category)
    return _render_ocr_home(request, context)


@login_required
def ocr_records(request):
    """查看当前用户的识别记录"""
    records = OCRRecord.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "ocr_demo/ocr_records.html", {"records": records})


def _is_admin_user(user):
    """判断用户是否为管理员（兼容 profile 缺失的情况）。"""
    profile = getattr(user, "profile", None)
    return profile is not None and profile.role == UserProfile.ROLE_ADMIN


def _parse_date(value):
    """将 'YYYY-MM-DD' 字符串解析为 date 对象，非法值返回 None。"""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _apply_record_filters(records, request):
    """根据请求参数对记录 queryset 进行筛选，返回筛选后的 queryset。"""
    q = request.GET.get("q", "").strip()
    if q:
        records = records.filter(
            Q(user__username__icontains=q)
            | Q(user__first_name__icontains=q)
            | Q(user__last_name__icontains=q)
            | Q(result_text__icontains=q)
        )

    compliance = request.GET.get("compliance", "").strip()
    if compliance == "compliant":
        records = records.filter(is_compliant=True)
    elif compliance == "non_compliant":
        records = records.filter(is_compliant=False)

    start_date = request.GET.get("start_date", "").strip()
    end_date = request.GET.get("end_date", "").strip()
    if start_date:
        d = _parse_date(start_date)
        if d:
            records = records.filter(created_at__date__gte=d)
    if end_date:
        d = _parse_date(end_date)
        if d:
            records = records.filter(created_at__date__lte=d)

    min_confidence = request.GET.get("min_confidence", "").strip()
    max_confidence = request.GET.get("max_confidence", "").strip()
    if min_confidence:
        try:
            records = records.filter(confidence__gte=float(min_confidence))
        except ValueError:
            pass
    if max_confidence:
        try:
            records = records.filter(confidence__lte=float(max_confidence))
        except ValueError:
            pass

    return records


@login_required
def admin_records(request):
    """管理员实训记录列表：查看、筛选、分页所有学生的识别记录（仅管理员可访问）。"""
    if not _is_admin_user(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    records = OCRRecord.objects.select_related("user").all().order_by("-created_at")
    records = _apply_record_filters(records, request)

    # 回填筛选参数（用于表单和分页链接）
    q = request.GET.get("q", "").strip()
    compliance = request.GET.get("compliance", "").strip()
    start_date = request.GET.get("start_date", "").strip()
    end_date = request.GET.get("end_date", "").strip()
    min_confidence = request.GET.get("min_confidence", "").strip()
    max_confidence = request.GET.get("max_confidence", "").strip()

    # 统计卡片（基于全部记录，不随筛选变化）
    today_count = OCRRecord.objects.filter(
        created_at__date=timezone.localdate()
    ).count()
    total_count = OCRRecord.objects.count()
    compliant_count = OCRRecord.objects.filter(is_compliant=True).count()
    avg_confidence = OCRRecord.objects.filter(confidence__isnull=False).aggregate(
        avg=Avg("confidence")
    )["avg"]
    compliance_rate = (compliant_count / total_count * 100) if total_count else 0

    # 分页（每页 10 条）
    paginator = Paginator(records, 10)
    page_obj = paginator.get_page(request.GET.get("page"))

    # 保留筛选参数用于分页链接回填
    query_params = request.GET.copy()
    query_params.pop("page", None)
    querystring = query_params.urlencode()

    context = {
        "active_nav": "training_records",
        "records": page_obj.object_list,
        "page_obj": page_obj,
        "today_count": today_count,
        "total_count": total_count,
        "compliant_count": compliant_count,
        "avg_confidence": avg_confidence * 100 if avg_confidence is not None else 0,
        "compliance_rate": compliance_rate,
        "querystring": querystring,
        # 回填筛选参数
        "q": q,
        "compliance": compliance,
        "start_date": start_date,
        "end_date": end_date,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
    }
    return render(request, "admin/training_records.html", context)


@login_required
def admin_export_records(request):
    """导出当前筛选条件下的识别记录为 CSV 文件（仅管理员可访问）。"""
    if not _is_admin_user(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    records = OCRRecord.objects.select_related("user").all().order_by("-created_at")
    records = _apply_record_filters(records, request)

    response = HttpResponse(content_type="text/csv; charset=utf-8-sig")
    response["Content-Disposition"] = 'attachment; filename="ocr_records.csv"'

    writer = csv.writer(response)
    writer.writerow(["ID", "用户名", "检测时间", "合规", "置信度", "缺失字段", "识别结果"])
    for r in records:
        writer.writerow([
            r.id,
            r.user.username,
            timezone.localtime(r.created_at).strftime("%Y-%m-%d %H:%M:%S"),
            "合规" if r.is_compliant else "不合规",
            f"{r.confidence:.4f}" if r.confidence is not None else "",
            "、".join(r.missing_fields or []),
            r.result_text or "",
        ])
    return response


@login_required
def admin_tag_data(request):
    """管理员标签数据管理：查看所有训练数据集（仅管理员可访问）。"""
    if not _is_admin_user(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    datasets = TrainingDataset.objects.select_related("created_by").annotate(
        image_count=Count("images"),
        verified_count=Count("images", filter=Q(images__annotation_status="verified")),
    ).order_by("-created_at")

    paginator = Paginator(datasets, 9)
    page_obj = paginator.get_page(request.GET.get("page"))

    context = {
        "active_nav": "tag_data",
        "datasets": page_obj.object_list,
        "page_obj": page_obj,
        "dataset_count": TrainingDataset.objects.count(),
        "image_count": TrainingImage.objects.count(),
    }
    return render(request, "admin/tag_data.html", context)


@login_required
def admin_model_config(request):
    """管理员模型配置：查看所有训练任务（仅管理员可访问）。"""
    if not _is_admin_user(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    jobs = TrainingJob.objects.select_related("dataset", "created_by").order_by("-created_at")
    paginator = Paginator(jobs, 10)
    page_obj = paginator.get_page(request.GET.get("page"))

    context = {
        "active_nav": "model_config",
        "jobs": page_obj.object_list,
        "page_obj": page_obj,
        "job_count": TrainingJob.objects.count(),
        "running_count": TrainingJob.objects.filter(status="running").count(),
    }
    return render(request, "admin/model_config.html", context)


@login_required
def admin_label_templates(request):
    """标签模板管理：维护合规标签模板及其字段（仅管理员可访问）。"""
    if not _is_admin_user(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    # 确保至少存在一个启用模板（懒种子默认模板）
    active = get_active_template()

    if request.method == "POST":
        action = request.POST.get("action", "").strip()

        if action == "create_template":
            name = request.POST.get("name", "").strip()
            description = request.POST.get("description", "").strip()
            if not name:
                messages.error(request, "模板名称不能为空。")
            else:
                tpl = LabelTemplate.objects.create(name=name, description=description, is_active=True)
                LabelTemplate.objects.exclude(pk=tpl.pk).update(is_active=False)
                messages.success(request, f"已新建模板「{name}」并启用。")
            return redirect("admin_label_templates")

        if action == "activate_template":
            tpl = get_object_or_404(LabelTemplate, id=request.POST.get("template_id"))
            LabelTemplate.objects.update(is_active=False)
            tpl.is_active = True
            tpl.save()
            messages.success(request, f"已启用模板「{tpl.name}」。")
            return redirect("admin_label_templates")

        if action == "delete_template":
            tpl = get_object_or_404(LabelTemplate, id=request.POST.get("template_id"))
            if LabelTemplate.objects.count() <= 1:
                messages.error(request, "至少需要保留一个模板。")
            else:
                name = tpl.name
                was_active = tpl.is_active
                tpl.delete()
                if was_active:
                    first = LabelTemplate.objects.order_by("id").first()
                    if first:
                        first.is_active = True
                        first.save()
                messages.success(request, f"已删除模板「{name}」。")
            return redirect("admin_label_templates")

        if action == "save_field":
            tpl = get_object_or_404(LabelTemplate, id=request.POST.get("template_id"))
            name = request.POST.get("name", "").strip()
            keywords = request.POST.get("keywords", "").strip()
            order_str = request.POST.get("order", "0").strip()
            if not name or not keywords:
                messages.error(request, "字段名称和关键词不能为空。")
                return redirect(f"{reverse('admin_label_templates')}?template_id={tpl.id}")
            try:
                order = int(order_str)
            except ValueError:
                order = 0
            field_id = request.POST.get("field_id", "").strip()
            if field_id:
                field = get_object_or_404(LabelField, id=field_id, template=tpl)
                field.name = name
                field.keywords = keywords
                field.order = order
                field.save()
                messages.success(request, f"已更新字段「{name}」。")
            else:
                LabelField.objects.create(template=tpl, name=name, keywords=keywords, order=order)
                messages.success(request, f"已新增字段「{name}」。")
            return redirect(f"{reverse('admin_label_templates')}?template_id={tpl.id}")

        if action == "delete_field":
            field = get_object_or_404(LabelField, id=request.POST.get("field_id"))
            tpl_id = field.template_id
            field.delete()
            messages.success(request, f"已删除字段「{field.name}」。")
            return redirect(f"{reverse('admin_label_templates')}?template_id={tpl_id}")

        if action == "edit_template":
            tpl = get_object_or_404(LabelTemplate, id=request.POST.get("template_id"))
            name = request.POST.get("name", "").strip()
            description = request.POST.get("description", "").strip()
            if not name:
                messages.error(request, "模板名称不能为空。")
            else:
                tpl.name = name
                tpl.description = description
                tpl.save()
                messages.success(request, f"已更新模板「{name}」。")
            return redirect("admin_label_templates")

        if action == "import_standard_template":
            # 一键导入 GB 7718 标准食品标签模板（11 个必标字段）
            name = request.POST.get("name", "").strip() or "标准食品标签（GB 7718）"
            tpl = LabelTemplate.objects.create(
                name=name,
                description="依据 GB 7718-2011《预包装食品标签通则》的标准必标字段模板",
                is_active=True,
                is_standard=True,
            )
            LabelTemplate.objects.exclude(pk=tpl.pk).update(is_active=False)
            for i, (fname, keywords) in enumerate(DEFAULT_FIELDS.items()):
                LabelField.objects.create(template=tpl, name=fname, keywords=",".join(keywords), order=i)
            messages.success(request, f"已导入标准模板「{name}」（{len(DEFAULT_FIELDS)} 个字段）并启用。")
            return redirect("admin_label_templates")

        return redirect("admin_label_templates")

    # GET
    templates = LabelTemplate.objects.annotate(field_count=Count("fields")).order_by("-is_active", "-updated_at")

    target_id = request.GET.get("template_id")
    target = LabelTemplate.objects.filter(id=target_id).first() if target_id else None
    if target is None:
        target = active

    fields = target.fields.order_by("order", "id") if target else LabelField.objects.none()
    edit_field = None
    edit_field_id = request.GET.get("edit_field_id")
    if edit_field_id and target:
        edit_field = LabelField.objects.filter(id=edit_field_id, template=target).first()

    edit_template = None
    edit_template_id = request.GET.get("edit_template_id")
    if edit_template_id:
        edit_template = LabelTemplate.objects.filter(id=edit_template_id).first()

    context = {
        "active_nav": "label_templates",
        "templates": templates,
        "target": target,
        "fields": fields,
        "edit_field": edit_field,
        "edit_template": edit_template,
        "standard_field_names": list(DEFAULT_FIELDS.keys()),
    }
    return render(request, "admin/label_templates.html", context)


def list_eval_datasets():
    """扫描项目根目录下一级子目录里的 rec_gt*.txt，作为可选评测数据。"""
    pattern = os.path.join(str(settings.BASE_DIR), "*", "rec_gt*.txt")
    datasets = []
    for path in sorted(glob.glob(pattern)):
        if "PaddleOCR" in path:
            continue
        rel = os.path.relpath(path, str(settings.BASE_DIR))
        datasets.append((path, rel))
    return datasets


def _run_evaluation(run):
    """在后台线程中执行一次模型评估并更新结果。"""
    from model_train.evaluate_rec import load_ocr, parse_gt, evaluate
    try:
        items, _ = parse_gt(run.data_path)
        ocr = load_ocr(run.model_dir or None)
        metrics = evaluate(ocr, items, "评估")
        run.char_accuracy = round(metrics["char_accuracy"], 2)
        run.line_accuracy = round(metrics["line_accuracy"], 2)
        run.total_lines = metrics["total_lines"]
        run.exact_lines = metrics["exact_lines"]
        run.metrics = {"worst_errors": metrics["worst_errors"]}
        run.status = "completed"
        run.completed_at = timezone.now()
        run.save()
    except Exception as e:
        run.status = "failed"
        run.error_message = str(e)
        run.completed_at = timezone.now()
        run.save()


@login_required
def admin_model_evaluate(request):
    """模型评估：选择评测数据与模型，后台评估 rec 模型准确率（仅管理员）。"""
    if not _is_admin_user(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    if request.method == "POST":
        action = request.POST.get("action", "").strip()

        if action == "start_eval":
            data_path = request.POST.get("data_path", "").strip()
            model_dir = request.POST.get("model_dir", "").strip()
            if not data_path or not os.path.exists(data_path):
                messages.error(request, "请选择有效的评测数据文件。")
                return redirect("admin_model_evaluate")
            if model_dir and not os.path.exists(os.path.join(model_dir, "inference.pdmodel")):
                messages.error(request, f"未在模型目录找到 inference.pdmodel：{model_dir}")
                return redirect("admin_model_evaluate")
            run = EvaluationRun.objects.create(
                data_path=data_path,
                model_dir=model_dir,
                status="running",
                created_by=request.user,
            )
            t = threading.Thread(target=_run_evaluation, args=(run,))
            t.daemon = True
            t.start()
            messages.success(request, "评估任务已启动，完成后会自动更新结果。")
            return redirect("admin_model_evaluate")

        if action == "delete_run":
            run = get_object_or_404(EvaluationRun, id=request.POST.get("run_id"))
            run.delete()
            messages.success(request, "已删除该评估记录。")
            return redirect("admin_model_evaluate")

        return redirect("admin_model_evaluate")

    runs = EvaluationRun.objects.select_related("created_by").order_by("-created_at")
    context = {
        "active_nav": "model_evaluate",
        "datasets": list_eval_datasets(),
        "runs": runs,
    }
    return render(request, "admin/model_evaluate.html", context)


@login_required
def admin_model_evaluate_detail(request, run_id):
    """模型评估详情：展示单次评估的指标与最差若干行。"""
    if not _is_admin_user(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")
    run = get_object_or_404(EvaluationRun, id=run_id)
    context = {
        "active_nav": "model_evaluate",
        "run": run,
        "worst_errors": run.metrics.get("worst_errors", []) if run.metrics else [],
    }
    return render(request, "admin/model_evaluate_detail.html", context)


@login_required
def samples_manage(request):
    """样本管理页面：查看所有已识别的样本，支持统计、分页与删除。"""
    if request.method == "POST":
        record_id = request.POST.get("record_id", "").strip()
        if record_id:
            record = get_object_or_404(OCRRecord, id=record_id)
            if _is_admin_user(request.user) or record.user == request.user:
                record.delete()
                messages.success(request, "样本删除成功！")
            else:
                messages.error(request, "您没有删除该样本的权限。")
        return redirect("samples_manage")

    samples = OCRRecord.objects.select_related("user").all().order_by("-created_at")

    total = samples.count()
    compliant = samples.filter(is_compliant=True).count()
    non_compliant = total - compliant

    paginator = Paginator(samples, 10)
    page_obj = paginator.get_page(request.GET.get("page"))

    return render(request, "ocr_demo/samples_manage.html", {
        "samples": page_obj.object_list,
        "page_obj": page_obj,
        "total_samples": total,
        "compliant_samples": compliant,
        "non_compliant_samples": non_compliant,
    })


@login_required
def samples_export(request):
    """导出样本列表为 CSV 文件（学生端导出本人样本，管理端导出全部）。"""
    samples = OCRRecord.objects.select_related("user").all()
    if not _is_admin_user(request.user):
        samples = samples.filter(user=request.user)
    samples = samples.order_by("-created_at")

    response = HttpResponse(content_type="text/csv; charset=utf-8-sig")
    response["Content-Disposition"] = 'attachment; filename="samples.csv"'

    writer = csv.writer(response)
    writer.writerow(["ID", "用户名", "检测时间", "合规", "置信度", "缺失字段", "识别结果"])
    for s in samples:
        writer.writerow([
            s.id,
            s.user.username,
            timezone.localtime(s.created_at).strftime("%Y-%m-%d %H:%M:%S"),
            "合规" if s.is_compliant else "不合规",
            f"{s.confidence:.4f}" if s.confidence is not None else "",
            "、".join(s.missing_fields or []),
            s.result_text or "",
        ])
    return response


# 「OCR 微调模型」固定使用已训练好的模型目录（纯英文路径，见 memory：中文路径 C++ 推理打不开）
FINETUNED_MODEL_DIR = "C:/Users/35459/ocr_models/rec_finetuned"

# 训练工作目录与一键微调脚本位置
MODEL_TRAIN_DIR = os.path.join(settings.BASE_DIR, "model_train")
FINETUNE_REC = os.path.join(MODEL_TRAIN_DIR, "finetune_rec.py")

# 正在运行的训练子进程（job_id -> Popen），供停止训练使用
_training_processes = {}


def _job_log_path(job):
    return os.path.join(settings.MEDIA_ROOT, "training_logs", f"job_{job.id}.log")


def _job_run_dir(job):
    """任务独占的训练目录（对应 finetune_rec.py 的 --run-name job_<id>）。

    数据、标注切分、配置与 checkpoint 都在这里，任务之间互不覆盖。
    """
    return os.path.join(MODEL_TRAIN_DIR, "runs", f"job_{job.id}")


def _export_dataset_zip(dataset):
    """把数据集内已标注图片裁剪成 rec 训练集（crops + rec_gt.txt），打包成 ZIP。

    返回 (zip_path, line_count)。与 finetune_dataset_export 共用同一份导出逻辑。
    """
    images = dataset.images.filter(annotation_status__in=["annotated", "verified"])
    export_dir = os.path.join(settings.MEDIA_ROOT, "training_exports")
    os.makedirs(export_dir, exist_ok=True)
    zip_path = os.path.join(export_dir, f"rec_dataset_{dataset.id}.zip")

    gt_lines = []
    idx = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            full_path = os.path.join(settings.MEDIA_ROOT, img.image.name)
            try:
                pil = Image.open(full_path)
            except Exception:
                continue
            for line in (img.annotation_bbox or []):
                pts = line.get("points") or []
                text = (line.get("text") or "").strip()
                if len(pts) < 4 or not text:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                try:
                    crop = pil.crop((min(xs), min(ys), max(xs), max(ys)))
                except Exception:
                    continue
                crop_name = f"crops/{idx:06d}.jpg"
                crop_buf = io.BytesIO()
                crop.convert("RGB").save(crop_buf, "JPEG")
                zf.writestr(crop_name, crop_buf.getvalue())
                # 去掉制表符/换行，避免破坏「图片路径\t文字」格式
                gt_lines.append(f"{crop_name}\t{text.replace(chr(9), ' ').replace(chr(10), ' ')}")
                idx += 1
        zf.writestr("rec_gt.txt", "\n".join(gt_lines))
    return zip_path, idx


def _parse_train_log(log_path, total_epochs):
    """解析训练日志，提取曲线数据与进度。

    返回 (metrics, progress)。metrics 含 train_loss / train_acc / train_epoch / lr /
    val_acc / val_loss / val_epoch。其中 val_loss 由 1 - val_acc 派生（PaddleOCR rec
    的 eval 只输出 acc 与 norm_edit_dis，不计算验证损失，故用错误率近似）。
    """
    epochs, losses, lrs, train_accs, val_accs, val_epochs = [], [], [], [], [], []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # epoch: [1/40], global_step: 10, lr: 0.000003, acc: 0.765, ..., loss: 4.02
                m = re.search(r"epoch:\s*\[(\d+)/(\d+)\].*?loss:\s*([0-9.eE+-]+)", line)
                if m:
                    epochs.append(int(m.group(1)))
                    losses.append(float(m.group(3)))
                    mlr = re.search(r"lr:\s*([0-9.eE+-]+)", line)
                    if mlr:
                        lrs.append(float(mlr.group(1)))
                    macc = re.search(r"acc:\s*([0-9.eE+-]+)", line)
                    if macc:
                        train_accs.append(float(macc.group(1)))
                # cur metric, acc: 0.9xxxx, norm_edit_dis: ...
                m2 = re.search(r"cur metric,\s*acc:\s*([0-9.eE+-]+)", line)
                if m2:
                    val_accs.append(float(m2.group(1)))
                    val_epochs.append(epochs[-1] if epochs else 0)

    metrics = {
        "train_loss": losses,
        "train_acc": train_accs,
        "train_epoch": epochs,
        "lr": lrs,
        "val_acc": val_accs,
        "val_loss": [round(1 - a, 4) for a in val_accs],
        "val_epoch": val_epochs,
    }
    progress = round((epochs[-1] / total_epochs) * 100, 1) if epochs and total_epochs else 0
    return metrics, min(progress, 99.0)


def _parse_training_params(post):
    """从表单读取并校验训练超参数，返回规范化的 dict。"""

    def _int(v, default, lo, hi):
        try:
            return max(lo, min(hi, int(v)))
        except (TypeError, ValueError):
            return default

    def _float(v, default):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    optimizer = post.get("optimizer", "adam")
    image_size = str(post.get("image_size", "320"))
    base_model = post.get("base_model", "official")
    return {
        "epochs": _int(post.get("epochs"), 10, 1, 100),
        "batch_size": _int(post.get("batch_size"), 16, 1, 256),
        "learning_rate": _float(post.get("learning_rate"), 0.001),
        "optimizer": optimizer if optimizer in ("adam", "adamw", "momentum") else "adam",
        "image_size": image_size if image_size in ("320", "640", "1024") else "320",
        "freeze_backbone": post.get("freeze_backbone") == "on",
        "cosine_lr": post.get("cosine_lr") == "on",
        "base_model": base_model if base_model in ("official", "local") else "official",
    }


def _build_training_command(job):
    """根据任务字段拼出 finetune_rec.py 的命令。返回 (cmd, error)。"""
    zip_path, line_count = _export_dataset_zip(job.dataset)
    if line_count < 2:
        return None, "数据集内已标注样本太少（<2 行），请先修正标注再训练。"

    cmd = [
        sys.executable, FINETUNE_REC,
        "--dataset", zip_path,
        "--epochs", str(job.epochs),
        "--batch-size", str(job.batch_size),
        "--lr", str(job.learning_rate),
        "--optimizer", job.optimizer,
        "--image-size", str(job.image_size),
        "--base-model", job.base_model,
        "--cpu",
        "--workdir", MODEL_TRAIN_DIR,
        # 每个任务独占 runs/job_<id>/，避免并发训练互相覆盖图片、标注切分与 checkpoint
        "--run-name", os.path.basename(_job_run_dir(job)),
    ]
    if job.freeze_backbone:
        cmd.append("--freeze-backbone")
    if not job.cosine_lr:
        cmd.append("--no-cosine")
    return cmd, None


def _monitor_training(job, proc, log_path):
    """后台线程：训练期间周期性解析日志回填指标，进程结束后定终态。"""
    while proc.poll() is None:
        time.sleep(2)
        metrics, progress = _parse_train_log(log_path, job.epochs)
        TrainingJob.objects.filter(pk=job.pk).update(metrics=metrics, progress=progress)

    # 进程结束，做最终写入
    metrics, _ = _parse_train_log(log_path, job.epochs)
    job.refresh_from_db()
    if job.status == "stopped":
        return
    if proc.returncode == 0:
        job.status = "completed"
        job.progress = 100
        job.completed_at = timezone.now()
        # 记下本任务导出的推理模型目录，避免多任务共用输出目录时分不清模型来自哪次训练
        inference_dir = os.path.join(_job_run_dir(job), "output", "inference")
        if os.path.exists(os.path.join(inference_dir, "inference.pdmodel")):
            job.model_path = inference_dir
    else:
        job.status = "failed"
    job.metrics = metrics
    job.save()


def _launch_training(job):
    """导出数据 → 启动真实训练子进程 → 起监控线程。返回 (ok, error)。"""
    cmd, err = _build_training_command(job)
    if err:
        job.status = "failed"
        job.metrics = {"error": err}
        job.save()
        return False, err

    log_path = _job_log_path(job)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    job.status = "running"
    job.started_at = timezone.now()
    job.progress = 0
    job.log_file = f"training_logs/job_{job.id}.log"
    job.metrics = {"train_loss": [], "train_epoch": [], "lr": [], "val_acc": [], "val_epoch": []}
    job.save()

    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=MODEL_TRAIN_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    _training_processes[job.id] = proc

    def _run():
        try:
            _monitor_training(job, proc, log_path)
        finally:
            log_file.close()
            _training_processes.pop(job.id, None)

    threading.Thread(target=_run, daemon=True).start()
    return True, None


@login_required
def finetune_model(request):
    """模型微调首页"""
    # 获取当前用户的训练数据集和任务
    datasets = TrainingDataset.objects.filter(created_by=request.user).order_by("-created_at")[:5]
    jobs = _visible_jobs(request.user).order_by("-created_at")[:5]
    # 训练页「开始训练」用的数据集下拉（只列有图片的数据集）
    train_datasets = TrainingDataset.objects.filter(
        created_by=request.user, images__isnull=False
    ).distinct().order_by("-created_at")

    # 统计数据
    dataset_count = TrainingDataset.objects.filter(created_by=request.user).count()
    job_count = _visible_jobs(request.user).count()
    image_count = TrainingImage.objects.filter(dataset__created_by=request.user).count()

    context = {
        "datasets": datasets,
        "jobs": jobs,
        "train_datasets": train_datasets,
        "dataset_count": dataset_count,
        "job_count": job_count,
        "image_count": image_count,
    }
    return render(request, "ocr_demo/model_finetune.html", context)


@login_required
def finetune_set_model(request):
    """切换识别模型（默认/微调/DB检测/CRNN识别/自定义五种模式）。"""
    if request.method != "POST":
        return redirect("finetune_model")

    settings_obj = SystemSettings.load()
    choice = request.POST.get("model", "official")
    valid_modes = [m[0] for m in SystemSettings.REC_MODEL_CHOICES]
    if choice not in valid_modes:
        choice = "official"

    custom_dir = (request.POST.get("custom_model_dir") or "").strip()

    if choice == "finetuned":
        # 需求2：直接用已训练好的微调模型，不再要求手填目录
        if not os.path.exists(os.path.join(FINETUNED_MODEL_DIR, "inference.pdmodel")):
            messages.error(request, f"未找到已训练微调模型：{FINETUNED_MODEL_DIR}")
            return redirect("finetune_model")
        settings_obj.finetuned_model_dir = FINETUNED_MODEL_DIR
    else:
        settings_obj.finetuned_model_dir = ""

    if choice == "custom":
        if not custom_dir:
            messages.error(request, "请填写自定义模型的推理目录（inference 目录）。")
            return redirect("finetune_model")
        if not os.path.exists(os.path.join(custom_dir, "inference.pdmodel")):
            messages.error(request, f"未在目录找到 inference.pdmodel：{custom_dir}")
            return redirect("finetune_model")
        settings_obj.custom_model_dir = custom_dir
    else:
        settings_obj.custom_model_dir = ""

    settings_obj.rec_model = choice
    settings_obj.save()

    # 清空缓存，让 get_ocr() 下次调用时重新加载模型
    global _ocr_instance, _ocr_key
    _ocr_instance = None
    _ocr_key = None

    label = dict(SystemSettings.REC_MODEL_CHOICES).get(choice, choice)
    messages.success(request, f"已切换识别模型：{label}")
    return redirect("finetune_model")


@login_required
def finetune_datasets(request):
    """训练数据集列表"""
    datasets = TrainingDataset.objects.filter(created_by=request.user).order_by("-created_at")
    
    # 搜索和筛选
    q = request.GET.get("q", "").strip()
    if q:
        datasets = datasets.filter(Q(name__icontains=q) | Q(description__icontains=q))
    
    dataset_type = request.GET.get("dataset_type", "").strip()
    if dataset_type:
        datasets = datasets.filter(dataset_type=dataset_type)
    
    # 分页
    paginator = Paginator(datasets, 10)
    page_obj = paginator.get_page(request.GET.get("page"))
    
    context = {
        "datasets": page_obj.object_list,
        "page_obj": page_obj,
        "q": q,
        "dataset_type": dataset_type,
    }
    return render(request, "ocr_demo/finetune_datasets.html", context)


@login_required
def finetune_dataset_create(request):
    """创建训练数据集"""
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        description = request.POST.get("description", "").strip()
        dataset_type = request.POST.get("dataset_type", "food_label")
        
        if not name:
            messages.error(request, "数据集名称不能为空！")
            return render(request, "ocr_demo/finetune_dataset_create.html")
        
        dataset = TrainingDataset.objects.create(
            name=name,
            description=description,
            dataset_type=dataset_type,
            created_by=request.user
        )
        
        messages.success(request, f"数据集「{name}」创建成功！")
        return redirect("finetune_dataset_detail", dataset_id=dataset.id)
    
    return render(request, "ocr_demo/finetune_dataset_create.html")


@login_required
def finetune_dataset_detail(request, dataset_id):
    """数据集详情页面"""
    dataset = get_object_or_404(TrainingDataset, id=dataset_id, created_by=request.user)
    images = dataset.images.all().order_by("-created_at")
    
    # 统计
    total = images.count()
    annotated = images.filter(annotation_status='annotated').count()
    verified = images.filter(annotation_status='verified').count()
    pending = images.filter(annotation_status='pending').count()
    pre_annotated = images.filter(annotation_status='pre_annotated').count()

    # 分页
    paginator = Paginator(images, 12)
    page_obj = paginator.get_page(request.GET.get("page"))

    context = {
        "dataset": dataset,
        "images": page_obj.object_list,
        "page_obj": page_obj,
        "total": total,
        "annotated": annotated,
        "verified": verified,
        "pending": pending,
        "pre_annotated": pre_annotated,
    }
    return render(request, "ocr_demo/finetune_dataset_detail.html", context)


@login_required
def finetune_image_upload(request, dataset_id):
    """上传训练图片"""
    dataset = get_object_or_404(TrainingDataset, id=dataset_id, created_by=request.user)
    
    if request.method == "POST":
        if "images" not in request.FILES:
            messages.error(request, "请选择要上传的图片文件！")
            return redirect("finetune_dataset_detail", dataset_id=dataset_id)
        
        uploaded_count = 0
        images = request.FILES.getlist("images")
        
        for image_file in images:
            if not image_file.content_type.startswith("image/"):
                continue
            
            file_name = f"{uuid.uuid4().hex}_{image_file.name}"
            file_path = default_storage.save(f"training_images/{datetime.now().strftime('%Y/%m/%d')}/{file_name}",
                                           ContentFile(image_file.read()))
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)

            # 自动预标注：跑 PaddleOCR，把识别到的文字行写入标注
            lines = _run_prelabel(full_path)
            annotation_text = "\n".join(l["text"] for l in lines)
            TrainingImage.objects.create(
                dataset=dataset,
                image=file_path,
                annotation_text=annotation_text,
                annotation_bbox=lines,
                structured_fields=extract_fields(annotation_text),
                annotation_status='pre_annotated' if lines else 'pending',
            )
            uploaded_count += 1
        
        messages.success(request, f"成功上传 {uploaded_count} 张图片！")
        if uploaded_count:
            log_action(request.user, "upload_image", f"上传 {uploaded_count} 张训练图片到「{dataset.name}」", request)
        return redirect("finetune_dataset_detail", dataset_id=dataset_id)
    
    return redirect("finetune_dataset_detail", dataset_id=dataset_id)


@login_required
def finetune_image_annotate(request, dataset_id, image_id):
    """标注训练图片：展示预标注框，支持逐行修正文字。"""
    dataset = get_object_or_404(TrainingDataset, id=dataset_id, created_by=request.user)
    image = get_object_or_404(TrainingImage, id=image_id, dataset=dataset)

    if request.method == "POST":
        # 逐行接收 text_0 / text_1 / ...，回写每条文字行
        lines = image.annotation_bbox or []
        new_lines = []
        for i, line in enumerate(lines):
            new_text = request.POST.get(f"text_{i}", "").strip()
            if new_text:
                line["text"] = new_text
                new_lines.append(line)

        # 结构化字段修正：接收 field_<字段名> 输入，回写品名/日期/类型/配料等
        structured = {}
        for name in extract_fields("").keys():
            structured[name] = request.POST.get(f"field_{name}", "").strip()

        if not new_lines:
            messages.error(request, "至少需要保留一条标注文字。")
            return redirect("finetune_image_annotate", dataset_id=dataset_id, image_id=image_id)

        image.annotation_bbox = new_lines
        image.annotation_text = "\n".join(l["text"] for l in new_lines)
        image.structured_fields = structured
        image.annotation_status = "annotated"
        image.annotated_by = request.user
        image.annotated_at = timezone.now()
        image.save()

        messages.success(request, "标注保存成功（含结构化字段修正）！")
        return redirect("finetune_dataset_detail", dataset_id=dataset_id)

    # 读取图片尺寸，供前端 SVG 叠加框按像素坐标绘制
    width = height = 0
    try:
        with Image.open(os.path.join(settings.MEDIA_ROOT, image.image.name)) as im:
            width, height = im.size
    except Exception:
        width = height = 0

    # 结构化字段：优先用已保存的修正值，否则从标注文本抽取；并补齐全部字段名
    structured = image.structured_fields or {}
    if not structured:
        structured = extract_fields(image.annotation_text or "")
    for name in extract_fields("").keys():
        structured.setdefault(name, "")

    context = {
        "dataset": dataset,
        "image": image,
        "lines": image.annotation_bbox or [],
        "structured_fields": structured,
        "width": width,
        "height": height,
    }
    return render(request, "ocr_demo/finetune_image_annotate.html", context)


@login_required
def finetune_dataset_prelabel(request, dataset_id):
    """批量对数据集内「待标注」图片补跑 OCR 预标注（供旧数据补标）。"""
    dataset = get_object_or_404(TrainingDataset, id=dataset_id, created_by=request.user)
    if request.method == "POST":
        images = dataset.images.filter(annotation_status="pending")
        count = 0
        for img in images:
            full_path = os.path.join(settings.MEDIA_ROOT, img.image.name)
            lines = _run_prelabel(full_path)
            img.annotation_text = "\n".join(l["text"] for l in lines)
            img.annotation_bbox = lines
            img.structured_fields = extract_fields(img.annotation_text)
            img.annotation_status = "pre_annotated" if lines else "pending"
            img.save()
            if lines:
                count += 1
        messages.success(request, f"预标注完成，共 {count} 张图片识别到文字。")
    return redirect("finetune_dataset_detail", dataset_id=dataset_id)


@login_required
def finetune_dataset_export(request, dataset_id):
    """导出数据集为 PaddleOCR rec 训练格式（裁剪文字行 + rec_gt.txt），打包 ZIP。"""
    dataset = get_object_or_404(TrainingDataset, id=dataset_id, created_by=request.user)
    zip_path, idx = _export_dataset_zip(dataset)

    if idx == 0:
        messages.warning(request, "暂无可导出的标注数据，请先修正标注。")
        return redirect("finetune_dataset_detail", dataset_id=dataset_id)

    with open(zip_path, "rb") as f:
        data = f.read()
    response = HttpResponse(data, content_type="application/zip")
    response["Content-Disposition"] = f'attachment; filename="rec_dataset_{dataset_id}.zip"'
    return response


def _visible_jobs(user):
    """返回当前用户可见的训练任务 queryset：管理员可见全部，学生仅见自己的。"""
    qs = TrainingJob.objects.all()
    if not is_admin_user(user):
        qs = qs.filter(created_by=user)
    return qs


@login_required
def finetune_jobs(request):
    """训练任务列表"""
    jobs = _visible_jobs(request.user).order_by("-created_at")
    
    # 筛选
    status = request.GET.get("status", "").strip()
    if status:
        jobs = jobs.filter(status=status)
    
    # 分页
    paginator = Paginator(jobs, 10)
    page_obj = paginator.get_page(request.GET.get("page"))
    
    context = {
        "jobs": page_obj.object_list,
        "page_obj": page_obj,
        "status": status,
    }
    return render(request, "ocr_demo/finetune_jobs.html", context)


@login_required
def finetune_job_create(request):
    """创建训练任务"""
    datasets = TrainingDataset.objects.filter(created_by=request.user, images__isnull=False).distinct()

    if request.method == "POST":
        dataset_id = request.POST.get("dataset_id")
        name = request.POST.get("name", "").strip()

        if not dataset_id or not name:
            messages.error(request, "请选择数据集并填写任务名称！")
            return render(request, "ocr_demo/finetune_job_create.html", {"datasets": datasets})

        dataset = get_object_or_404(TrainingDataset, id=dataset_id, created_by=request.user)
        params = _parse_training_params(request.POST)

        job = TrainingJob.objects.create(
            dataset=dataset,
            name=name,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            optimizer=params["optimizer"],
            image_size=params["image_size"],
            freeze_backbone=params["freeze_backbone"],
            cosine_lr=params["cosine_lr"],
            base_model=params["base_model"],
            created_by=request.user
        )

        messages.success(request, f"训练任务「{name}」创建成功！")
        return redirect("finetune_job_detail", job_id=job.id)

    return render(request, "ocr_demo/finetune_job_create.html", {"datasets": datasets})


@login_required
@ensure_csrf_cookie
def finetune_job_detail(request, job_id):
    """训练任务详情"""
    job = get_object_or_404(_visible_jobs(request.user), id=job_id)

    context = {
        "job": job,
    }
    return render(request, "ocr_demo/finetune_job_detail.html", context)


@login_required
@require_POST
def finetune_job_start(request, job_id):
    """开始训练任务（真实启动 PaddleOCR 微调子进程）"""
    job = get_object_or_404(_visible_jobs(request.user), id=job_id)

    if job.status not in ['pending', 'failed', 'stopped']:
        return JsonResponse({"success": False, "message": "任务当前状态无法启动"})

    ok, err = _launch_training(job)
    if not ok:
        return JsonResponse({"success": False, "message": err})

    return JsonResponse({"success": True, "message": "训练任务已启动"})


@login_required
@require_POST
def finetune_job_stop(request, job_id):
    """停止训练任务（终止子进程）"""
    job = get_object_or_404(_visible_jobs(request.user), id=job_id)

    if job.status != 'running':
        return JsonResponse({"success": False, "message": "任务未在运行中"})

    proc = _training_processes.pop(job.id, None)
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass

    job.status = 'stopped'
    job.save()

    return JsonResponse({"success": True, "message": "任务已停止"})


@login_required
def finetune_job_logs(request, job_id):
    """获取训练日志与曲线指标"""
    job = get_object_or_404(_visible_jobs(request.user), id=job_id)

    log_path = _job_log_path(job)
    logs = ""
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            logs = f.read()

    return JsonResponse({
        "logs": logs,
        "status": job.status,
        "progress": job.progress,
        "metrics": job.metrics or {},
    })


@login_required
@require_POST
def finetune_start_training(request):
    """模型微调页内联「开始训练」：创建任务并真实启动训练，返回 job_id。"""
    dataset_id = request.POST.get("dataset_id")
    name = (request.POST.get("name") or "").strip()

    if not dataset_id:
        return JsonResponse({"success": False, "message": "请选择训练数据集。"})

    dataset = get_object_or_404(TrainingDataset, id=dataset_id, created_by=request.user)
    if not name:
        name = f"{dataset.name}-微调任务"

    params = _parse_training_params(request.POST)
    job = TrainingJob.objects.create(
        dataset=dataset,
        name=name,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        optimizer=params["optimizer"],
        image_size=params["image_size"],
        freeze_backbone=params["freeze_backbone"],
        cosine_lr=params["cosine_lr"],
        base_model=params["base_model"],
        created_by=request.user
    )

    ok, err = _launch_training(job)
    if not ok:
        return JsonResponse({"success": False, "message": err})

    return JsonResponse({"success": True, "job_id": job.id})