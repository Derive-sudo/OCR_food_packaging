#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
食品外包装 OCR 识别模型（rec）评测脚本

对导出的 rec 数据集（crops/ + rec_gt.txt / rec_gt_val.txt / rec_gt_train.txt），
分别用【官方 PP-OCRv4】和【微调后的识别模型】跑识别，统计：
    - 字符准确率（Char Accuracy）：1 − 编辑距离 / 最长字符串长度，逐行求平均
    - 整行准确率（Line Accuracy）：预测与真值完全一致的行数占比

用法：
    # 仅官方模型（拿基线）
    python model_train/evaluate_rec.py --data ./runs/<run_name>/rec_gt_val.txt

    # 官方 vs 微调对比
    python model_train/evaluate_rec.py --data ./runs/<run_name>/rec_gt_val.txt \
        --model ./runs/<run_name>/output/inference

说明：
    --data  接受 rec_gt*.txt 文件，或包含它的目录（自动优先取 rec_gt_val.txt）。
    --model 必须是「推理模型目录」（内含 inference.pdmodel / inference.pdiparams），
            由 finetune_rec.py 训练后自动导出，或用 PaddleOCR tools/export_model.py 导出。

输出（默认写到 --data 所在目录）：
    evaluate_rec_report.md    可读报告（供实训报告使用）
    evaluate_rec_report.json  原始指标
"""

import argparse
import json
import os
import sys
from datetime import datetime


def load_ocr(rec_model_dir=None):
    """加载 PaddleOCR 识别模型（用于 rec-only 评测）。

    rec_model_dir=None 使用官方 PP-OCRv4；否则加载指定目录下的微调推理模型。
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        sys.exit("[eval] 缺少 paddleocr，请先执行：pip install paddleocr")

    kwargs = dict(use_angle_cls=False, lang="ch", show_log=False)
    if rec_model_dir:
        kwargs["rec_model_dir"] = rec_model_dir
    return PaddleOCR(**kwargs)


def _extract_text(result):
    """从 rec-only 结果中健壮地提取第一段识别文本。

    不同 PaddleOCR 版本 rec-only 返回结构略有差异，这里做兼容解析。
    """
    if not result:
        return ""
    page = result[0] if isinstance(result, (list, tuple)) else result
    if not isinstance(page, (list, tuple)):
        return "" if page is None else str(page).strip()
    if not page:
        return ""
    first = page[0]
    if isinstance(first, str):
        return first.strip()
    if isinstance(first, (list, tuple)) and first:
        return str(first[0]).strip()
    return ""


def recognize_text(ocr, img_path):
    """对单张文字行小图跑 rec-only 识别，返回识别文本。"""
    result = ocr.ocr(img_path, det=False, rec=True, cls=False)
    return _extract_text(result)


def parse_gt(data_path):
    """定位并解析 rec_gt*.txt，返回 [(图片绝对路径, 真值文本), ...] 与 gt 文件路径。"""
    if os.path.isdir(data_path):
        found = None
        for name in ("rec_gt_val.txt", "rec_gt.txt", "rec_gt_train.txt"):
            candidate = os.path.join(data_path, name)
            if os.path.exists(candidate):
                found = candidate
                break
        if not found:
            sys.exit(f"[eval] 目录下未找到 rec_gt*.txt：{data_path}")
        data_path = found

    if not os.path.exists(data_path):
        sys.exit(f"[eval] 找不到标注文件：{data_path}")

    base = os.path.dirname(os.path.abspath(data_path))
    items = []
    with open(data_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n").rstrip("\r")
            if not ln.strip():
                continue
            if "\t" in ln:
                rel, text = ln.split("\t", 1)
            else:
                parts = ln.split(" ", 1)
                if len(parts) != 2:
                    continue
                rel, text = parts
            text = text.strip()
            if not text:
                continue
            img_path = os.path.join(base, rel.strip())
            if os.path.exists(img_path):
                items.append((img_path, text))
    return items, data_path


def levenshtein(a, b):
    """编辑距离（Levenshtein），用于衡量预测与真值的差异。"""
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ai == b[j - 1] else 1),
            )
        prev = cur
    return prev[n]


def evaluate(ocr, items, label):
    """对 items 逐行跑识别，返回指标 dict。"""
    total = 0
    sum_char_acc = 0.0
    exact = 0
    errors = []  # (char_acc, pred, gt, path)

    for img_path, gt in items:
        try:
            pred = recognize_text(ocr, img_path)
        except Exception:
            pred = ""
        total += 1
        if pred == gt:
            exact += 1
        dist = levenshtein(pred, gt)
        acc = 1.0 - dist / max(len(gt), len(pred), 1)
        sum_char_acc += acc
        if pred != gt:
            errors.append((acc, pred, gt, img_path))

    errors.sort(key=lambda x: x[0])  # 最差的排前面
    return {
        "label": label,
        "total_lines": total,
        "char_accuracy": (sum_char_acc / total * 100) if total else 0.0,
        "line_accuracy": (exact / total * 100) if total else 0.0,
        "exact_lines": exact,
        "worst_errors": [
            {"pred": p, "gt": g, "char_acc": round(a * 100, 2), "file": os.path.basename(f)}
            for a, p, g, f in errors[:10]
        ],
    }


def _md_escape(s):
    return s.replace("|", "\\|").replace("\n", " ")


def write_markdown(path, report):
    off = report["official"]
    fine = report["finetuned"]

    lines = []
    lines.append("# 食品外包装 OCR 识别模型评测报告")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 评测数据：`{os.path.basename(report['data_file'])}`（{off['total_lines']} 条文字行）")
    if report["model_dir"]:
        lines.append(f"- 微调模型：`{report['model_dir']}`")
    lines.append("")
    lines.append("## 指标说明")
    lines.append("- **字符准确率（Char Accuracy）**：1 − 编辑距离 / 最长字符串长度，逐行求平均，衡量单字识别正确率。")
    lines.append("- **整行准确率（Line Accuracy）**：识别文本与真值完全一致的行数占比，衡量整行识别正确率。")
    lines.append("")
    lines.append("## 结果对比")
    lines.append("")
    lines.append("| 指标 | 官方 PP-OCRv4 | 微调模型 | 提升 |")
    lines.append("| --- | --- | --- | --- |")

    if fine:
        char_delta = fine["char_accuracy"] - off["char_accuracy"]
        line_delta = fine["line_accuracy"] - off["line_accuracy"]
        lines.append(f"| 字符准确率 | {off['char_accuracy']:.2f}% | {fine['char_accuracy']:.2f}% | {char_delta:+.2f}% |")
        lines.append(f"| 整行准确率 | {off['line_accuracy']:.2f}% | {fine['line_accuracy']:.2f}% | {line_delta:+.2f}% |")
        lines.append(f"| 整行完全一致 | {off['exact_lines']}/{off['total_lines']} | {fine['exact_lines']}/{fine['total_lines']} | — |")
    else:
        lines.append(f"| 字符准确率 | {off['char_accuracy']:.2f}% | — | — |")
        lines.append(f"| 整行准确率 | {off['line_accuracy']:.2f}% | — | — |")
        lines.append(f"| 整行完全一致 | {off['exact_lines']}/{off['total_lines']} | — | — |")

    lines.append("")
    if fine:
        if fine["char_accuracy"] > off["char_accuracy"]:
            lines.append("> ✅ 微调后识别效果整体提升。")
        else:
            lines.append("> ⚠️ 微调模型暂未超过官方模型，建议检查数据量 / 训练轮数 / 学习率。")
        lines.append("")
        lines.append("## 微调模型识别最差的 10 行（可据此定向补充数据）")
        lines.append("")
        lines.append("| 真值 | 微调识别 | 字符准确率 |")
        lines.append("| --- | --- | --- |")
        for e in fine["worst_errors"]:
            lines.append(f"| {_md_escape(e['gt'])} | {_md_escape(e['pred'])} | {e['char_acc']}% |")
    else:
        lines.append("## 官方模型识别最差的 10 行")
        lines.append("")
        lines.append("| 真值 | 识别 | 字符准确率 |")
        lines.append("| --- | --- | --- |")
        for e in off["worst_errors"]:
            lines.append(f"| {_md_escape(e['gt'])} | {_md_escape(e['pred'])} | {e['char_acc']}% |")

    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="食品外包装 OCR 识别模型（rec）评测")
    parser.add_argument("--data", required=True, help="rec_gt*.txt 或包含它的目录")
    parser.add_argument("--model", default=None, help="微调后的推理模型目录（可选）")
    parser.add_argument("--out", default=None, help="报告输出目录（默认与 --data 同目录）")
    args = parser.parse_args()

    items, gt_path = parse_gt(args.data)
    if not items:
        sys.exit(f"[eval] 标注文件里没有可用的（图片 + 文本）样本：{gt_path}")
    print(f"[eval] 加载样本 {len(items)} 条：{gt_path}", flush=True)

    out_dir = os.path.abspath(args.out or os.path.dirname(gt_path))
    os.makedirs(out_dir, exist_ok=True)

    # 1) 官方模型
    print("[eval] 加载官方 PP-OCRv4 模型…", flush=True)
    official_ocr = load_ocr(None)
    print("[eval] 官方模型识别中…", flush=True)
    official_metrics = evaluate(official_ocr, items, "官方 PP-OCRv4")

    # 2) 微调模型（可选）
    fine_metrics = None
    if args.model:
        model_dir = os.path.abspath(args.model)
        if not os.path.exists(os.path.join(model_dir, "inference.pdmodel")):
            sys.exit(f"[eval] --model 必须是推理模型目录（内含 inference.pdmodel），请先导出：{model_dir}")
        print("[eval] 加载微调模型…", flush=True)
        fine_ocr = load_ocr(model_dir)
        print("[eval] 微调模型识别中…", flush=True)
        fine_metrics = evaluate(fine_ocr, items, "微调模型")

    # 3) 生成报告
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_file": gt_path,
        "model_dir": args.model,
        "official": official_metrics,
        "finetuned": fine_metrics,
    }
    json_path = os.path.join(out_dir, "evaluate_rec_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(out_dir, "evaluate_rec_report.md")
    write_markdown(md_path, report)

    print(f"[eval] 完成。报告：\n  {md_path}\n  {json_path}", flush=True)


if __name__ == "__main__":
    main()
