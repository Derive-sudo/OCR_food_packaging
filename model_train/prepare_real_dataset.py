#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实采集图 → rec 训练集 一键转换脚本（按识别置信度三档分级）

把整张食品外包装照片转成识别模型（rec）微调所需的格式，并按识别置信度分成三档：

    1. 可信标注  conf ≥ 0.8             → rec_gt.txt（直接可用，无需改）
    2. 复核校正  0.5 ≤ conf < 0.8        → rec_gt_review.txt（打开逐条人工修正）
    3. 重新识别  conf < 0.5              → rec_gt_redo.txt（对裁出的小图用角度分类再识别一次，
                                            仍不合格则保留待人工处理或删除）

用法：
    python model_train/prepare_real_dataset.py --data model_train/data --out real_data

说明：
    - 三份文件都是「crops/xxxx.jpg <TAB> 文本 [<TAB> 置信度]」格式。
    - 训练前把 rec_gt.txt + 修正后的 rec_gt_review.txt + 处理后的 rec_gt_redo.txt
      合并成一份 rec_gt.txt 即可作为 Stage 2 训练集（--dataset real_data）。
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
from paddleocr import PaddleOCR


def log(msg):
    print(f"[real-data] {msg}", flush=True)


def read_img(path):
    """中文路径安全的读图（cv2.imread 在 Windows 中文路径下会失败）。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def write_img(path, img, ext=".jpg"):
    """中文路径安全的写图。"""
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def pad_box(points, img_w, img_h, pad=3):
    """把 4 点框转成 (x, y, w, h) 并做边界外扩与越界裁剪。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_w, x2 + pad)
    y2 = min(img_h, y2 + pad)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def extract_line(ocr, crop_path):
    """对单行小图再识别（开角度分类纠正旋转），返回 (text, conf)，失败返回 ("", 0.0)。"""
    try:
        result = ocr.ocr(crop_path, cls=True)
    except Exception:
        return "", 0.0
    page = result[0] if isinstance(result, (list, tuple)) else result
    if not page:
        return "", 0.0
    best = None
    for line in page:
        try:
            _, (text, conf) = line[0], line[1]
        except Exception:
            continue
        text = (text or "").strip()
        if not text:
            continue
        if best is None or conf > best[1]:
            best = (text, conf)
    return best if best else ("", 0.0)


def main():
    parser = argparse.ArgumentParser(description="真实图 → rec 训练集转换（三档置信度分级）")
    parser.add_argument("--data", default="dataset", help="真实整图目录（默认 ./dataset）")
    parser.add_argument("--out", default="real_data", help="输出目录（默认 ./real_data）")
    parser.add_argument("--trust-conf", type=float, default=0.8, help="可信标注阈值（默认 0.8）")
    parser.add_argument("--redo-conf", type=float, default=0.5, help="重新识别阈值（默认 0.5）")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data)
    if not os.path.isdir(data_dir):
        sys.exit(f"[real-data] 找不到图片目录：{data_dir}")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(exts))
    if not imgs:
        sys.exit(f"[real-data] 目录下没有图片：{data_dir}")
    log(f"共 {len(imgs)} 张整图，开始检测裁剪…")

    out_dir = os.path.abspath(args.out)
    crops_dir = os.path.join(out_dir, "crops")
    if os.path.isdir(crops_dir):
        shutil.rmtree(crops_dir)
    os.makedirs(crops_dir)

    ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)

    trusted = []   # conf ≥ trust-conf
    review = []    # redo-conf ≤ conf < trust-conf
    redo = []      # conf < redo-conf，待二次识别

    n_crops = 0

    for i, name in enumerate(imgs, 1):
        img_path = os.path.join(data_dir, name)
        try:
            result = ocr.ocr(img_path, cls=False)
        except Exception as e:
            log(f"跳过 {name}（识别异常：{e}）")
            continue

        page = result[0] if isinstance(result, (list, tuple)) else result
        if not page:
            continue

        img = read_img(img_path)
        if img is None:
            log(f"跳过 {name}（图片无法读取）")
            continue
        h, w = img.shape[:2]

        for line in page:
            # PaddleOCR 行结构：[[4个点], (文本, 置信度)]
            box, (text, conf) = line[0], line[1]
            text = (text or "").strip()
            if not text:
                continue

            x, y, bw, bh = pad_box(box, w, h, pad=3)
            if bw < 4 or bh < 4:
                continue

            crop = img[y:y + bh, x:x + bw]
            n_crops += 1
            rel = f"crops/{n_crops:06d}.jpg"
            write_img(os.path.join(out_dir, rel), crop)

            if conf >= args.trust_conf:
                trusted.append(f"{rel}\t{text}")
            elif conf >= args.redo_conf:
                review.append(f"{rel}\t{text}\t{conf:.2f}")
            else:
                redo.append((rel, text, conf))

        if i % 20 == 0:
            log(f"进度 {i}/{len(imgs)}，累计裁出 {n_crops} 行")

    # 二次识别：对 <0.5 的小图开角度分类再识别一次
    redo_final = []
    n_redo_promoted = 0
    if redo:
        log(f"对 {len(redo)} 条低置信度小图做二次识别（角度分类）…")
        for rel, old_text, old_conf in redo:
            crop_path = os.path.join(out_dir, rel)
            new_text, new_conf = extract_line(ocr, crop_path)
            if new_conf >= args.redo_conf:
                review.append(f"{rel}\t{new_text}\t{new_conf:.2f}")
                n_redo_promoted += 1
            else:
                keep_text = new_text if new_text else old_text
                redo_final.append(f"{rel}\t{keep_text}\t{new_conf:.2f}")

    # 写三份文件
    def write_lines(fname, lines):
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    write_lines("rec_gt.txt", trusted)
    write_lines("rec_gt_review.txt", review)
    write_lines("rec_gt_redo.txt", redo_final)

    log(f"完成，共裁出 {n_crops} 行 → {out_dir}/")
    log(f"  可信标注（≥{args.trust_conf}）: {len(trusted)} 行 → rec_gt.txt")
    log(f"  复核校正（{args.redo_conf}~{args.trust_conf}）: {len(review)} 行 → rec_gt_review.txt")
    log(f"  重新识别（<{args.redo_conf}）: {len(redo_final)} 行 → rec_gt_redo.txt"
        + (f"（二次识别提升 {n_redo_promoted} 条已并入复核）" if n_redo_promoted else ""))
    log("下一步：修正 rec_gt_review.txt 与 rec_gt_redo.txt，再把三份合并成一份 rec_gt.txt 用于训练")


if __name__ == "__main__":
    main()
