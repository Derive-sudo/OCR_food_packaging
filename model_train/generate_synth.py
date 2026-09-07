#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
食品包装文字行 合成数据生成器（rec 训练数据）

用 Windows 中文字体，把食品包装高频文字（商品名 / 配料 / 营养成分 / 日期 /
净含量 / 贮存说明等）渲染成「单行文字图」，直接输出 PaddleOCR rec 训练格式：

    crops/000000.jpg    <-- 单行文字图
    rec_gt.txt          <-- 每行「图片路径\t文字」，真值 100% 准确

为什么有效：官方 PP-OCRv4 已能识别通用中文，微调主要缺「食品包装高频词汇」的
覆盖。合成数据能以零标注成本补齐这部分，配合少量真实照片即可明显提升识别率。

用法：
    python model_train/generate_synth.py --count 3000 --out synth_data

生成的 synth_data 目录可直接喂给微调脚本：
    python model_train/finetune_rec.py --dataset synth_data --cpu
"""

import argparse
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Windows 常见中文字体（按存在与否自动筛选）
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑粗体
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
    r"C:\Windows\Fonts\simkai.ttf",   # 楷体
    r"C:\Windows\Fonts\simfang.ttf",  # 仿宋
]

# ============ 食品包装高频词汇库 ============
FIXED_PHRASES = [
    # —— 商品名 / 品类 ——
    "纯牛奶", "酸牛奶", "发酵乳", "风味发酵乳", "全脂牛奶", "低脂牛奶", "鲜牛奶",
    "脱脂乳粉", "全脂乳粉", "婴幼儿配方奶粉",
    "矿泉水", "纯净水", "天然水", "气泡水", "苏打水",
    "果汁饮料", "碳酸饮料", "茶饮料", "运动饮料", "植物蛋白饮料",
    "饼干", "曲奇", "威化饼干", "苏打饼干", "夹心饼干", "薯片", "薯条",
    "方便面", "泡面", "挂面", "意大利面", "拉面",
    "火腿肠", "香肠", "腊肠", "肉松", "午餐肉",
    "面包", "蛋糕", "吐司", "蛋黄派", "月饼", "麻薯",
    "巧克力", "糖果", "奶糖", "口香糖", "棒棒糖", "果冻",
    "食用油", "大豆油", "花生油", "菜籽油", "橄榄油", "调和油",
    "酱油", "生抽", "老抽", "蚝油", "陈醋", "白醋", "料酒", "芝麻油",
    "食盐", "味精", "白糖", "红糖", "冰糖", "蜂蜜", "淀粉",
    "大米", "面粉", "小麦粉", "玉米面", "燕麦片", "挂面",
    "罐头", "水果罐头", "八宝粥", "午餐肉罐头",
    "坚果", "瓜子", "花生", "核桃", "腰果", "开心果",
    "豆腐", "豆干", "腐竹", "火锅底料", "辣椒酱", "豆瓣酱", "番茄酱",
    # —— 配料表 ——
    "配料表", "配料", "配料：生牛乳", "配料：水、白砂糖、食用盐",
    "配料：小麦粉、食用植物油、白砂糖", "食品添加剂", "食用香精",
    "防腐剂", "膨松剂", "乳化剂", "增稠剂", "色素", "甜味剂", "酸度调节剂",
    "白砂糖", "食用盐", "植物油", "小麦粉", "淀粉", "可可粉", "乳粉",
    "鸡蛋", "奶油", "麦芽糖", "山梨酸钾", "柠檬酸", "碳酸氢钠",
    # —— 营养成分 ——
    "营养成分表", "能量", "蛋白质", "脂肪", "碳水化合物", "钠", "钙",
    "膳食纤维", "维生素C", "每100克", "每100毫升", "每份", "营养素参考值",
    # —— 生产 / 合规信息 ——
    "生产日期", "保质期", "保质期至", "见包装", "见瓶盖", "见罐底",
    "生产许可证编号", "执行标准", "产品标准号", "食品生产许可证",
    "产地", "生产商", "制造商", "生产者", "地址", "电话",
    "贮存条件", "储存条件", "贮存方法", "保存方法",
    "请置于阴凉干燥处保存", "避光保存", "开封后请尽快食用", "开封后请冷藏保存",
    "阴凉干燥处", "常温保存", "冷藏保存", "置于阴凉干燥处",
    # —— 净含量 / 口味 / 宣称 ——
    "净含量", "净重", "规格",
    "非转基因", "无添加", "原味", "香辣味", "麻辣味", "烧烤味", "番茄味", "海苔味",
]

# 配料组合词（随机拼接生成「配料：…」长句）
INGREDIENT_WORDS = [
    "水", "白砂糖", "食用盐", "小麦粉", "植物油", "淀粉", "乳粉", "可可粉",
    "鸡蛋", "奶油", "麦芽糖", "山梨酸钾", "柠檬酸", "碳酸氢钠", "维生素C",
    "食用香精", "食品添加剂", "脱脂乳粉", "大豆油", "果葡糖浆",
]

# 营养成分词与单位
NUTRIENTS = [
    ("能量", "千焦"), ("蛋白质", "克"), ("脂肪", "克"), ("碳水化合物", "克"),
    ("钠", "毫克"), ("钙", "毫克"), ("膳食纤维", "克"), ("维生素C", "毫克"),
]


def random_date():
    y = random.randint(2023, 2026)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    fmt = random.choice([
        f"{y}年{m:02d}月{d:02d}日",
        f"{y}/{m:02d}/{d:02d}",
        f"{y}-{m:02d}-{d:02d}",
        f"{y}.{m:02d}.{d:02d}",
    ])
    prefix = random.choice(["生产日期：", "生产日期:", "生产日期见包装 ", "保质期至:", "保质期至 ", "保质期:"])
    return prefix + fmt


def random_quantity():
    unit = random.choice(["克", "千克", "毫升", "升", "kg", "ml", "g", "L", "公斤"])
    if unit in ("克", "g"):
        val = str(random.choice([100, 200, 250, 300, 400, 500, 800, 1000, 1500]))
    elif unit in ("千克", "公斤", "kg"):
        val = str(random.choice([1, 2, 5, 10, 20]))
    elif unit in ("毫升", "ml"):
        val = str(random.choice([100, 200, 250, 300, 500, 550, 600, 1000, 1500]))
    else:
        val = str(random.choice([1, 2, 5, 10, 20]))
    prefix = random.choice(["净含量：", "净含量:", "净重:", "规格:", "净含量 "])
    return prefix + val + unit


def random_nutrition():
    name, unit = random.choice(NUTRIENTS)
    val = str(round(random.uniform(0.1, 2000), random.choice([0, 1])))
    return f"{name} {val}{unit}"


def random_ingredient():
    k = random.randint(3, 7)
    words = random.sample(INGREDIENT_WORDS, k)
    return "配料：" + "、".join(words)


def random_expiry():
    n = random.choice([3, 6, 9, 12, 18, 24])
    return random.choice([f"保质期：{n}个月", f"保质期{n}个月", f"保质期 {n} 个月"])


def sample_text():
    r = random.random()
    if r < 0.35:
        return random.choice(FIXED_PHRASES)
    if r < 0.55:
        return random_date()
    if r < 0.68:
        return random_quantity()
    if r < 0.82:
        return random_nutrition()
    if r < 0.92:
        return random_ingredient()
    return random_expiry()


def add_noise(img):
    """叠加轻微椒盐/高斯噪声，模拟拍摄噪点。"""
    try:
        import numpy as np
    except ImportError:
        return img
    arr = np.asarray(img).astype(np.int16)
    h, w = arr.shape[:2]
    mask = np.random.random((h, w)) < 0.004
    noise = np.random.randint(-35, 35, arr.shape)
    arr = arr + noise * mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_line(text, font_path, out_path):
    """把一段文字渲染成单行文字图。"""
    font_size = random.randint(30, 54)
    font = ImageFont.truetype(font_path, font_size)

    meas = Image.new("RGB", (4, 4))
    left, top, right, bottom = ImageDraw.Draw(meas).textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top

    pad = random.randint(8, 20)
    W = max(tw + pad * 2, 24)
    H = max(th + pad * 2, 24)

    bg = random.choice([
        (255, 255, 255), (252, 252, 252), (250, 250, 248),
        (248, 248, 245), (255, 254, 250), (244, 245, 244),
    ])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    fg = random.choice([
        (0, 0, 0), (15, 15, 15), (30, 30, 30),
        (45, 40, 35), (15, 25, 45), (50, 50, 48),
    ])
    d.text((pad - left, pad - top), text, font=font, fill=fg)

    if random.random() < 0.45:
        angle = random.uniform(-2.5, 2.5)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=bg)

    if random.random() < 0.12:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))

    if random.random() < 0.5:
        img = add_noise(img)

    img.convert("RGB").save(out_path, "JPEG", quality=95)


def main():
    parser = argparse.ArgumentParser(description="食品包装文字行合成数据生成器")
    parser.add_argument("--count", type=int, default=3000, help="生成图片张数（默认 3000）")
    parser.add_argument("--out", default="synth_data", help="输出目录（默认 synth_data）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选，便于复现）")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    fonts = [p for p in FONT_CANDIDATES if os.path.exists(p)]
    if not fonts:
        sys.exit("[synth] 未找到中文字体，请确认 C:\\Windows\\Fonts 下存在 msyh.ttc 等字体。")
    print(f"[synth] 使用字体：{len(fonts)} 种", flush=True)

    out_dir = os.path.abspath(args.out)
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    gt_lines = []
    for idx in range(args.count):
        text = sample_text()
        font = random.choice(fonts)
        rel = f"crops/{idx:06d}.jpg"
        try:
            render_line(text, font, os.path.join(out_dir, rel))
        except Exception:
            continue
        gt_lines.append(f"{rel}\t{text}")
        if (idx + 1) % 500 == 0:
            print(f"[synth] 已生成 {idx + 1}/{args.count}", flush=True)

    gt_path = os.path.join(out_dir, "rec_gt.txt")
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(gt_lines))

    print(f"[synth] 完成：{len(gt_lines)} 张 → {gt_path}", flush=True)
    print("[synth] 可直接微调：python model_train/finetune_rec.py --dataset "
          f"{out_dir} --cpu", flush=True)


if __name__ == "__main__":
    main()
