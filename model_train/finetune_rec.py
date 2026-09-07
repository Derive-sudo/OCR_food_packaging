"""
PaddleOCR PP-OCRv4 识别模型（rec）微调脚本

把「导出 rec 训练集」得到的 ZIP 或文件夹，一键完成：
    1. 克隆 PaddleOCR 源码（训练工具在源码里，pip 包只有推断）
    2. 下载 PP-OCRv4 识别预训练权重
    3. 解压数据并切分 train / val
    4. 基于官方配置生成微调配置
    5. 启动 tools/train.py 开始训练
    6. 训练完成后导出推理模型（inference.pdmodel / inference.pdiparams）

每次训练在 model_train/runs/<run_name>/ 下独占一套目录：

    runs/<run_name>/
        data/                解压出的图片与标注（--dataset 传文件夹时直接引用源目录，不复制）
        rec_gt_train.txt     本次训练的切分
        rec_gt_val.txt
        config.yml           本次训练的配置
        output/              checkpoint（latest / best_accuracy）与导出的 inference/

这样并发训练不会互相覆盖图片、标注切分和 checkpoint——共用一套固定路径时，
后启动的任务会把先启动任务的同名 crops 覆盖掉，导致图文错配、loss 降不下来。

用法示例：
    python model_train/finetune_rec.py --dataset rec_dataset_1.zip --epochs 50 --batch-size 64
    python model_train/finetune_rec.py --dataset synth_data --run-name synth_v1 --cpu --no-aug

"""

import argparse
import glob
import os
import random
import re
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

PADDLEOCR_REPO = "https://github.com/PaddlePaddle/PaddleOCR.git"
PADDLEOCR_TAG = "release/2.7"
PRETRAINED_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar"
PRETRAINED_TAR = "ch_PP-OCRv4_rec_train.tar"
CONFIG_REL = os.path.join("configs", "rec", "PP-OCRv4", "ch_PP-OCRv4_rec.yml")
CHAR_DICT_REL = os.path.join("ppocr", "utils", "ppocr_keys_v1.txt")


def log(msg):
    print(f"[finetune] {msg}", flush=True)


def run(cmd, cwd=None):
    log("  $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_paddleocr(repo_dir):
    """克隆 PaddleOCR 源码（含 tools/train.py 和官方配置）。"""
    train_py = os.path.join(repo_dir, "tools", "train.py")
    if os.path.exists(train_py):
        log(f"PaddleOCR 源码已存在：{repo_dir}")
        return repo_dir
    log(f"克隆 PaddleOCR（{PADDLEOCR_TAG}）… 首次较慢，请耐心等待")
    parent = os.path.dirname(repo_dir)
    os.makedirs(parent, exist_ok=True)
    run(["git", "clone", "--depth", "1", "-b", PADDLEOCR_TAG, PADDLEOCR_REPO, repo_dir])
    return repo_dir


def download_pretrained(workdir):
    """下载并解压 PP-OCRv4 识别预训练权重，返回 student 权重目录。"""
    extract_dir = os.path.join(workdir, "pretrain_models")
    student_dir = os.path.join(extract_dir, "ch_PP-OCRv4_rec_train", "student")
    # tar 解压后是 student.pdparams 文件（不是 student/ 目录），按文件判断避免重复解压 92MB
    if os.path.exists(student_dir + ".pdparams"):
        log(f"预训练权重已存在：{student_dir}.pdparams")
        return student_dir
    os.makedirs(extract_dir, exist_ok=True)
    tar_path = os.path.join(extract_dir, PRETRAINED_TAR)
    if not os.path.exists(tar_path):
        log(f"下载预训练权重：{PRETRAINED_URL}")
        urllib.request.urlretrieve(PRETRAINED_URL, tar_path)
    log("解压预训练权重…")
    with tarfile.open(tar_path) as tar:
        tar.extractall(extract_dir)
    return student_dir


def prepare_data(dataset_path, run_dir, val_ratio):
    """解压（如需）数据集，切分 train/val，返回 (data_dir, train_gt, val_gt)。

    ZIP 解压到本次 run 独占的 data/；传文件夹时只读引用源目录、不复制。
    两种情况下切分文件都写在 run 目录里，因此并发训练不会覆盖彼此的图片与标注。
    """
    if dataset_path.endswith(".zip"):
        data_dir = os.path.join(run_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        log(f"解压数据集：{dataset_path} → {data_dir}")
        with zipfile.ZipFile(dataset_path) as zf:
            zf.extractall(data_dir)
    else:
        # 传入的是文件夹：直接引用（只读，不往里写切分文件）
        data_dir = os.path.abspath(dataset_path)

    gt_path = os.path.join(data_dir, "rec_gt.txt")
    if not os.path.exists(gt_path):
        sys.exit(f"[finetune] 未在数据目录找到 rec_gt.txt：{data_dir}")

    with open(gt_path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if len(lines) < 2:
        sys.exit("[finetune] 标注样本太少（<2 行），请先修正并导出更多数据。")

    random.shuffle(lines)
    n_val = max(1, int(len(lines) * val_ratio))
    val_lines, train_lines = lines[:n_val], lines[n_val:]

    train_gt = os.path.join(run_dir, "rec_gt_train.txt")
    val_gt = os.path.join(run_dir, "rec_gt_val.txt")
    with open(train_gt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))
    with open(val_gt, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines))
    log(f"数据切分完成：train {len(train_lines)} 条 / val {len(val_lines)} 条")
    return data_dir, train_gt, val_gt


def find_local_checkpoint(workdir):
    """找最近一次训练产出的 checkpoint，返回不含扩展名的路径前缀（找不到返回 None）。

    跨所有 run 目录按修改时间取最新；同时兼容按 run 隔离改造之前的旧固定路径。
    """
    candidates = glob.glob(os.path.join(workdir, "runs", "*", "output", "latest.pdparams"))
    legacy = os.path.join(workdir, "output", "rec_ppocr_v4", "latest.pdparams")
    if os.path.exists(legacy):
        candidates.append(legacy)
    if not candidates:
        return None
    newest = max(candidates, key=os.path.getmtime)
    return newest[: -len(".pdparams")]


def generate_config(repo_dir, data_dir, train_gt, val_gt, run_dir, args):
    """复制官方 PP-OCRv4 rec 配置并覆盖关键字段，返回新配置路径。"""
    try:
        import yaml
    except ImportError:
        sys.exit("[finetune] 缺少 pyyaml，请先执行：pip install pyyaml")

    src = os.path.join(repo_dir, CONFIG_REL)
    with open(src, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    char_dict = os.path.join(repo_dir, CHAR_DICT_REL)
    if getattr(args, "base_model", "official") == "local":
        student_dir = find_local_checkpoint(args.workdir)
        if not student_dir:
            sys.exit("[finetune] --base-model local 但没找到任何本地 checkpoint"
                     "（runs/*/output/latest.pdparams），请先完成一次训练，或改用 --base-model official。")
        log(f"以本地 checkpoint 为训练起点：{student_dir}.pdparams")
    else:
        student_dir = os.path.join(args.workdir, "pretrain_models",
                                   "ch_PP-OCRv4_rec_train", "student")
    save_dir = os.path.join(run_dir, "output")
    use_gpu = not args.cpu
    # Windows 下 DataLoader 多进程易出错，统一置 0
    num_workers = 0 if os.name == "nt" else 4

    def set_nested(d, dotted, value):
        keys = dotted.split(".")
        cur = d
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value

    set_nested(cfg, "Global.character_dict_path", char_dict)
    set_nested(cfg, "Global.pretrained_model", student_dir)
    set_nested(cfg, "Global.epoch_num", args.epochs)
    set_nested(cfg, "Global.save_model_dir", save_dir)
    set_nested(cfg, "Global.use_gpu", use_gpu)
    set_nested(cfg, "Train.dataset.data_dir", data_dir + os.sep)
    set_nested(cfg, "Train.dataset.label_file_list", [train_gt])
    # rec 训练走 MultiScaleSampler，真实 batch 由 first_bs 决定；
    # batch_size_per_card 只在无 sampler 时（Eval）生效，所以两处都要改。
    set_nested(cfg, "Train.sampler.first_bs", args.batch_size)
    set_nested(cfg, "Train.loader.batch_size_per_card", args.batch_size)
    set_nested(cfg, "Train.loader.num_workers", num_workers)
    set_nested(cfg, "Optimizer.lr.learning_rate", args.lr)
    set_nested(cfg, "Eval.dataset.data_dir", data_dir + os.sep)
    set_nested(cfg, "Eval.dataset.label_file_list", [val_gt])
    set_nested(cfg, "Eval.loader.batch_size_per_card", args.batch_size)

    # 优化器：Adam / AdamW / Momentum（SGD 对应 Momentum）
    _OPT_NAMES = {"adam": "Adam", "adamw": "AdamW", "momentum": "Momentum"}
    set_nested(cfg, "Optimizer.name",
               _OPT_NAMES.get(getattr(args, "optimizer", "adam"), "Adam"))
    # 学习率调度：Cosine（默认）或 Piecewise 分段衰减
    if getattr(args, "no_cosine", False):
        set_nested(cfg, "Optimizer.lr.name", "Piecewise")
        set_nested(cfg, "Optimizer.lr.decay_epochs",
                   [max(1, int(args.epochs * 0.5)), max(2, int(args.epochs * 0.8))])
        set_nested(cfg, "Optimizer.lr.values", [args.lr, args.lr * 0.1, args.lr * 0.01])
    else:
        set_nested(cfg, "Optimizer.lr.name", "Cosine")

    # 图片尺寸：宽度 W 需与采样 scales / RecConAug / RecResizeImg 三处同步
    img_w = int(getattr(args, "image_size", 320))
    set_nested(cfg, "Train.sampler.scales", [[img_w, 32], [img_w, 48], [img_w, 64]])
    for t in cfg.get("Train", {}).get("dataset", {}).get("transforms", []):
        if isinstance(t, dict) and "RecConAug" in t:
            t["RecConAug"]["image_shape"] = [48, img_w, 3]
    for t in cfg.get("Eval", {}).get("dataset", {}).get("transforms", []):
        if isinstance(t, dict) and "RecResizeImg" in t:
            t["RecResizeImg"]["image_shape"] = [3, 48, img_w]

    # 冻结 Backbone（由 run_train.py 读取并 stop_gradient）
    set_nested(cfg, "Global.freeze_backbone", bool(getattr(args, "freeze_backbone", False)))
    # 让曲线点更早出现：每个 batch 打一次 loss，每个 epoch 做一次验证
    set_nested(cfg, "Global.print_batch_step", 1)
    set_nested(cfg, "Global.eval_batch_epoch", 1)

    if getattr(args, "no_aug", False):
        transforms = cfg.get("Train", {}).get("dataset", {}).get("transforms", [])
        cfg["Train"]["dataset"]["transforms"] = [
            t for t in transforms
            if not (isinstance(t, dict) and ("RecAug" in t or "RecConAug" in t))
        ]
        cfg["Train"]["dataset"].pop("ext_op_transform_idx", None)
        log("已禁用 RecAug / RecConAug 增强（--no-aug），训练将大幅提速。")

    out = os.path.join(run_dir, "config.yml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    log(f"微调配置已生成：{out}")
    return out


def export_inference_model(repo_dir, config_path, run_dir):
    """把训练好的 best 模型导出为 PaddleOCR 推理格式（inference.pdmodel/pdiparams）。

    训练产出的是训练态 checkpoint（model.pdparams），只有导出为推理格式后，
    才能被 evaluate_rec.py 或系统识别服务通过 rec_model_dir 加载。
    """
    best_dir = os.path.join(run_dir, "output", "best_accuracy")
    if not os.path.isdir(best_dir):
        log("未找到 best_accuracy 目录，跳过模型导出（训练可能未完成或未产生 best 模型）。")
        return None
    out_dir = os.path.join(run_dir, "output", "inference")
    export_cmd = [
        sys.executable, "tools/export_model.py",
        "-c", config_path,
        "-o", f"Global.pretrained_model={best_dir}",
        f"Global.save_inference_dir={out_dir}",
    ]
    log("导出推理模型（inference.pdmodel / inference.pdiparams）…")
    run(export_cmd, cwd=repo_dir)
    log(f"推理模型已导出：{out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR PP-OCRv4 识别模型微调")
    parser.add_argument("--dataset", required=True, help="导出的 rec 数据集 ZIP 或文件夹路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（默认 50）")
    parser.add_argument("--batch-size", type=int, default=64, help="单卡 batch size（默认 64）")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率（默认 1e-4，单卡小 batch 建议 1e-4~5e-5）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（默认 0.2）")
    parser.add_argument("--cpu", action="store_true", help="使用 CPU 训练（默认用 GPU）")
    parser.add_argument("--no-aug", action="store_true",
                        help="禁用 RecAug/RecConAug 几何增强（CPU 训练大幅提速，合成数据建议开启）")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "momentum"], default="adam",
                        help="优化器（默认 adam；SGD 对应 momentum）")
    parser.add_argument("--image-size", type=int, choices=[320, 640, 1024], default=320,
                        help="训练图片宽度（默认 320，需与采样/增强/Resize 三处一致）")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="冻结 Backbone，只训练 Head（迁移学习常用）")
    parser.add_argument("--no-cosine", action="store_true",
                        help="关闭 Cosine 学习率衰减，改用分段 Piecewise 衰减")
    parser.add_argument("--base-model", choices=["official", "local"], default="official",
                        help="预训练起点：official=官方 PP-OCRv4 rec，local=最近一次训练的 runs/*/output/latest")
    parser.add_argument("--workdir", default=None, help="工作目录（默认脚本所在目录）")
    parser.add_argument("--run-name", default=None,
                        help="本次训练的名称，决定 runs/<run-name>/ 目录（默认按数据集名+时间戳生成）。"
                             "并发训练务必用不同名称，否则会互相覆盖数据与 checkpoint")
    parser.add_argument("--prepare-only", action="store_true", help="只准备环境/数据/配置，不启动训练")
    args = parser.parse_args()

    args.workdir = os.path.abspath(args.workdir or os.path.dirname(__file__))
    os.makedirs(args.workdir, exist_ok=True)

    # 每次训练独占一个 run 目录：数据、切分、配置、checkpoint 全部隔离
    run_name = args.run_name or "{}_{}".format(
        os.path.splitext(os.path.basename(args.dataset.rstrip("/\\")))[0],
        time.strftime("%Y%m%d_%H%M%S"),
    )
    run_name = re.sub(r"[^0-9A-Za-z_.\-]", "_", run_name) or "run"
    run_dir = os.path.join(args.workdir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    log(f"本次训练目录：{run_dir}")

    repo_dir = os.path.join(args.workdir, "PaddleOCR")
    clone_paddleocr(repo_dir)
    download_pretrained(args.workdir)
    data_dir, train_gt, val_gt = prepare_data(args.dataset, run_dir, args.val_ratio)
    config_path = generate_config(repo_dir, data_dir, train_gt, val_gt, run_dir, args)

    # 用自定义 runner（支持冻结 Backbone），替代 tools/train.py
    runner = os.path.join(args.workdir, "run_train.py")
    train_cmd = [sys.executable, runner, "-c", config_path]
    log("训练命令：")
    log("  " + " ".join(train_cmd) + f"   (cwd={repo_dir})")
    log(f"训练输出模型将保存在 {os.path.join(run_dir, 'output')}")

    if args.prepare_only:
        log("已按 --prepare-only 跳过训练。")
        return

    log("开始训练（Ctrl+C 可中断）…")
    run(train_cmd, cwd=repo_dir)

    inference_dir = export_inference_model(repo_dir, config_path, run_dir)
    if inference_dir:
        log(f"推理模型目录（填到系统「自定义模型」处即可使用）：{inference_dir}")
        log(f"完成。可用它做评测：python model_train/evaluate_rec.py "
            f"--data {val_gt} --model {inference_dir}")


if __name__ == "__main__":
    main()
