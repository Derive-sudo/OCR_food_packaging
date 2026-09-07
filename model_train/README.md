# 识别模型（rec）微调说明

本目录用于把「自采数据集」微调成自定义的 PaddleOCR 识别模型。核心脚本是
[finetune_rec.py](finetune_rec.py)，它把下面 5 步一键串起来。

## 整体流程

```
上传包装图 → 自动 OCR 预标注 → 浏览器内逐行改字 → 导出 rec 训练集(ZIP)
                                                            │
                                                            ▼
                              python model_train/finetune_rec.py --dataset 导出.zip
                                                            │
                                                            ▼
                              output/rec_ppocr_v4/ 训练出的新识别模型
```

## 前置依赖

- Python 3.12（本机已装）
- `paddlepaddle`（已装 2.6.2）、`paddleocr`（已装 2.7.0.3，仅推断用）
- `git`（克隆 PaddleOCR 源码用）
- `pyyaml`（缺了脚本会提示，执行 `pip install pyyaml`）

## 用法

```bash
# 最小用法：训练导出的数据集
python model_train/finetune_rec.py --dataset <导出的rec数据集.zip>

# 常用参数
python model_train/finetune_rec.py \
    --dataset rec_dataset_1.zip \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --cpu \
    --val-ratio 0.2
```

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dataset` | 必填 | 导出的 rec 数据集 ZIP 或文件夹 |
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 64 | 单卡 batch size |
| `--lr` | 1e-4 | 学习率 |
| `--val-ratio` | 0.2 | 验证集占比 |
| `--cpu` | 关 | 用 CPU 训练（无 GPU 时加这个） |
| `--workdir` | 脚本目录 | 工作目录（源码/权重/输出都放这） |
| `--prepare-only` | 关 | 只准备环境/数据/配置，不启动训练 |

## 脚本做了什么

1. **克隆 PaddleOCR 源码**（`release/2.7` 分支）——`tools/train.py` 和官方配置只在源码里，pip 包没有。
2. **下载预训练权重** `ch_PP-OCRv4_rec_train.tar`，用其 `student/` 目录做初始化。
3. **解压数据 + 切分**：把 `rec_gt.txt` 按 8:2 切成 `rec_gt_train.txt` / `rec_gt_val.txt`。
4. **生成微调配置**：复制官方 `ch_PP-OCRv4_rec.yml`，覆盖字符字典、预训练路径、数据路径、batch、lr、epoch 等。
5. **启动训练**：`python tools/train.py -c <生成配置>`。

## 产物

- `output/rec_ppocr_v4/` — 训练输出的模型（含 `best_accuracy`、`latest` 等）
- `ch_PP-OCRv4_rec_finetune.yml` — 本次微调用的最终配置
- `PaddleOCR/` — 克隆的源码
- `pretrain_models/` — 预训练权重
- `data/` — 解压后的训练数据

## 关键提示

- **字符字典必须匹配预训练模型头**。脚本默认用 `ppocr_keys_v1.txt`（6623 字），与官方 PP-OCRv4 一致。若改成自定义字典，最后一层全连接层无法加载预训练权重，初期 acc=0 是正常现象，需要更多数据。
- **数据量**：若不改字典，建议 5000 条以上效果才稳定；数据少时可以先用默认字典「微调已有能力」，几百条也能让模型更贴近食品包装场景。
- **学习率**：默认配置按 8 卡设计（总 batch 1024）。单卡小 batch 建议：batch 64 → lr 5e-5~1e-4；batch 128 → lr 1e-4~2e-4。显存不够就降 batch、同步降 lr。
- **无 GPU**：加 `--cpu`，会很慢，仅适合小数据集验证流程是否跑通。

## 训练后的接入

训练完成后，把输出目录里的推理模型接入本系统「模型切换」功能即可（`ocr_demo` 中读取 `use_angle_cls` / `lang` 的地方，替换成自定义 rec 模型路径）。
