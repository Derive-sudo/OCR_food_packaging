# OCR_food_packaging — 食品包装 OCR 识别与合规检测系统

基于 **Django 4.2 + PaddleOCR** 的食品包装文字识别与标签合规检测平台。通过 OCR 识别食品包装上的**配料、生产日期、保质期**等信息，并按 GB 7718《预包装食品标签通则》自动检测标签合规性，同时提供标签模板管理、模型微调训练与评测、学生/教师双角色管理等能力。

## 功能特性

- **食品包装文字识别**：识别包装上的配料表、生产日期、保质期、营养成分等关键信息
- **标签合规检测**：按 GB 7718 对识别结果进行关键词标注与合规判定（`is_compliant` / `compliance_issues`）
- **标签模板管理**：自定义标签模板与字段（`LabelTemplate` / `LabelField`），支持标准字段校验
- **图片优化与预处理**：识别前图片增强（`image_optimize`）
- **数据集管理与标注**：训练数据集、图片标注、样本管理
- **模型微调与评测**：基于 PaddleOCR 的识别模型微调训练（`TrainingJob`），训练曲线可视化与效果评测（`EvaluationRun`）
- **用户认证与权限**：学生 / 教师双角色，班级管理、加入申请审核、操作审计日志（`AuditLog`）
- **后台管理**：系统设置、模型配置、训练记录、评测详情等管理页面

## 项目结构

```
OCR_food_packaging/
├── manage.py                # Django 管理入口
├── requirements.txt         # 依赖清单
├── run_dev.bat              # 本地开发启动脚本（含环境变量，不入库）
├── food_ocr_system/         # Django 项目配置（settings / urls / wsgi）
├── ocr_demo/                # OCR 识别核心应用（识别、合规检测、数据集、微调）
├── user_auth/               # 用户认证应用（学生/教师、班级、审计）
├── data_manager/            # 数据管理应用
├── templates/               # 前端页面模板
├── docs/                    # 实训与评估文档
├── model_train/             # 模型训练脚本（自研，PaddleOCR 需另行安装）
│   ├── prepare_real_dataset.py   # 真实数据集准备
│   ├── generate_synth.py         # 合成数据生成
│   ├── finetune_rec.py           # 识别模型微调
│   ├── evaluate_rec.py           # 模型评测
│   └── ch_PP-OCRv4_rec_finetune.yml  # 微调配置
├── media/                   # 上传媒体（不入库）
├── static/                  # 静态资源
└── db.sqlite3               # SQLite 缓存数据库（默认使用 MySQL，不入库）
```

## 环境要求

- **Python 3.10 – 3.12**（paddlepaddle 对 Python 版本有硬性要求，3.8 无法安装新版）
- **MySQL 8 / 9**（需提前创建数据库，字符集 `utf8mb4`）
- 依赖见 `requirements.txt`：Django、paddlepaddle、paddleocr、opencv-python、Pillow、numpy、pymysql、PyYAML

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/Derive-sudo/OCR_food_packaging.git
cd OCR_food_packaging

# 2. 安装依赖
pip install -r requirements.txt

# 3. 创建 MySQL 数据库
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS food_ocr_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 4. 配置环境变量（Windows PowerShell 示例，Linux/macOS 用 export）
$env:DJANGO_SECRET_KEY = "请设置一个随机密钥"
$env:DB_NAME     = "food_ocr_system"
$env:DB_USER     = "root"
$env:DB_PASSWORD = "你的数据库密码"
$env:DB_HOST     = "127.0.0.1"
$env:DB_PORT     = "3306"

# 5. 数据库迁移
python manage.py migrate

# 6. 启动服务
python manage.py runserver
```

> Windows 本地开发也可以直接双击 `run_dev.bat`（脚本内置环境变量，自动完成 4–6 步）。

## 环境变量说明

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | 是 | Django 密钥，生产环境务必设置为随机值 |
| `DB_NAME` | 否 | 数据库名，默认 `food_ocr_system` |
| `DB_USER` | 否 | 数据库用户，默认 `root` |
| `DB_PASSWORD` | 是 | 数据库密码（未设置时无法连接数据库） |
| `DB_HOST` | 否 | 数据库地址，默认 `127.0.0.1` |
| `DB_PORT` | 否 | 数据库端口，默认 `3306` |

## 模型训练（可选）

数据准备与微调训练脚本位于 `model_train/`，需要先安装 PaddleOCR 源码与训练依赖：

1. `python model_train/prepare_real_dataset.py` — 准备真实数据集
2. `python model_train/generate_synth.py` — 生成合成训练数据
3. `python model_train/finetune_rec.py` — 微调识别模型（配置见 `ch_PP-OCRv4_rec_finetune.yml`）
4. `python model_train/evaluate_rec.py` — 评测模型效果

## 文档

- `docs/模型微调实训文档.md` — 模型微调实训流程
- `docs/模型基础评估报告模板.md` — 模型评估报告模板
- 《实训部署手册》《模型基础评估报告》等 Word 文档由本地维护，不入库

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
