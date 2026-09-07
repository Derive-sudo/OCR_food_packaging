"""食品标签合规检测工具模块。"""

import re

from .models import LabelField, LabelTemplate

# 默认的食品标签必须标注字段及其常见关键词（作为种子数据与兜底）。
# 只要识别文本中出现任意一个关键词，即视为该字段已被标注。
# 顺序即展示顺序，覆盖 GB 7718《预包装食品标签通则》要求标注的核心信息。
DEFAULT_FIELDS = {
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


def _seed_default_fields(template):
    """为模板种入默认字段（仅用于新建默认模板时）。"""
    for i, (name, keywords) in enumerate(DEFAULT_FIELDS.items()):
        LabelField.objects.create(
            template=template,
            name=name,
            keywords=",".join(keywords),
            order=i,
        )


def get_active_template():
    """获取（必要时创建）当前启用的合规标签模板。"""
    template = LabelTemplate.objects.filter(is_active=True).first()
    if template is None:
        template = LabelTemplate.objects.order_by("id").first()
        if template is None:
            # 种子模板直接用 DEFAULT_FIELDS（即 GB 7718 必标字段），标为国标模板
            template = LabelTemplate.objects.create(
                name="通用食品标签",
                description="系统默认合规模板",
                is_standard=True,
            )
        template.is_active = True
        template.save()
        if not template.fields.exists():
            _seed_default_fields(template)
    return template


def _active_template_fields(template=None):
    """返回当前启用模板的字段列表：`[{"name": ..., "keywords": [...]}, ...]`。

    若启用模板的字段为空，则回退到 DEFAULT_FIELDS 兜底，避免「删除所有字段后合规检测失效」。
    """
    template = template or get_active_template()
    fields = list(template.fields.order_by("order", "id"))
    if not fields:
        return [{"name": name, "keywords": keywords} for name, keywords in DEFAULT_FIELDS.items()]
    return [{"name": f.name, "keywords": f.keyword_list()} for f in fields]


def get_standard_fields():
    """返回固定 GB 7718 必标字段列表，供识别端合规检测与字段展示使用。

    识别端用的模型已由标注数据集训练而来、能直接输出标签字段内容，因此识别阶段
    不再依赖可编辑标签模板，而是固定按 GB 7718《预包装食品标签通则》的必标字段判定。
    可编辑标签模板仅用于「识别图片修正」构建训练数据集。

    返回:
        list: [{"name": 字段名, "keywords": 关键词列表}, ...]
    """
    return [{"name": name, "keywords": keywords} for name, keywords in DEFAULT_FIELDS.items()]


# 生产日期等字段的常见日期格式
DATE_PATTERN = re.compile(r"(\d{4}\s*[年/.\-]\s*\d{1,2}\s*[月/.\-]\s*\d{1,2}\s*日?)")

# 有固定格式的字段：值必须匹配对应正则，否则视为「内容格式不符」。
# 按「字段名包含该词」匹配，因此自定义模板里叫「生产日期(必标)」也能命中。
VALUE_PATTERNS = {
    "生产日期": DATE_PATTERN,
    "净含量": re.compile(r"\d"),
    "生产许可证": re.compile(r"SC\s*\d{6,}", re.I),
    "产品标准": re.compile(r"(GB|Q/|NY|SB/|DB)\s*[\w./-]*\d", re.I),
}

# GB 7718 允许「生产日期见包装」这类写法，命中时豁免上面的格式校验
VALUE_EXEMPT = re.compile(
    r"见(包装|袋|瓶|盒|罐|标签|喷码|封口|生产日期|正面|背面|底部|顶部|另)"
)


def _strip_label(value, keywords):
    """去掉字段名/关键词前缀及常见分隔符，只保留字段值（优先匹配最长关键词）。"""
    v = value
    for k in sorted(keywords, key=len, reverse=True):
        if k in v:
            v = v[v.find(k) + len(k):]
            break
    return v.strip("：:， ,、\t ")


def _pick_value(lines, keywords, other_keywords, multiline=False):
    """定位字段并取出它的值，返回 (是否出现字段名, 值)。

    字段名与值分行印刷很常见（包装上「配料」独占一行、内容在下一行），
    因此本行取不到值时会顺延到后续行，直到遇到其它字段的关键词为止。
    """
    for i, line in enumerate(lines):
        if not any(k in line for k in keywords):
            continue
        head = _strip_label(line, keywords)
        parts = [head] if head else []
        limit = 8 if multiline else 1
        for nxt in lines[i + 1:]:
            if len(parts) >= limit or any(k in nxt for k in other_keywords):
                break
            if nxt:
                parts.append(nxt)
        return True, " ".join(parts).strip()
    return False, ""


def _judge_field(name, found, value):
    """判断字段是否算「已标注」，返回 (是否合格, 不合格原因)。"""
    if not found:
        return False, "未标注"
    if not value:
        return False, "只印了字段名、无内容"
    for key, pattern in VALUE_PATTERNS.items():
        if key in name and not pattern.search(value) and not VALUE_EXEMPT.search(value):
            return False, "内容格式不符"
    return True, ""


def analyze_label(text, fields=None):
    """按启用模板逐字段解析识别文本，返回每个字段的取值与判定结果。

    合规检测与字段提取共用本函数，避免出现「判定合规、但字段值全为空」的矛盾结果。

    「已标注」需同时满足三点：出现该字段的任一关键词、关键词后能取到非空的值、
    值符合该字段的格式要求（仅对生产日期、净含量、生产许可证编号、产品标准代号生效）。

    返回:
        list: [{"name": 字段名, "value": 取到的值, "ok": 是否合格, "reason": 不合格原因}, ...]
    """
    if fields is None:
        fields = _active_template_fields()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

    all_keywords = set()
    for field in fields:
        all_keywords.update(field["keywords"])

    results = []
    for field in fields:
        others = all_keywords - set(field["keywords"])
        # 配料表等可能占多行，需向后合并
        multiline = "配料" in field["name"]
        found, value = _pick_value(lines, field["keywords"], others, multiline)
        # 字段名含「日期」时，尽量只保留日期部分
        if value and "日期" in field["name"]:
            m = DATE_PATTERN.search(value)
            value = m.group(1) if m else value
        if len(value) > 200:
            value = value[:200] + "…"
        ok, reason = _judge_field(field["name"], found, value)
        results.append({"name": field["name"], "value": value, "ok": ok, "reason": reason})
    return results


def check_compliance(text, fields=None):
    """检查识别文本是否标注了全部必需字段。

    识别端默认按固定 GB 7718 必标字段判定（不依赖可编辑标签模板）；调用方可传入
    `fields` 改用指定字段列表。

    参数:
        text: OCR 识别结果文本
        fields: 可选，字段列表；缺省用 get_standard_fields()

    返回:
        dict: {
            "is_compliant": bool,   是否合规（所有必需字段均已标注）
            "missing_fields": list, 未达标的字段名
            "issues": str,          逐项说明未达标原因（合规时为空字符串）
        }
    """
    if fields is None:
        fields = get_standard_fields()
    if not text:
        return {
            "is_compliant": False,
            "missing_fields": [f["name"] for f in fields],
            "issues": "未识别到任何文字，无法进行合规检测",
        }

    results = analyze_label(text, fields)
    failed = [r for r in results if not r["ok"]]
    missing_fields = [r["name"] for r in failed]

    issues = ""
    if failed:
        issues = "以下必需标注字段未达标：" + "；".join(
            f"{r['name']}（{r['reason']}）" for r in failed
        )

    return {
        "is_compliant": not failed,
        "missing_fields": missing_fields,
        "issues": issues,
    }


def extract_fields(text, fields=None):
    """从 OCR 识别文本中抽取结构化字段。

    缺省按启用标签模板抽取（供「识别图片修正」构建训练数据集时预标注）；
    识别端需固定国标展示时传入 `fields=get_standard_fields()`。
    与 check_compliance 共用 analyze_label，保证「显示的值」与「合规判定」一致。
    返回 dict，键为字段名，值为识别到的文本片段（未识别到则为空字符串）。
    """
    return {r["name"]: r["value"] for r in analyze_label(text, fields)}
