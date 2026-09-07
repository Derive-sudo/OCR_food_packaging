from django.db import models
from django.contrib.auth.models import User


class OCRRecord(models.Model):
    PACKAGING_CHOICES = [
        ("Box", "盒装"),
        ("Bottle", "瓶装"),
        ("Can", "罐装"),
        ("Label", "标签"),
        ("Bag", "袋装"),
        ("Cup", "杯装"),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="ocr_records",
        verbose_name="User"
    )
    packaging_category = models.CharField(
        max_length=20,
        choices=PACKAGING_CHOICES,
        blank=True,
        default="",
        verbose_name="包装分类",
    )
    image = models.ImageField(
        upload_to="ocr_images/",
        max_length=255,
        verbose_name="Uploaded Image"
    )
    result_text = models.TextField(
        blank=True,
        null=True,
        verbose_name="OCR Result Text"
    )
    confidence = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Confidence Score"
    )
    is_compliant = models.BooleanField(
        default=False,
        verbose_name="是否合规"
    )
    compliance_issues = models.TextField(
        blank=True,
        null=True,
        verbose_name="合规问题"
    )
    missing_fields = models.JSONField(
        default=list,
        blank=True,
        null=True,
        verbose_name="缺失字段"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Created At"
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name="Updated At"
    )

    class Meta:
        verbose_name = "OCR Result"
        verbose_name_plural = "OCR Results"
        ordering = ["-created_at"]

    def __str__(self):
        return f"OCR Result for {self.user.username} at {self.created_at}"


class TrainingDataset(models.Model):
    """训练数据集模型 - 用于存储标注好的训练样本"""
    
    DATASET_TYPES = [
        ('food_label', '食品标签'),
        ('general', '通用文字'),
        ('custom', '自定义'),
    ]
    
    name = models.CharField(max_length=100, verbose_name="数据集名称")
    description = models.TextField(blank=True, verbose_name="数据集描述")
    dataset_type = models.CharField(
        max_length=20,
        choices=DATASET_TYPES,
        default='food_label',
        verbose_name="数据集类型"
    )
    created_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="created_datasets",
        verbose_name="创建者"
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    is_active = models.BooleanField(default=True, verbose_name="是否激活")
    
    class Meta:
        verbose_name = "训练数据集"
        verbose_name_plural = "训练数据集"
        ordering = ["-created_at"]
    
    def __str__(self):
        return f"{self.name} - {self.get_dataset_type_display()}"


class TrainingImage(models.Model):
    """训练图片模型 - 每个训练数据集包含多张图片及其标注"""
    
    ANNOTATION_STATUS = [
        ('pending', '待标注'),
        ('pre_annotated', '待审核'),
        ('annotated', '已修正'),
        ('verified', '已审核'),
        ('rejected', '已驳回'),
    ]
    
    dataset = models.ForeignKey(
        TrainingDataset,
        on_delete=models.CASCADE,
        related_name="images",
        verbose_name="所属数据集"
    )
    image = models.ImageField(
        upload_to="training_images/",
        max_length=255,
        verbose_name="训练图片"
    )
    annotation_text = models.TextField(
        blank=True,
        null=True,
        verbose_name="标注文本"
    )
    annotation_bbox = models.JSONField(
        default=list,
        blank=True,
        null=True,
        verbose_name="标注边界框"
    )
    structured_fields = models.JSONField(
        default=dict,
        blank=True,
        null=True,
        verbose_name="结构化字段（品名/类型/日期/配料等）"
    )
    annotation_status = models.CharField(
        max_length=20,
        choices=ANNOTATION_STATUS,
        default='pending',
        verbose_name="标注状态"
    )
    annotated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="annotated_images",
        verbose_name="标注者"
    )
    annotated_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="标注时间"
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="上传时间")
    
    class Meta:
        verbose_name = "训练图片"
        verbose_name_plural = "训练图片"
        ordering = ["-created_at"]
    
    def __str__(self):
        return f"{self.dataset.name} - Image {self.id}"


class TrainingJob(models.Model):
    """训练任务模型 - 记录模型训练任务的状态"""

    JOB_STATUS = [
        ('pending', '等待中'),
        ('running', '运行中'),
        ('completed', '已完成'),
        ('failed', '失败'),
        ('stopped', '已停止'),
    ]

    OPTIMIZER_CHOICES = [
        ('adam', 'Adam'),
        ('adamw', 'AdamW'),
        ('momentum', 'Momentum (SGD)'),
    ]

    BASE_MODEL_CHOICES = [
        ('official', '官方 PP-OCRv4 rec 预训练'),
        ('local', '本地最新微调 checkpoint'),
    ]

    dataset = models.ForeignKey(
        TrainingDataset,
        on_delete=models.CASCADE,
        related_name="training_jobs",
        verbose_name="训练数据集"
    )
    name = models.CharField(max_length=100, verbose_name="训练任务名称")
    model_type = models.CharField(
        max_length=50,
        default='paddleocr',
        verbose_name="模型类型"
    )
    base_model = models.CharField(
        max_length=20,
        choices=BASE_MODEL_CHOICES,
        default='official',
        verbose_name="预训练起点"
    )
    optimizer = models.CharField(
        max_length=20,
        choices=OPTIMIZER_CHOICES,
        default='adam',
        verbose_name="优化器"
    )
    image_size = models.CharField(
        max_length=10,
        default='320',
        verbose_name="图片宽度"
    )
    freeze_backbone = models.BooleanField(
        default=False,
        verbose_name="冻结 Backbone"
    )
    cosine_lr = models.BooleanField(
        default=True,
        verbose_name="Cosine 学习率"
    )
    status = models.CharField(
        max_length=20,
        choices=JOB_STATUS,
        default='pending',
        verbose_name="训练状态"
    )
    progress = models.FloatField(
        default=0,
        verbose_name="训练进度（0-100）"
    )
    epochs = models.IntegerField(
        default=10,
        verbose_name="训练轮数"
    )
    batch_size = models.IntegerField(
        default=16,
        verbose_name="批次大小"
    )
    learning_rate = models.FloatField(
        default=0.001,
        verbose_name="学习率"
    )
    log_file = models.FileField(
        upload_to="training_logs/",
        max_length=255,
        blank=True,
        null=True,
        verbose_name="训练日志文件"
    )
    model_path = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        verbose_name="模型保存路径"
    )
    metrics = models.JSONField(
        default=dict,
        blank=True,
        null=True,
        verbose_name="训练指标"
    )
    started_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="开始时间"
    )
    completed_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="完成时间"
    )
    created_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="training_jobs",
        verbose_name="创建者"
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    
    class Meta:
        verbose_name = "训练任务"
        verbose_name_plural = "训练任务"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} - {self.get_status_display()}"


class LabelTemplate(models.Model):
    """合规标签模板：定义食品标签必须标注的字段集合。"""

    name = models.CharField(max_length=100, verbose_name="模板名称")
    description = models.TextField(blank=True, verbose_name="模板描述")
    is_active = models.BooleanField(default=False, verbose_name="是否启用")
    is_standard = models.BooleanField(
        default=False,
        verbose_name="国标模板",
        help_text="标记来自 GB 7718 标准的模板；用于「识别图片修正」的字段预标注，不影响识别端的合规判定",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        verbose_name = "标签模板"
        verbose_name_plural = "标签模板"
        ordering = ["-is_active", "-updated_at"]

    def __str__(self):
        return f"{self.name}{'（启用）' if self.is_active else ''}"


class LabelField(models.Model):
    """标签模板字段：字段名 + 关键词，用于合规检测与字段提取。"""

    template = models.ForeignKey(
        LabelTemplate,
        on_delete=models.CASCADE,
        related_name="fields",
        verbose_name="所属模板",
    )
    name = models.CharField(max_length=50, verbose_name="字段名称")
    keywords = models.CharField(
        max_length=255,
        verbose_name="关键词",
        help_text="逗号分隔的关键词",
    )
    order = models.IntegerField(default=0, verbose_name="排序")

    class Meta:
        verbose_name = "标签字段"
        verbose_name_plural = "标签字段"
        ordering = ["order", "id"]

    def keyword_list(self):
        """把逗号分隔的关键词拆成去空列表。"""
        return [k.strip() for k in self.keywords.split(",") if k.strip()]

    def __str__(self):
        return f"{self.template.name} - {self.name}"


class EvaluationRun(models.Model):
    """模型评估任务：记录一次 rec 识别模型的准确率评估结果。"""

    STATUS_CHOICES = [
        ("running", "评估中"),
        ("completed", "已完成"),
        ("failed", "失败"),
    ]

    data_path = models.CharField(max_length=255, verbose_name="评测数据文件")
    model_dir = models.CharField(
        max_length=255,
        blank=True,
        default="",
        verbose_name="推理模型目录（空=官方 PP-OCRv4）",
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="running",
        verbose_name="状态",
    )
    char_accuracy = models.FloatField(null=True, blank=True, verbose_name="字符准确率(%)")
    line_accuracy = models.FloatField(null=True, blank=True, verbose_name="整行准确率(%)")
    total_lines = models.IntegerField(default=0, verbose_name="总行数")
    exact_lines = models.IntegerField(default=0, verbose_name="整行一致数")
    metrics = models.JSONField(default=dict, blank=True, verbose_name="详细指标")
    error_message = models.TextField(blank=True, verbose_name="失败原因")
    created_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="evaluation_runs",
        verbose_name="创建者",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name="完成时间")

    class Meta:
        verbose_name = "模型评估"
        verbose_name_plural = "模型评估"
        ordering = ["-created_at"]

    def model_label(self):
        return self.model_dir if self.model_dir else "官方 PP-OCRv4"

    def __str__(self):
        return f"评估 #{self.id} - {self.get_status_display()}"