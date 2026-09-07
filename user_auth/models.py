from django.contrib.auth.models import User
from django.db import models


class UserProfile(models.Model):
    """用户资料模型。

    通过 OneToOne 与 Django 内置 User 关联，用于在不改动 User 的前提下
    扩展「角色」等额外字段。用户注册 / 创建时，由 signals.py 中的信号自动生成。
    """

    # 角色常量与选项（中文显示用于后台和前端展示）
    ROLE_STUDENT = "student"
    ROLE_ADMIN = "admin"
    ROLE_CHOICES = [
        (ROLE_STUDENT, "学生"),
        (ROLE_ADMIN, "管理员"),
    ]

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="profile",  # 可通过 user.profile 反向访问
        verbose_name="用户",
    )
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default=ROLE_STUDENT,
        verbose_name="角色",
    )

    class Meta:
        verbose_name = "用户资料"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"{self.user.username} - {self.get_role_display()}"

    def is_admin(self):
        """判断当前用户是否为管理员。"""
        return self.role == self.ROLE_ADMIN


class SystemSettings(models.Model):
    """系统全局设置（单例：主键固定为 1，通过 load() 获取）。"""

    system_name = models.CharField(
        max_length=100,
        default="食品外包装 OCR 实训平台",
        verbose_name="系统名称",
    )
    admin_email = models.EmailField(
        default="admin@foodpack.edu",
        verbose_name="管理员邮箱",
    )
    default_epochs = models.IntegerField(
        default=50,
        verbose_name="默认训练轮数",
    )
    compliance_threshold = models.IntegerField(
        default=85,
        verbose_name="合规置信度阈值(%)",
    )
    # 识别模型模式：默认 OCR / 微调 / DB 检测 / CRNN 识别 / 自定义
    REC_MODEL_CHOICES = [
        ("official", "默认 OCR（官方 PP-OCRv4）"),
        ("finetuned", "OCR 微调模型"),
        ("db_det", "DB 文本检测"),
        ("crnn_rec", "CRNN 文字识别"),
        ("custom", "自定义微调模型"),
    ]

    rec_model = models.CharField(
        max_length=20,
        choices=REC_MODEL_CHOICES,
        default="official",
        verbose_name="识别模型",
    )
    finetuned_model_dir = models.CharField(
        max_length=255,
        blank=True,
        default="",
        verbose_name="微调模型目录",
    )
    custom_model_dir = models.CharField(
        max_length=255,
        blank=True,
        default="",
        verbose_name="自定义模型目录",
    )
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        verbose_name = "系统设置"
        verbose_name_plural = verbose_name

    @classmethod
    def load(cls):
        """获取（必要时创建）唯一的系统设置对象。"""
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self):
        return self.system_name


class StudentClass(models.Model):
    """实训班级：将学生分组，用于班级管理与加入申请。"""

    name = models.CharField(max_length=100, verbose_name="班级名称")
    description = models.TextField(blank=True, verbose_name="班级描述")
    students = models.ManyToManyField(
        User,
        related_name="classes",
        blank=True,
        verbose_name="班级学生",
    )
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_classes",
        verbose_name="创建者",
    )
    is_active = models.BooleanField(default=True, verbose_name="是否启用")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "实训班级"
        verbose_name_plural = "实训班级"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class JoinRequest(models.Model):
    """加入申请：学生申请加入某个班级，管理员审核。"""

    STATUS_PENDING = "pending"
    STATUS_APPROVED = "approved"
    STATUS_REJECTED = "rejected"
    STATUS_CHOICES = [
        (STATUS_PENDING, "待审核"),
        (STATUS_APPROVED, "已通过"),
        (STATUS_REJECTED, "已拒绝"),
    ]

    student = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="join_requests",
        verbose_name="申请学生",
    )
    target_class = models.ForeignKey(
        StudentClass,
        on_delete=models.CASCADE,
        related_name="join_requests",
        verbose_name="目标班级",
    )
    message = models.TextField(blank=True, verbose_name="申请说明")
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
        verbose_name="状态",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="申请时间")
    reviewed_at = models.DateTimeField(null=True, blank=True, verbose_name="处理时间")
    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reviewed_join_requests",
        verbose_name="处理人",
    )

    class Meta:
        verbose_name = "加入申请"
        verbose_name_plural = "加入申请"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.student.username} 申请加入 {self.target_class.name}"


class AuditLog(models.Model):
    """操作日志：记录管理员的关键操作，便于审计。"""

    ACTION_CHOICES = [
        ("login", "登录"),
        ("logout", "登出"),
        ("create_student", "添加学生"),
        ("edit_student", "编辑学生"),
        ("reset_password", "重置密码"),
        ("delete_student", "删除学生"),
        ("create_class", "创建班级"),
        ("archive_class", "归档班级"),
        ("unarchive_class", "恢复班级"),
        ("approve_request", "通过申请"),
        ("reject_request", "拒绝申请"),
        ("save_settings", "保存设置"),
        ("export_records", "导出记录"),
        ("ocr_detect", "OCR 识别"),
        ("upload_image", "上传图片"),
        ("export_logs", "导出日志"),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name="audit_logs",
        verbose_name="操作人",
    )
    action = models.CharField(max_length=30, choices=ACTION_CHOICES, verbose_name="操作类型")
    detail = models.CharField(max_length=255, blank=True, verbose_name="操作详情")
    ip_address = models.GenericIPAddressField(null=True, blank=True, verbose_name="IP 地址")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="操作时间")

    class Meta:
        verbose_name = "操作日志"
        verbose_name_plural = "操作日志"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user} - {self.get_action_display()}"


def log_action(user, action, detail="", request=None):
    """记录一条操作日志（供各管理端视图调用）。"""
    ip = request.META.get("REMOTE_ADDR") if request is not None else None
    AuditLog.objects.create(
        user=user,
        action=action,
        detail=detail[:255],
        ip_address=ip,
    )
