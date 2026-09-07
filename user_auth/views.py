import csv
from datetime import timedelta
from urllib.parse import urlencode

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.db.models import Avg, Count
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .forms import RegistrationForm
from .models import AuditLog, JoinRequest, StudentClass, SystemSettings, UserProfile, log_action
from ocr_demo.models import OCRRecord

def user_login(request):
    """登录视图：认证成功后交给 home 视图做角色跳转。"""
    # 已登录用户直接进入角色首页，避免重复登录
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f"欢迎回来，{user.username}！")
            return redirect("home")  
        else:
            messages.error(request, "用户名或密码错误，请重试。")

    return render(request, "login.html")


def user_logout(request):
    """登出视图：退出登录并返回登录页。"""
    logout(request)
    messages.info(request, "您已安全退出登录。")
    return redirect("login")


def user_register(request):
    """注册视图：支持学生和管理员自由注册"""
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        form = RegistrationForm(request.POST)
        role = request.POST.get("role", UserProfile.ROLE_STUDENT)  # 默认角色为学生

        if form.is_valid():
            user = form.save()
            if role == UserProfile.ROLE_ADMIN:
                user.profile.role = UserProfile.ROLE_ADMIN
                user.profile.save()

            login(request, user)
            messages.success(request, f"注册成功，欢迎{'管理员' if role == UserProfile.ROLE_ADMIN else '同学'}！")
            return redirect("home")
    else:
        form = RegistrationForm()

    return render(request, "register.html", {"form": form})


@login_required
def home(request):
    """首页路由：根据当前用户角色跳转到对应仪表盘。"""
    if _is_admin(request.user):
        return redirect("admin_dashboard")
    return redirect("student_dashboard")


@login_required
def student_dashboard(request):
    """学生首页仪表盘：展示个人检测概览、趋势与最近记录。"""
    my_records = OCRRecord.objects.filter(user=request.user)
    total = my_records.count()
    compliant = my_records.filter(is_compliant=True).count()
    non_compliant = total - compliant
    compliance_rate = (compliant / total * 100) if total else 0

    # 近 7 天检测趋势（用于折线图）
    today = timezone.localdate()
    trend = []
    max_v = 1
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        c = my_records.filter(created_at__date=d, is_compliant=True).count()
        n = my_records.filter(created_at__date=d, is_compliant=False).count()
        trend.append({"date": d.strftime("%m-%d"), "compliant": c, "non_compliant": n})
        max_v = max(max_v, c, n)

    # 生成 SVG 折线坐标（viewBox 0 0 700 220）
    W, H, PADX, PADY, n = 700, 220, 34, 22, len(trend)
    def _pts(key):
        pts = []
        for idx, item in enumerate(trend):
            x = PADX + (W - 2 * PADX) * idx / (n - 1)
            y = H - PADY - (H - 2 * PADY) * item[key] / max_v
            pts.append(f"{x:.0f},{y:.0f}")
        return " ".join(pts)

    context = {
        "total_records": total,
        "compliant_count": compliant,
        "non_compliant_count": non_compliant,
        "compliance_rate": compliance_rate,
        "non_compliance_rate": (100 - compliance_rate),
        "trend": trend,
        "compliant_points": _pts("compliant"),
        "non_compliant_points": _pts("non_compliant"),
        "recent_records": my_records.order_by("-created_at")[:6],
    }
    return render(request, "dashboard/student_dashboard.html", context)

@login_required
def model_finetune(request):
    """model finetune"""
    return render(request, "ocr_demo/finetune.html")

@login_required
def admin_dashboard(request):
    """管理员首页仪表盘：系统整体数据概览（仅管理员可访问）。"""
    if not _is_admin(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    total_users = User.objects.count()
    student_count = UserProfile.objects.filter(role=UserProfile.ROLE_STUDENT).count()
    total_records = OCRRecord.objects.count()
    compliant_count = OCRRecord.objects.filter(is_compliant=True).count()
    today_count = OCRRecord.objects.filter(created_at__date=timezone.localdate()).count()
    compliance_rate = (compliant_count / total_records * 100) if total_records else 0

    avg_confidence = OCRRecord.objects.filter(confidence__isnull=False).aggregate(
        avg=Avg("confidence")
    )["avg"]

    # 近 7 天检测趋势（用于柱状图）
    today = timezone.localdate()
    trend = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        trend.append({
            "date": d.strftime("%m-%d"),
            "count": OCRRecord.objects.filter(created_at__date=d).count(),
        })
    max_count = max([t["count"] for t in trend] or [1])

    context = {
        "active_nav": "dashboard",
        "total_users": total_users,
        "student_count": student_count,
        "total_records": total_records,
        "compliant_count": compliant_count,
        "today_count": today_count,
        "compliance_rate": compliance_rate,
        "avg_confidence": avg_confidence * 100 if avg_confidence is not None else 0,
        "trend": trend,
        "max_count": max_count,
        "recent_records": OCRRecord.objects.select_related("user").order_by("-created_at")[:8],
    }
    return render(request, "admin/dashboard.html", context)


@login_required
def admin_students(request):
    """管理员学生管理：查看所有学生及其检测统计（仅管理员可访问）。"""
    if not _is_admin(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    if request.method == "POST":
        action = request.POST.get("action", "create_student")

        if action == "reset_password":
            student = get_object_or_404(User, id=request.POST.get("student_id"))
            new_password = request.POST.get("new_password", "")
            if len(new_password) < 6:
                messages.error(request, "新密码至少 6 位。")
            else:
                student.set_password(new_password)
                student.save()
                log_action(request.user, "reset_password", f"重置学生「{student.username}」密码", request)
                messages.success(request, f"已重置「{student.username}」的密码。")
            return redirect("admin_students")

        if action == "delete_student":
            student = get_object_or_404(User, id=request.POST.get("student_id"))
            username = student.username
            student.delete()
            log_action(request.user, "delete_student", f"删除学生「{username}」", request)
            messages.success(request, f"学生「{username}」已删除。")
            return redirect("admin_students")

        if action == "edit_student":
            student = get_object_or_404(User, id=request.POST.get("student_id"))
            email = request.POST.get("email", "").strip()
            first_name = request.POST.get("first_name", "").strip()
            student.email = email
            student.first_name = first_name
            student.save()
            log_action(request.user, "edit_student", f"编辑学生「{student.username}」信息", request)
            messages.success(request, f"学生「{student.username}」信息已更新。")
            return redirect("admin_students")

        # 默认：创建学生
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        email = request.POST.get("email", "").strip()
        if not username or len(password) < 6:
            messages.error(request, "请填写用户名，且密码至少 6 位。")
        elif User.objects.filter(username=username).exists():
            messages.error(request, "该用户名已存在。")
        else:
            User.objects.create_user(username=username, password=password, email=email)
            # UserProfile 由信号自动创建（默认角色为学生）
            log_action(request.user, "create_student", f"添加学生「{username}」", request)
            messages.success(request, f"学生「{username}」已添加。")
        return redirect("admin_students")

    students = UserProfile.objects.filter(role=UserProfile.ROLE_STUDENT).select_related("user").order_by("-user__date_joined")
    paginator = Paginator(students, 10)
    page_obj = paginator.get_page(request.GET.get("page"))

    rows = []
    for profile in page_obj.object_list:
        u = profile.user
        recs = OCRRecord.objects.filter(user=u)
        total = recs.count()
        compliant = recs.filter(is_compliant=True).count()
        rows.append({
            "profile": profile,
            "user": u,
            "total_records": total,
            "compliant_count": compliant,
            "pass_rate": (compliant / total * 100) if total else 0,
        })

    context = {
        "active_nav": "students",
        "rows": rows,
        "page_obj": page_obj,
        "student_count": UserProfile.objects.filter(role=UserProfile.ROLE_STUDENT).count(),
        "total_records": OCRRecord.objects.count(),
        "total_users": User.objects.count(),
    }
    return render(request, "admin/students.html", context)


@login_required
def admin_classes(request):
    """管理员班级管理：查看/创建班级。"""
    if not _is_admin(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        description = request.POST.get("description", "").strip()
        if not name:
            messages.error(request, "班级名称不能为空。")
        else:
            cls = StudentClass.objects.create(name=name, description=description, created_by=request.user)
            log_action(request.user, "create_class", f"创建班级「{cls.name}」", request)
            messages.success(request, f"班级「{cls.name}」创建成功。")
        return redirect("admin_classes")

    classes = StudentClass.objects.annotate(student_count=Count("students")).order_by("-created_at")
    context = {
        "active_nav": "classes",
        "classes": classes,
        "student_count": UserProfile.objects.filter(role=UserProfile.ROLE_STUDENT).count(),
    }
    return render(request, "admin/classes.html", context)


@login_required
def admin_class_detail(request, class_id):
    """管理员班级详情：查看名册、添加/移除学生、启用/归档班级。"""
    if not _is_admin(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    cls = get_object_or_404(StudentClass, id=class_id)

    if request.method == "POST":
        action = request.POST.get("action", "")
        if action == "add_student":
            username = request.POST.get("username", "").strip()
            student = User.objects.filter(username=username, profile__role=UserProfile.ROLE_STUDENT).first()
            if student is None:
                messages.error(request, "未找到该学生（仅限学生账号）。")
            elif cls.students.filter(id=student.id).exists():
                messages.info(request, f"学生「{username}」已在班级中。")
            else:
                cls.students.add(student)
                messages.success(request, f"已将学生「{username}」加入班级。")
        elif action == "remove_student":
            student = cls.students.filter(id=request.POST.get("student_id")).first()
            if student:
                cls.students.remove(student)
                messages.success(request, f"已将「{student.username}」移出班级。")
        elif action == "toggle_active":
            cls.is_active = not cls.is_active
            cls.save()
            log_action(request.user, "archive_class" if not cls.is_active else "unarchive_class", f"班级「{cls.name}」", request)
            messages.success(request, "班级状态已更新。")
        return redirect("admin_class_detail", class_id=cls.id)

    roster = cls.students.select_related("profile").order_by("username")
    context = {
        "active_nav": "classes",
        "cls": cls,
        "roster": roster,
    }
    return render(request, "admin/class_detail.html", context)


@login_required
def admin_join_requests(request):
    """管理员加入申请管理：审核通过/拒绝学生加入班级的申请。"""
    if not _is_admin(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    if request.method == "POST":
        req = get_object_or_404(JoinRequest, id=request.POST.get("request_id"))
        decision = request.POST.get("decision")
        if decision == "approve":
            req.status = JoinRequest.STATUS_APPROVED
            req.target_class.students.add(req.student)
            log_action(request.user, "approve_request", f"通过 {req.student.username} 加入「{req.target_class.name}」", request)
            messages.success(request, f"已通过 {req.student.username} 的申请。")
        elif decision == "reject":
            req.status = JoinRequest.STATUS_REJECTED
            log_action(request.user, "reject_request", f"拒绝 {req.student.username} 加入「{req.target_class.name}」", request)
            messages.info(request, f"已拒绝 {req.student.username} 的申请。")
        req.reviewed_by = request.user
        req.reviewed_at = timezone.now()
        req.save()
        return redirect("admin_join_requests")

    requests_qs = JoinRequest.objects.select_related("student", "target_class").order_by("-created_at")
    status = request.GET.get("status", "")
    if status in ("pending", "approved", "rejected"):
        requests_qs = requests_qs.filter(status=status)
    paginator = Paginator(requests_qs, 10)
    page_obj = paginator.get_page(request.GET.get("page"))

    context = {
        "active_nav": "join_requests",
        "page_obj": page_obj,
        "requests": page_obj.object_list,
        "status": status,
        "pending_count": JoinRequest.objects.filter(status=JoinRequest.STATUS_PENDING).count(),
        "querystring": urlencode({"status": status}) if status else "",
    }
    return render(request, "admin/join_requests.html", context)


@login_required
def admin_profile(request):
    """管理员个人中心：查看资料并修改密码。"""
    if not _is_admin(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    if request.method == "POST":
        old_password = request.POST.get("old_password", "")
        new_password = request.POST.get("new_password", "")
        confirm_password = request.POST.get("confirm_password", "")
        if not request.user.check_password(old_password):
            messages.error(request, "当前密码不正确。")
        elif len(new_password) < 6:
            messages.error(request, "新密码至少需要 6 位。")
        elif new_password != confirm_password:
            messages.error(request, "两次输入的新密码不一致。")
        else:
            request.user.set_password(new_password)
            request.user.save()
            update_session_auth_hash(request, request.user)
            messages.success(request, "密码修改成功。")
            return redirect("admin_profile")

    context = {
        "active_nav": "profile",
        "profile_user": request.user,
        "student_count": UserProfile.objects.filter(role=UserProfile.ROLE_STUDENT).count(),
        "total_records": OCRRecord.objects.count(),
    }
    return render(request, "admin/profile.html", context)


@login_required
def admin_logs(request):
    """管理员操作日志：按类型筛选并分页展示。"""
    if not _is_admin(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    logs = AuditLog.objects.select_related("user").order_by("-created_at")
    action = request.GET.get("action", "")
    if action:
        logs = logs.filter(action=action)
    paginator = Paginator(logs, 15)
    page_obj = paginator.get_page(request.GET.get("page"))

    context = {
        "active_nav": "logs",
        "page_obj": page_obj,
        "logs": page_obj.object_list,
        "action": action,
        "action_choices": AuditLog.ACTION_CHOICES,
        "querystring": urlencode({"action": action}) if action else "",
    }
    return render(request, "admin/logs.html", context)


@login_required
def admin_logs_export(request):
    """导出操作日志为 CSV 文件（支持按类型筛选，仅管理员可访问）。"""
    if not _is_admin(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    logs = AuditLog.objects.select_related("user").order_by("-created_at")
    action = request.GET.get("action", "")
    if action:
        logs = logs.filter(action=action)

    log_action(request.user, "export_logs", f"导出操作日志（{'类型=' + action if action else '全部'}）", request)

    response = HttpResponse(content_type="text/csv; charset=utf-8-sig")
    response["Content-Disposition"] = 'attachment; filename="audit_logs.csv"'

    writer = csv.writer(response)
    writer.writerow(["ID", "操作时间", "操作人", "角色", "操作类型", "操作详情", "IP 地址"])
    for lg in logs:
        role = "管理员" if (lg.user and getattr(lg.user, "profile", None) and lg.user.profile.role == UserProfile.ROLE_ADMIN) else "学生"
        writer.writerow([
            lg.id,
            timezone.localtime(lg.created_at).strftime("%Y-%m-%d %H:%M:%S"),
            lg.user.username if lg.user else "",
            role,
            lg.get_action_display(),
            lg.detail or "",
            lg.ip_address or "",
        ])
    return response


@login_required
def admin_settings(request):
    """管理员系统设置：读取并保存全局配置。"""
    if not _is_admin(request.user):
        messages.error(request, "您没有访问管理员页面的权限。")
        return redirect("student_dashboard")

    settings_obj = SystemSettings.load()

    if request.method == "POST":
        system_name = request.POST.get("system_name", "").strip()
        admin_email = request.POST.get("admin_email", "").strip()
        if not system_name:
            messages.error(request, "系统名称不能为空。")
            return redirect("admin_settings")

        settings_obj.system_name = system_name
        settings_obj.admin_email = admin_email
        try:
            settings_obj.default_epochs = int(request.POST.get("default_epochs", settings_obj.default_epochs))
            settings_obj.compliance_threshold = int(request.POST.get("compliance_threshold", settings_obj.compliance_threshold))
        except (TypeError, ValueError):
            messages.error(request, "参数格式不正确。")
            return redirect("admin_settings")
        settings_obj.save()
        log_action(request.user, "save_settings", "更新系统设置", request)
        messages.success(request, "系统设置已保存。")
        return redirect("admin_settings")

    context = {"active_nav": "settings", "settings_obj": settings_obj}
    return render(request, "admin/settings.html", context)


@login_required
def student_classes(request):
    """学生端：浏览班级、申请加入、查看我的班级与申请进度。"""
    if _is_admin(request.user):
        return redirect("admin_classes")

    if request.method == "POST":
        cls = get_object_or_404(StudentClass, id=request.POST.get("class_id"), is_active=True)
        message = request.POST.get("message", "").strip()
        if cls.students.filter(id=request.user.id).exists():
            messages.info(request, "您已在该班级中。")
        elif JoinRequest.objects.filter(student=request.user, target_class=cls, status=JoinRequest.STATUS_PENDING).exists():
            messages.info(request, "您已提交过该班级的申请，请等待审核。")
        else:
            JoinRequest.objects.create(student=request.user, target_class=cls, message=message)
            messages.success(request, f"已提交加入「{cls.name}」的申请。")
        return redirect("student_classes")

    active_classes = StudentClass.objects.filter(is_active=True).annotate(student_count=Count("students")).order_by("-created_at")
    my_classes = request.user.classes.all()
    my_requests = JoinRequest.objects.filter(student=request.user).select_related("target_class").order_by("-created_at")

    context = {
        "active_classes": active_classes,
        "my_classes": my_classes,
        "my_requests": my_requests,
        "joined_ids": set(my_classes.values_list("id", flat=True)),
        "requested_ids": set(
            JoinRequest.objects.filter(student=request.user, status=JoinRequest.STATUS_PENDING)
            .values_list("target_class_id", flat=True)
        ),
    }
    return render(request, "dashboard/student_classes.html", context)


def _is_admin(user):
    """判断用户是否为管理员（兼容 profile 缺失的情况）。"""
    profile = getattr(user, "profile", None)
    return profile is not None and profile.role == UserProfile.ROLE_ADMIN
