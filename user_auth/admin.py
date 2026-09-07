from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User

from .models import UserProfile


class UserProfileInline(admin.StackedInline):
    """在 User 编辑页中内嵌展示 / 编辑 UserProfile。"""
    model = UserProfile
    can_delete = False
    verbose_name_plural = "用户资料"


class CustomUserAdmin(UserAdmin):
    """用带内嵌 UserProfile 的 admin 替换默认 UserAdmin。"""
    inlines = [UserProfileInline]


# 注销默认的 User 注册，改用带内嵌资料的 CustomUserAdmin
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)
admin.site.register(UserProfile)
