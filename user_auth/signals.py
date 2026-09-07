from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import UserProfile


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, raw=False, **kwargs):
    """当 User 被创建（注册 / 命令行创建用户）时，自动创建对应的 UserProfile。

    raw=True 表示通过 loaddata 等导入 fixture，此时 fixture 已含 UserProfile，跳过以免重复创建。
    """
    if raw:
        return
    if created:
        # 超级用户默认赋予「管理员」角色，普通用户默认为「学生」
        role = (
            UserProfile.ROLE_ADMIN
            if instance.is_superuser
            else UserProfile.ROLE_STUDENT
        )
        UserProfile.objects.create(user=instance, role=role)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, raw=False, **kwargs):
    """当 User 被保存时，确保其 UserProfile 存在（用于兜底旧数据）。"""
    if raw:
        return
    UserProfile.objects.get_or_create(user=instance)
