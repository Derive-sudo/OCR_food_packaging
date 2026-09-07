from django.apps import AppConfig


class UserAuthConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "user_auth"

    def ready(self):
        # 导入信号，确保创建 User 时自动生成 UserProfile
        import user_auth.signals  # noqa: F401
