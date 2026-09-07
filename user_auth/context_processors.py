from .models import SystemSettings


def site_settings(request):
    """向所有模板注入全局系统设置（用于品牌名、系统名等）。"""
    return {"site_settings": SystemSettings.load()}
