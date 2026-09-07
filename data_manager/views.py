from django.shortcuts import render
from django.contrib.auth.decorators import login_required


@login_required
def data_home(request):
    """数据管理首页"""
    return render(request, "data_home.html")
