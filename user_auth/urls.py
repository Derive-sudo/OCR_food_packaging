from django.urls import path

from . import views

urlpatterns = [
    path("login/", views.user_login, name="login"),
    path("logout/", views.user_logout, name="logout"),
    path("register/", views.user_register, name="register"),
    path("", views.home, name="home"),
    path("dashboard/student/", views.student_dashboard, name="student_dashboard"),
    path("classes/", views.student_classes, name="student_classes"),
    path("dashboard/admin/", views.admin_dashboard, name="admin_dashboard"),
    path("dashboard/admin/students/", views.admin_students, name="admin_students"),
    path("dashboard/admin/classes/", views.admin_classes, name="admin_classes"),
    path("dashboard/admin/classes/<int:class_id>/", views.admin_class_detail, name="admin_class_detail"),
    path("dashboard/admin/join-requests/", views.admin_join_requests, name="admin_join_requests"),
    path("dashboard/admin/profile/", views.admin_profile, name="admin_profile"),
    path("dashboard/admin/logs/", views.admin_logs, name="admin_logs"),
    path("dashboard/admin/logs/export/", views.admin_logs_export, name="admin_logs_export"),
    path("dashboard/admin/settings/", views.admin_settings, name="admin_settings"),
]
