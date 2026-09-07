from django.contrib.auth.forms import UserCreationForm


class RegistrationForm(UserCreationForm):
    """注册表单：在 Django 内置注册表单基础上，统一添加 Bootstrap 样式。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"
