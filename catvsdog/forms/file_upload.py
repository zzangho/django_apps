import django.forms as forms
from ..models import ImageExample

class UploadFileForm(forms.Form):
#    file = forms.FileField()
    image = forms.ImageField()