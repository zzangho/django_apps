from django.db import models
from django import forms
from scipy import misc
import os

# Create your models here.
class ImageExample(models.Model):
    name = models.CharField(max_length=255)
    photo = models.ImageField('backup_catdog')




