from django.db import models

# Create your models here.

class React(models.Model):
    
    review = models.CharField(max_length=100000) 
