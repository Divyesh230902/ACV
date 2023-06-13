from django.db import models

# Create your models here.
class image_storage(models.Model):

    img = models.ImageField(upload_to="imgs")
    date_time = models.DateTimeField(auto_now=True)
    fruit_output = models.CharField(max_length=20,null=True)

    def __str__(self):
        return self.fruit_output