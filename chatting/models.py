from django.conf import settings
from django.db import models
from django.utils import timezone


class User(models.Model):
    text = models.CharField(max_length=300)
    send_date = models.DateTimeField(default=timezone.now)

    def send(self):
        #self.send_date = timezone.now()
        self.save()

    def __str__(self):
        return self.text