from django.db import models
from django.utils import timezone 
from django.contrib.auth.models import User 
from django.urls import reverse


class Post(models.Model):
    title = models.CharField(max_length=150)
    content=models.TextField()
    date_posted= models.DateTimeField(default=timezone.now) #auto_now_add =True
    author = models.ForeignKey(User, on_delete= models.CASCADE)

    def __str__(self):
        return self.title


    def get_absolute_url(self):
        return reverse('post-details', kwargs= {'pk': self.pk})

    
