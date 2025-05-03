from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    url = models.URLField(blank=True, null=True)
    is_fake = models.BooleanField(null=True)  # True for fake, False for real
    fake_probability = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
