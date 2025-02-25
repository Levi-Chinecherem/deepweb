from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze/', views.analyze_deepfake, name='analyze_deepfake'),
    path('status/', views.process_status, name='process_status'),
]