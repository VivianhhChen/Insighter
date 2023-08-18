"""Myproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.conf.urls import url
from Insighter import views
from django.urls import path
from django.contrib import admin
# from django.conf.urls import include

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('admin/', admin.site.urls),
    # url(r'^', include('homepage.urls')),
    path('svc/', views.svc, name='svc'),
    path('svc/svc_prediction/', views.svc_prediction, name='svc_prediction'),
    path('lstm/', views.lstm, name='lstm'),
    path('random_forest/', views.random_forest, name='random_forest'),
    path('random_forest/random_forest_prediction', views.random_forest_prediction, name='random_forest_prediction'),
    path('lstm/lstm_prediction', views.lstm_prediction, name='lstm_prediction')
]
