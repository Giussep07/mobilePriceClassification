"""
URL configuration for mobilePriceClassification project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.contrib import admin
from django.urls import path
from login.views import login_view, logout_view
from home.views import index_home
from svm.views import index_svm
from predict.views import index_predict
from nb.views import index_nb
from models_versus.views import index_model_versus

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('', index_home, name='home'),
    path('svm', index_svm, name='svm'),
    path('nb', index_nb, name='nb'),
    path('svm_vs_nb', index_model_versus, name='svm_vs_nb'),
    path('predict', index_predict, name='predict'),
]
