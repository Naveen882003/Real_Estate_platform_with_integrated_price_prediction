from django.urls import path
from . import views

urlpatterns=[
    path("",views.first_template,name="first_template"),
    path("result/",views.result,name="result"),
]