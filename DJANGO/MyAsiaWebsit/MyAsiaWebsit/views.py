
from django.contrib import admin
from django.urls import path
from django.views import View

from django.shortcuts import HttpResponse,render

def login(request):

    return render(request,'list.html')