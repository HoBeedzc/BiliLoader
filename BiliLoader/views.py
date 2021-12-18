from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse_lazy

# Create your views here.

ctx = {}


def empty_redirect(request):
    return HttpResponseRedirect('/index')


def index(request):
    return render(request, 'index.html', context=ctx)


def predict(request):
    return render(request, 'predict.html', context=ctx)


def text_ir(request):
    return render(request, 'text-ir.html', context=ctx)


def img_ir(request):
    return render(request, 'img-ir.html', context=ctx)


def credit(request):
    return render(request, 'credit.html', context=ctx)
