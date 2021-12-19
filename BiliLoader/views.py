from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse_lazy

from model.BM25.core import bm25f

# Create your views here.

ctx = {}


def empty_redirect(request):
    return HttpResponseRedirect('/index')


def index(request):
    return render(request, 'index.html', context=ctx)


def predict(request):
    return render(request, 'predict.html', context=ctx)


def text_ir(request):
    keyword = request.GET.get('search_for', None)
    title_weight = request.GET.get('title_weight', None)
    if not title_weight:
        title_weight = '0.8'
    if not keyword:
        return render(request, 'text-ir.html', context=ctx)
    title_weight = float(title_weight)
    res = bm25f(keyword, v1=title_weight, v2=1 - title_weight)
    print(res)
    return render(request, 'text-ir.html', context=ctx)


def img_ir(request):
    return render(request, 'img-ir.html', context=ctx)


def credit(request):
    return render(request, 'credit.html', context=ctx)
