from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse_lazy

from model.BM25.core import bm25f
from model.PictureIR.core import picIR
import pandas as pd
import os
from BiliLoader.settings import BASE_DIR

# Create your views here.

ctx = {}


def empty_redirect(request):
    return HttpResponseRedirect('/index')


def index(request):
    return render(request, 'index.html', context=ctx)


def predict(request):
    if request.method == "GET":
        return render(request, 'predict.html', context=ctx)
    picture_obj = request.FILES.get('cover')
    path = os.path.join(BASE_DIR, 'static/cache/', picture_obj.name)
    print(path)
    with open(path, 'wb') as f:
        for content in picture_obj.chunks():
            f.write(content)
    return render(request, 'predict.html', context=ctx)


def text_ir(request):
    keyword = request.GET.get('search_for', None)
    title_weight = request.GET.get('title_weight', None)
    limit = request.GET.get('limit', None)
    if not title_weight:
        title_weight = '0.8'
    if not keyword:
        return render(request, 'text-ir.html', context=ctx)
    title_weight = float(title_weight)
    res = bm25f(keyword, v1=title_weight, v2=1 - title_weight)
    print(res)
    if limit:
        res = res[:int(limit)]
    data = pd.read_csv(os.path.join(BASE_DIR, 'data/data_df.csv'), engine='python', encoding='utf-8')
    data = data[data['bvid'].isin([x[0] for x in res])].to_dict(orient='index')
    return render(request, 'text-ir.html', context={'res': data})


def img_ir(request):
    if request.method == "GET":
        return render(request, 'img-ir.html', context=ctx)
    limit = request.POST.get('limit', None)
    picture_obj = request.FILES.get('cover')
    path = os.path.join(BASE_DIR, 'static/cache/', picture_obj.name)
    with open(path, 'wb') as f:
        for content in picture_obj.chunks():
            f.write(content)
    if not limit:
        limit = '6'
    res = picIR(path, limit=limit)
    return render(request, 'img-ir.html', context={'res': res, 'input': path})


def credit(request):
    return render(request, 'credit.html', context=ctx)
