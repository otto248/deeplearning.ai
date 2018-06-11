#!/usr/bin/python
#encoding:utf-8
from django.shortcuts import render

import simplejson
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import unquote
# from scrapy01.scrapy01.main import processor
from logisticservice.ModelManageService import ModelManageService
# Create your views here.

from collectelastic.ElasticSearchUtil import ElasticSearchUtil

def index(request):
    # savegoodsvo()
    return render(request, 'base.html',locals())

def collectdata(request):

    # url = request.POST.get('url')
    # url = request.GET.get('url')
    host = '10.3.5.61:9200'
    modelservice = ModelManageService()
    currentdata = modelservice.getCurrentData()
    name='elastic_data_'+currentdata
    result = dict()
    esAction = ElasticSearchUtil(host,name)
    result = esAction.processor()
    json = simplejson.dumps(result, sort_keys=True, indent='    ')
    # return render(request, 'base.html', locals())
    return HttpResponse(json)




