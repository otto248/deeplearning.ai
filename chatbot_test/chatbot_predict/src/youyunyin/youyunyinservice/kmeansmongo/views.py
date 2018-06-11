from django.shortcuts import render

import simplejson
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import unquote
# from scrapy01.scrapy01.main import processor
from logisticservice.ModelManageService import ModelManageService
# Create your views here.

from kmeansmongo.KMeansMongoService import KMeansMongoService

def index(request):
    # savegoodsvo()
    return render(request, 'base.html',locals())

def kmeanstrain(request):

    # url = request.POST.get('url')
    # url = request.GET.get('url')
    # host = '10.3.5.61:9200'
    modelservice = ModelManageService()
    filename = modelservice.getLocateDataFileName()
    print(filename)
    filename="/opt/modules/youyunyin/youyunyinservice/templates/datafile/"+filename
    print(filename)
    result = dict()
    esAction = KMeansMongoService(filename)
    result = esAction.processor()
    para = dict()
    json = simplejson.dumps(result, sort_keys=True, indent='    ')
    # return render(request, 'base.html', locals())
    return HttpResponse(json)




