from django.shortcuts import render

import simplejson
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import unquote
import json
# from scrapy01.scrapy01.main import processor

# Create your views here.

from logisticservice.LogisticServiceTest import LogisticServiceTest
from logisticservice.ModelManageService import ModelManageService
from youyunyin.youyunyinservice.logisticservice.PredictService import PredictService

def index(request):
    # savegoodsvo()
    return render(request, 'base.html',locals())


def logistictrain(request):

    host = '10.3.5.61:9200'
    modelservice = ModelManageService()
    names=modelservice.getLocateModelName()
    path='C:\\Users\\wushzh\\Desktop\\rule\\text2.csv'
    ssname = names[0]
    lrname = names[1]
    result = dict()
    esAction = LogisticServiceTest(path,ssname,lrname)
    result = esAction.processor()
    json = simplejson.dumps(result, sort_keys=True, indent='    ')
    # return render(request, 'base.html', locals())
    return HttpResponse(json)

@csrf_exempt
def predict(request):
    data = request.POST.get('data')
    import json
    data=json.loads(request.body.decode('utf-8'))
    result = dict()
    predictService=PredictService()
    result = predictService.processor(data['data'])
    json = simplejson.dumps(result, sort_keys=True, indent='    ')
    # return render(request, 'base.html', locals())
    return HttpResponse(json)




