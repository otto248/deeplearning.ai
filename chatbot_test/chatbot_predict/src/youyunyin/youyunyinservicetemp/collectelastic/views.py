from django.shortcuts import render

import simplejson
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import unquote
# from scrapy01.scrapy01.main import processor

# Create your views here.

from collectelastic.ElasticSearchUtil import ElasticSearchUtil

def index(request):
    # savegoodsvo()
    return render(request, 'base.html',locals())

def collectdata(request):

    # url = request.POST.get('url')
    # url = request.GET.get('url')
    host = '10.3.5.61:9200'
    esAction = ElasticSearchUtil(host)
    esAction.collectdata()
    para = dict()
    # if request.method == 'POST':
    #     return processPost(request)
    # else:
    #     return processGet(request)
    message=""
    result = dict()
    if (message is None):
        # prc.go_processor(url, tenantId, taskid)
        result['issucess'] = "True"
        result['message'] = "任务正在执行"
    else:
        result['issucess'] = "False"
        result['message'] = message

    json = simplejson.dumps(result, sort_keys=True, indent='    ')
    # return render(request, 'base.html', locals())
    return HttpResponse(json)




