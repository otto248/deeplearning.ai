#!/usr/bin/python
#encoding:utf-8

import json
from elasticsearch import Elasticsearch
from pymongo import MongoClient
import matplotlib as mpl
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
import pathlib

class KMeansMongoService:
    def __init__(self,filename):
        self.dbserver = '172.20.12.80'
        self.filename=filename
        # dbport = settings['MONGODB_PORT']
        dbname = 'nccloud-gateway-elasticsearch-' + "crawler"
        self.dbname = dbname
        collname = 'crawldetailapivo'
        client = MongoClient(self.dbserver)
        db = client.admin
        db.authenticate("ublinker", "nash2017")
        db = client[dbname]
        self.col = db[collname]

    def kmeansprocessor(self):
        host = '10.3.5.61:9200'
        esAction = self
        query = {"from": 0, "size": 10, 'query': {'match_all': {}}}
        # query = {'query': {'match': {'bsid':'k7kpreoh'}}}
        #   query={"query": {
        #   "bool": {
        #     "should": [
        #       { "match": { 'appid': 'clMnxIkYIB0838868687'}},
        #       { "match": { 'nodecode':'k7kpreoh'}}
        #     ]
        #   }
        # }}
        # requests.get('http://172.20.12.80:9200/dfndsfyfsr0835468931_201803/clMnxIkYIB0838868687_busi?page=1&size=10000000')
        aa = esAction.col.find(
            {'operatype': {'$nin': [0, 19]}, 'bstotal': {'$exists': 'false'}, 'nettotal': {'$exists': 'false'},
             'txtotal': {'$exists': 'false'}},
            {'nettotal': 1, 'operatype': 1, 'bstotal': 1, 'txtotal': 1}).batch_size(100000)
        df = None
        bstotal = None
        totaldata = []
        for hit in aa:
            data = hit
            tsdata = dict()
            bstotal = data["bstotal"]
            nettotal = data["nettotal"]
            txtotal = data['txtotal']
            operatype = data['operatype']
            totaldata.append([bstotal, nettotal, txtotal, operatype])

        df = pd.DataFrame(np.array(totaldata))
        data = df.as_matrix()

        km = KMeans(n_clusters=5)
        print(df)
        km.fit(df)
        y = km.fit_predict(data)
        print(y)
        ab = y.size
        bc = y.tolist()
        for i in range(0, ab):
            classresult = bc[i]
            totaldata[i].append(classresult)

        path = pathlib.Path(self.filename)
        if not(path.exists()):
            print("不存在下面路径文件"+self.filename)
            path.touch()

        dataframe = pd.DataFrame(np.array(totaldata))
        dataframe.to_csv(self.filename, index=False, sep=',')
        # dataframe.to_csv("C:\\Users\\wushzh\\Desktop\\rule\\text3.csv", index=False, sep=',')
        # np.savetxt('C:\\Users\\wushzh\\Desktop\\rule\\elastic2.data', np.array(totaldata))

        print("所有样本距离聚族中心点的总距离和：", km.inertia_)
        # 防止中文乱码属性的设置\n",
        mpl.rcParams['font.sans-serif'] = [u'SimHei']
        mpl.rcParams['axes.unicode_minus'] = False

        y_hat = km.predict(data)

        def expandBorder(a, b):
            d = (b - a) * 0.1
            return a - d, b + d

    def processor(self):
        result = dict()
        try:
            name = self.kmeansprocessor()
            result['state'] = "True"
            result['message'] = "训练数据kmeans聚类完成时间"
        except Exception as e:
            message = str(e)
            result['state'] = "False"
            result['message'] = message
            print(message)
        finally:
            print('训练数据OK')
        return result

