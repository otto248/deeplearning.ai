#!/usr/bin/python
#encoding:utf-8

import json
from elasticsearch import Elasticsearch
# from scrapy.conf import settings
from pymongo import MongoClient
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import pandas as pd
from sklearn.cluster import KMeans

class ElasticSearchUtil:
    def __init__(self, host,collname):
        self.host = host
        self.conn = Elasticsearch([self.host])
        self.dbserver = '172.20.9.29'
        # dbport = settings['MONGODB_PORT']
        dbname = 'nccloud-analysis-' + "youyunyin"
        self.dbname = dbname
        collname = 'elastic_data_20180515'
        client = MongoClient(self.dbserver)
        db = client.admin
        db.authenticate("ublinker", "nash2017")
        db = client[dbname]
        self.col = db[collname]

    def getElasticCon(self):
        self.conn = Elasticsearch([self.host])

    def __del__(self):
        self.close()

    def check(self):
        '''
        输出当前系统的ES信息
        :return:
        '''
        return self.conn.info()

    def insertDocument(self, index, type, body, id=None):
        '''
        插入一条数据body到指定的index、指定的type下;可指定Id,若不指定,ES会自动生成
        :param index: 待插入的index值
        :param type: 待插入的type值
        :param body: 待插入的数据 -> dict型
        :param id: 自定义Id值
        :return:
        '''
        return self.conn.index(index=index, doc_type=type, body=body, id=id)

    def insertDataFrame(self, index, type, dataFrame):
        '''
        批量插入接口;
        bulk接口所要求的数据列表结构为:[{{optionType}: {Condition}}, {data}]
        其中optionType可为index、delete、update
        Condition可设置每条数据所对应的index值和type值
        data为具体要插入/更新的单条数据
        :param index: 默认插入的index值
        :param type: 默认插入的type值
        :param dataFrame: 待插入数据集
        :return:
        '''
        dataList = dataFrame.to_dict(orient='records')
        insertHeadInfoList = [{"index": {}} for i in range(len(dataList))]
        temp = [dict] * (len(dataList) * 2)
        temp[::2] = insertHeadInfoList
        temp[1::2] = dataList
        try:
            return self.conn.bulk(index=index, doc_type=type, body=temp)
        except Exception as e:
            return str(e)

    def deleteDocById(self, index, type, id):
        '''
        删除指定index、type、id对应的数据
        :param index:
        :param type:
        :param id:
        :return:
        '''
        return self.conn.delete(index=index, doc_type=type, id=id)

    def deleteDocByQuery(self, index, query, type=None):
        '''
        删除idnex下符合条件query的所有数据
        :param index:
        :param query: 满足DSL语法格式
        :param type:
        :return:
        '''
        return self.conn.delete_by_query(index=index, body=query, doc_type=type)

    def deleteAllDocByIndex(self, index, type=None):
        '''
        删除指定index下的所有数据
        :param index:
        :return:
        '''
        try:
            query = {'query': {'match_all': {}}}
            return self.conn.delete_by_query(index=index, body=query, doc_type=type)
        except Exception as e:
            return str(e) + ' -> ' + index

    def searchDoc(self, index=None, type=None, body=None):
        '''
        查找index下所有符合条件的数据
        :param index:
        :param type:
        :param body: 筛选语句,符合DSL语法格式
        :return:
        '''
        # return self.conn.search(index = "dfndsfyfsr0835468931_201803", body = {"query": {"match_all": {}}})
        return self.conn.search(index=index, doc_type=type, body=body)

    def getDocById(self, index, type, id):
        '''
        获取指定index、type、id对应的数据
        :param index:
        :param type:
        :param id:
        :return:
        '''
        return self.conn.get(index=index, doc_type=type, id=id)

    def updateDocById(self, index, type, id, body=None):
        '''
        更新指定index、type、id所对应的数据
        :param index:
        :param type:
        :param id:
        :param body: 待更新的值
        :return:
        '''
        return self.conn.update(index=index, doc_type=type, id=id, body=body)


    def close(self):
     if self.conn is not None:
        try:
            self.conn.close()
        except Exception as e:
            pass
        finally:
            self.conn = None

    def collectdata(self):
        host = '10.3.5.61:9200'
        esAction = self
        query = {"from": 0, "size": 10, 'query': {'match_all': {}}}
        res = esAction.searchDoc('dfndsfyfsr0835468931_201803', 'clMnxIkYIB0838868687_busi', query)
        a = res['hits']
        a = a['total']
        n = int(a / 1000)
        for num in range(0, n):
            # page1 = str(2 * num - 1)
            query = {"from": num * 1000, "size": 1000, 'query': {'match_all': {}}}
            # query = {"from": 10000, "size": 100, 'query': {'match_all': {}}}
            res = esAction.searchDoc('dfndsfyfsr0835468931_201803', 'clMnxIkYIB0838868687_busi', query)
            if (num > 1700):
                break
            for hit in res['hits']['hits']:
                data = hit["_source"]
                esAction.col.insert(data)
                # print(hit["_source"])
        return  "同步数据"+num*1000

    def processor(self):
        result = dict()
        try:
            name = self.collectdata()
            result['state'] = "True"
            result['message'] = "模型保存完成，模型名称;" + name + "完成时间"
            # + time.localtime(time.time())
        except Exception as e:
            message = str(e)
            result['state'] = "False"
            result['message'] = message

        return result


