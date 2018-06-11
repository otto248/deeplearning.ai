#!/usr/bin/python
#encoding:utf-8


import time
from pymongo import MongoClient

class ModelManageService:

    def __init__(self):
        self.name=''

    def getLocateModelName(self):
        localname = time.strftime('%Y%m%d', time.localtime(time.time()))
        ssname = "ss_" + localname+".model"
        lrname = "lr_" + localname+".model"
        names = [ssname, lrname]
        print(localname)
        return names

    def getLastestModel(self):
        name = ''
        return

    def addModelName(self, ssname, lrname):
        ssname = ''

    def getMongoclient(self):
        if (self.col is None):
            self.dbserver = '172.20.12.80'
            # dbport = settings['MONGODB_PORT']
            dbname = 'nccloud-gateway-elasticsearch-' + "crawler"
            self.dbname = dbname
            collname = 'crawldetailapivo'
            client = MongoClient(self.dbserver)
            db = client.admin
            db.authenticate("ublinker", "nash2017")
            db = client[dbname]
            self.col = db[collname]