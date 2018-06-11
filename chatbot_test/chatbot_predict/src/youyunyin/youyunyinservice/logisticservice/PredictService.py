#!/usr/bin/python
#encoding:utf-8


# 模型加载
## 引入包
from sklearn.externals import joblib
from logisticservice.ModelManageService import ModelManageService
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import pandas as pd
import traceback
import json

class PredictService():
    def __init__(self):
        self.name=''

    def predict(self,datas):
        totaldata = []
        for hit in datas:
            data = hit
            tsdata = dict()
            bstotal = data["bstotal"]
            nettotal = data["nettotal"]
            txtotal = data['txtotal']
            operatype = data['operatype']
            totaldata.append([bstotal, nettotal, txtotal, operatype])

        df = pd.DataFrame(np.array(totaldata))
        # 获取模型
        modelservice = ModelManageService()
        names = modelservice.getLocateModelName()
        ssname=names[0]
        lrname = names[1]
        print(ssname)
        print(lrname)
        # ssname="ss.model"
        oss = joblib.load('ss_20180508.model')
        # lrname="lr.model"
        olr = joblib.load('lr_20180508.model')
        # 数据预测
        ## a. 预测数据格式化(归一化)
        X_test = oss.transform(df)  # 使用模型进行归一化操作
        ## b. 结果数据预测
        Y_predict = olr.predict(X_test)
        print(Y_predict)
        return Y_predict

    def processor(self,datas):
        result = dict()
        try:
            value = self.predict(datas)
            result['state'] = "True"
            value_serial = np.array(value)
            value_serial = pd.Series(value).to_json(orient='values')
            result['value'] =value_serial
        except Exception as e:
            message=str(e)
            result['state'] = "False"
            result['message'] = message

        return result


# service =PredictService()
# service.predict()