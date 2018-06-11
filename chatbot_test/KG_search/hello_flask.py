# coding=gbk
#!/usr/bin/env python3

import flask
import json
import datetime
from flask import Response,request,jsonify,session
from flask_cors import CORS
from search import read_database
from search import fuzzymatch
from search import search
import logging

# Create the application.
APP = flask.Flask(__name__)
results = None

# r'/*' 是通配符，让本服务器所有的 URL 都允许跨域请求
CORS(APP, resources=r'/*')

# 记录日志
file_handler = logging.FileHandler('APP.log')
APP.logger.addHandler(file_handler)
APP.logger.setLevel(logging.INFO)

@APP.route('/')
def index():
    """ 显示可在 '/' 访问的 index 页面
    """
#    APP.logger.info('informing')
#    APP.logger.warning('warning')
#    APP.logger.error('screaming bloody murder!')
    return flask.render_template('index.html')

@APP.route('/todimsearch')
def todimsearch():
    global results
    results=read_database()
    return flask.render_template('dimsearch.html')
	
@APP.route('/dimsearch/<dimname>',methods=['GET'])	
def dimsearch(dimname):
    global results
    dimraw_info = results
    dimmatchEntity = fuzzymatch(dimname,dimraw_info[2])
    return Response(json.dumps(dimmatchEntity,ensure_ascii=False), mimetype='application/json')

@APP.route('/search/<selectName>/<type>',methods=['POST','GET'])
def relativesearch(selectName,type):
    global results
    raw_info = results
    matchEntity = fuzzymatch(selectName,raw_info[2])
    df_search = search(raw_info[0],matchEntity,raw_info[1],type)
    final_result = df_search.to_json(orient='records',force_ascii=False)
    return Response(final_result, mimetype='application/json')

@APP.errorhandler(404)
def not_found(error=None):
    message = {
            'status': 404,
            'message': 'Not Found: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 404

    return resp
	
@APP.route('/users/<userid>', methods = ['GET'])
def api_users(userid):
    users = {'1':'john', '2':'steve', '3':'bill'}
    
    if userid in users:
        return jsonify({userid:users[userid]})
    else:
        return not_found()

if __name__ == '__main__':
    APP.debug=True
    APP.run(host='0.0.0.0',port=8888)


