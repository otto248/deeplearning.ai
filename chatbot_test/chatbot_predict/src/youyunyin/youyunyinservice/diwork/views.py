from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

from . import util
import pandas as pd
from datetime import datetime
import os
from django.views.decorators.csrf import csrf_exempt
# import logging
# logger = logging.getLogger('sourceDns.webdns.views')

BASE_DIR = os.getcwd() + '/diwork/templates'

"""extract data from origin"""

@csrf_exempt
def extract_origin(request):
    try:
        if request.method == 'POST':
            time = request.POST.get('time')
            if time == None:
                time = datetime.now().strftime("20%y-%-m")

            talendid = 'diwork'
            appid = 'diwork'
            conn = util.connect_to_origin_data(talendid = talendid, appid = appid,time = time)
            df = pd.DataFrame(list(conn.find()))
            file_name = talendid + '-' + appid + '-' + time
            df.to_csv(BASE_DIR + '/data/diwork/dirty_data/' + file_name)
            return HttpResponse('1')
    except Exception as e:
            return HttpResponse(e)
            #logger.error(e)

"""Upload data to mongo"""

@csrf_exempt
def to_database(request):
    """Upload dirty data to Mongo"""
    try:
        if request.method == 'GET':
            conn = util.connect_to_training_data(talendid = 'diwork', appid = 'diwork')
            data = util.read_raw_data()
            mongo_data = data.T.to_dict().values()
            conn.insert_many(mongo_data)
            return HttpResponse('1')

    except Exception as e:
        return HttpResponse (e)
        #logger.error(e)


"""Training w2v model"""

@csrf_exempt
def training_w2v(request):
    """Training word vector"""
    try:
        if request.method == 'GET':
            import pandas as pd
            conn = util.connect_to_training_data(talendid = 'diwork', appid = 'diwork')
            raw_data = pd.DataFrame.from_records(conn.find()).drop(['_id'],axis=1)
            util.generate_w2v_training_corpus(raw_data)

            now = datetime.now().strftime('20%y-%-m')

            corpus_dir = BASE_DIR + '/model/w2v_corpus-' + now
            from gensim.models import word2vec
            import gensim
            import logging
            #训练日志
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
            #导入训练文件
            sentences = word2vec.Text8Corpus(corpus_dir)
            #初始化模型
            model = gensim.models.Word2Vec(size=100, window=5, min_count=1)
            #构建模型字典
            model.build_vocab(sentences)
            #模型训练
            model.train(sentences, total_examples=model.corpus_count, epochs=100)
            #模型保存
            model.save("{}".format(BASE_DIR + '/model/chara_vec_100-' + now))
            return HttpResponse('1')
    except Exception as e:
            return HttpResponse(e)
            #logger.error(e)


"""Training classifier"""
import tensorflow as tf
global graph
graph = tf.get_default_graph()

@csrf_exempt
def training_classifier(request):
    try:
        if request.method == 'GET':
            with graph.as_default():
                import pandas as pd
                now = datetime.now().strftime('20%y-%-m')
                conn = util.connect_to_training_data(talendid='diwork', appid='diwork')
                raw_data = pd.DataFrame.from_records(conn.find()).drop(['_id'],axis=1)
    
                w2v_model_path = BASE_DIR + '/model/chara_vec_100-' + now
                classifer_path = BASE_DIR + '/model/'
    
                maxLen = 30
                model_w2v = util.read_w2v(w2v_model_path)
                word_to_index = util.word_to_index(model_w2v)
                X_train_indices, Y_train_oh = util.training_set_generation(raw_data, word_to_index, maxLen)
                word_to_vec_map = util.word_to_vec_map(model_w2v)
                keras_model = util.keras_model((maxLen,), word_to_vec_map, word_to_index)
                keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                keras_model.fit(X_train_indices, Y_train_oh, epochs = 10, batch_size = 16, shuffle=True)
    
                #save model
                model_json = keras_model.to_json()
                with open(classifer_path + "model-" + now + ".json", "w+") as json_file:
                    json_file.write(model_json)
                keras_model.save_weights(classifer_path + "model-" + now + ".h5")
                return HttpResponse('1')

    except Exception as e:
            return HttpResponse(e)
            # logger.error(e)



    

