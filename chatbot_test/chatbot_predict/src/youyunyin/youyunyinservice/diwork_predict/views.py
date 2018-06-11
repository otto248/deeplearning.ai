from django.shortcuts import render
from django.http import HttpResponse
from keras.models import model_from_json
import tensorflow as tf
from datetime import datetime
import os
import numpy as np
from . import util
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
"""Load classifier as global parameter"""

try:
    global graph
    graph = tf.get_default_graph()
    MODEL_DIR = os.path.dirname(os.getcwd()) + '/youyunyinservice/diwork/templates/model/'
    now = datetime.now().strftime('20%y-%-m')
    print("Loading w2v model...")
    model_w2v = util.read_w2v(MODEL_DIR + 'chara_vec_100-' + now)
    print("Loading keras model...")
    with open(MODEL_DIR + "model-" + now + ".json",'r') as f:
        loaded_model_json = f.read()

    keras_model = model_from_json(loaded_model_json)
    keras_model.load_weights(MODEL_DIR + "model-" + now + ".h5")
    print ("Loading succeed")
except:
    pass



"""Predict a sentence and return the correspond intent"""
@csrf_exempt
def predict(request):

    if request.method == 'POST':
        try:
            with graph.as_default():
                text = request.POST.get('text')
                #text = '和杨林召开个视频会议'
                x = util.preprocess(text,model_w2v)
                pred = keras_model.predict(x)
                y_class = util.y_word_to_indice()
                if max(pred[0]) > 0.9:
                    predicted_class = y_class[np.argmax(pred)]
                    return HttpResponse(predicted_class)
                else:
                     return HttpResponse('No matched due to low-probability')
        except Exception as e:
            return HttpResponse(e)
    else:
        return HttpResponse('Please using POST method')
