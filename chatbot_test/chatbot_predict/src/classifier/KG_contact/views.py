from django.shortcuts import render
from django.http import HttpResponse
from keras.models import model_from_json
import tensorflow as tf
from . import util
import os
from django.views.decorators.csrf import csrf_exempt
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
# Create your views here.
"""Load classifier as global parameter"""

try:
    global graph
    graph = tf.get_default_graph()
    MODEL_DIR = os.path.dirname(os.getcwd()) + '/classifier/KG_contact/templates/model/'
    #now = datetime.now().strftime('20%y-%-m')
    print("Loading w2v model...")
    model_w2v = util.read_w2v(MODEL_DIR + 'chara_vec_50')
    
    print("Loading keras model...")
    with open(MODEL_DIR + "model" + ".json",'r') as f:
        loaded_model_json = f.read()
    keras_model = model_from_json(loaded_model_json)
    keras_model.load_weights(MODEL_DIR + "model" + ".h5")
    
    print ("Loading user dictionary")
    department = list(set(pd.read_csv(MODEL_DIR + 'usr_dict_department')['department']))
    person = list(set(pd.read_csv(MODEL_DIR + 'usr_dict_person')['person']))
    for d in department:
        jieba.add_word(d, tag = 'department')
    for p in person:
        jieba.add_word(p, tag = 'yyname')
    print ('loading success')
except Exception as e:
    print (e)

def cut_word(text,department):
    words = pseg.cut(text)
    for word,flag in words:
        if flag == 'yyname':
            return {"yyname": word}
        
        elif flag == 'department':
            return {"department": word}
    
    return None
        

        
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            with graph.as_default():
                text = request.POST.get('text')
                department = list(set(pd.read_csv(MODEL_DIR + 'usr_dict_department')['department']))
                cut_words = cut_word(text, department)
                #text = '和杨林召开个视频会议'
                x = util.preprocess(text,model_w2v)
                pred = keras_model.predict(x)
                y_class = util.y_word_to_indice()
                #if max(pred[0]) > 0.95:
                intend = y_class[np.argmax(pred)]
                #else:
                #     return HttpResponse('No matched due to low-probability')
                if cut_words != None:
                    response = util.search_entity(cut_words,intend)
                    return HttpResponse(response)
                else:
                    return HttpResponse("No matched entity found")
                
        except Exception as e:
            return HttpResponse(e)
    else:
        return HttpResponse('Please using POST method')
