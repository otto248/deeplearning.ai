#导入词向量模型
def read_w2v(path):
    import gensim
    model_w2v = gensim.models.Word2Vec.load(path)#传入w2v的model地址
    return model_w2v


#构建字到索引的字典
def word_to_index(model_w2v):
    i = 0
    word_to_index = {}
    for word in model_w2v.wv.vocab.keys():
        word_to_index[word] = i
        i += 1
    return word_to_index

#将输入的句子转换为数字索引表示
def sentences_to_indices(X, word_to_index, max_len=30):
    """    
    参数:
    X -- 句子的矩阵，维度为（m，1）
    word_to_index -- 字到索引的字典
    max_len -- 句子的最大长度，每个句子的长度不会大于这个值 
    
    返回:
    X_indices -- 句子的索引表示，维度为 (m, max_len)
    """
    import numpy as np
    #训练集的数量
    m = X.shape[0]
    #初始化一个0矩阵，矩阵维度为(m,max_len)
    X_indices = np.zeros((m,max_len))
    #对每个句子遍历
    for i in range(m): 
            # 读取第i个训练样本，是字的列表
        sentence_words = X[i]   
            # 初始化j=0
        j = 0   
    # 遍历列表中的每个字，并并转换为数字
        for w in sentence_words:
            # 将X_indices中用对应的字的索引填充
            X_indices[i, j] = word_to_index[w]
            #j+1
            j = j+1          
    return X_indices

def convert_to_one_hot(Y, C):
    import numpy as np
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

#生成训练集X，Y
def training_set_generation(raw_data,word_to_index,maxLen):
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    train_X = []

    for index in raw_data.index:
        X = raw_data.iloc[index]['X']
        sentence = []
        for x in X:
            if x != ' ':
                sentence.append(x)
        train_X.append(sentence)

    X = np.array(train_X)

    X_train_indices = sentences_to_indices(X, word_to_index, maxLen)

    le = preprocessing.LabelEncoder()
    le.fit(raw_data['Y'])
    Y = le.transform(raw_data['Y'])
    Y_train_oh = convert_to_one_hot(Y, C = 7)
        
    return X_train_indices, Y_train_oh



def softmax(x):
    import numpy as np
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def word_to_vec_map(model_w2v):
    """
    参数：
    model:训练好的w2v参数

    返回：
    word_to_vec_map:字到向量表示的字典
    """
    import numpy as np
    #初始化字典
    word_to_vec_map = {}
    #遍历w2v模型中的每个词，读取每个词的向量，并存储到word_to_vec_map
    for i in range(len(model_w2v.wv.vocab)):
        embedding_vector = model_w2v.wv[model_w2v.wv.index2word[i]]
        if embedding_vector is not None:
            word_to_vec_map[model_w2v.wv.index2word[i]] = embedding_vector
    return word_to_vec_map

def y_word_to_indice():
    y_word_to_indice = {0:'找职位',
                        1:'找邮箱',
                        2:'找部门',
                        3:'找领导'}
    return y_word_to_indice


def predict(inputs,model_w2v,model_classifier,maxLen):#model is the pre_trained w2v
    import numpy as np
    inputs_list = []
    for each in inputs:
        if each in model_w2v.wv.vocab:
            inputs_list.append(each)
        
    X_test = np.array([inputs_list])        
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
    pred = model_classifier.predict(X_test_indices)
    class_ = y_word_to_indice[np.argmax(pred)]
    return class_

def preprocess(inputs,model_w2v):
    import numpy as np
    inputs_list = [word for word in inputs if word in model_w2v.wv.vocab]             
    X = np.array([inputs_list])  
    X_indices = sentences_to_indices(X, word_to_index(model_w2v), 30)
    return X_indices

        
def mongo_connect():
    talendid = 'KG'
    from pymongo import MongoClient
    dbserver = '172.20.9.29'
    dbname = 'nccloud-analysis-' + talendid
    collname = talendid + '_contact_list'
    client = MongoClient(dbserver)
    client.admin.authenticate("ublinker","nash2017")
    db = client[dbname]
    collection = db[collname]
    return collection


def fuzzymatch(usr_input,collection):
    import re
    suggestions = []
    pattern = '.*?'.join(usr_input)
    regex = re.compile(pattern)
    for item in collection:
        match = regex.search(item)
        if match:
            suggestions.append((len(match.group()), match.start(), item))
    return [x for _, _, x in sorted(suggestions)]

def search_entity(cut_words,intend):
    intend_relation = {
        "找职位":"职位",
        "找邮箱":"邮箱",
        "找部门":"所属部门",
        "找领导":"部门经理",
        
    }
    relation = intend_relation[intend]
    conn = mongo_connect()
    result = []
    if intend != "找领导":
        result.append(list(conn.find({"stuff": cut_words['yyname'], "relation":relation})))
    else:
        for key, value in cut_words.items():
            if key == "yyname":
                result.append(list(conn.find({"stuff": value,"relation":relation})))
            elif key == "department":
                result.append(list(conn.find({"department":value, "relation":relation})))
    return result

