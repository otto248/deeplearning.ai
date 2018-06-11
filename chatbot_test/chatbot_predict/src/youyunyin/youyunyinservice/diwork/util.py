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

#预训练的embedding层
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    参数：
    word_to_vec_map -- 字典，映射出每个词的向量表示
    word_to_index -- 字典，映射出词在的字典中词典表示

    Returns:
    embedding_layer -- keras的embedding层
    """
    from keras.layers.embeddings import Embedding
    import numpy as np
    #keras embedding层+1
    vocab_len = len(word_to_index) + 1
    #embedding的向量维度
    emb_dim = word_to_vec_map['崔'].shape[0]
    # 初始化embedding矩阵,维度为（vocab_len,emb_dim）
    emb_matrix = np.zeros((vocab_len,emb_dim))
    # 读取embedding每个字的向量表示，并填入embedding矩阵中
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    # 创建keras的embedding实例，不可训练的，并初始化一个权重
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


#搭建分类神经网络
def keras_model(input_shape, word_to_vec_map, word_to_index):
    """ 
    参数：
    input_shape -- 输入的维度
    word_to_vec_map:字到向量表示的字典
    word_to_index -- 字到词典索引表示的字典

    返回：
    model -- 可以训练的keras实例
    """
        
    import numpy as np
    from keras.models import Model
    from keras.layers import Dense, Input, Dropout, LSTM, Activation
    from keras.layers.embeddings import Embedding
    
    #初始化网络层的输入维度
    sentence_indices = Input(input_shape, dtype='int32')
    #导入预训练的embedding层
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    #搭建embedding层
    embeddings = embedding_layer(sentence_indices)
    #搭建第一层LSTM，神经元为128,全输出   
    X = LSTM(128, return_sequences=True)(embeddings)
    #搭建dropout层
    X = Dropout(0.5)(X)
    #搭建第二层LSTM，神经元为128,只在最后输出   
    X = LSTM(128, return_sequences=False)(X)
    #搭建dropout层
    X = Dropout(0.5)(X)
    #搭建7分类的全连接层
    X = Dense(7)(X)
    #输出每个分类的概率
    X = Activation('softmax')(X)
    #创建模型的实例化对象
    keras_model = Model(inputs=sentence_indices, outputs=X)
    
    return keras_model
"""
def y_word_to_indice():
    y_word_to_indice = {0:'创建日程',
                       1:'发消息',
                       2:'召开视频会议',
                       3:'打开应用',
                       4:'找人（联系方式，名片）',
                       5:'查看报销单，打印报销单，最近的报销单',
                       6:'获取代办列表'}
    return y_word_to_indice
"""
def y_word_to_indice():
    y_word_to_indice = {0:'schedule',
                       1:'sendmessage',
                       2:'videoconference',
                       3:'searchapp',
                       4:'searchpeople',
                       5:'reimbursement',
                       6:'todlist'}
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

def read_raw_data():
    import os
    import pandas as pd
    df = pd.DataFrame()
    filePath = (os.getcwd() + '/diwork/templates/data/diwork/cleaned_data')
    for file in os.listdir(filePath):
        extension = file.split('.')[-1].lower() 
        assert extension == 'csv' or 'xlsx'
        if extension == 'csv':
            raw_data = pd.read_csv(filePath + '/' + file)
        else:
            raw_data = pd.read_excel(filePath + '/' + file)
            
        df = pd.concat([df,raw_data])
    return df

def generate_w2v_training_corpus(df):
    import os
    model_dir = os.getcwd() + '/diwork/templates/model/'

    train_corpus = []
    for index in df.index:
        X = df.loc[index]['X']     
        Y = df.loc[index]['Y']
        for x in X:
            train_corpus.append(x)
        for y in Y:
            train_corpus.append(y)
    from datetime import datetime
    now = datetime.now().strftime('20%y-%-m')
    with open(model_dir + 'w2v_corpus-'+ now,'w+') as f:
        for word in train_corpus:
            f.write(word)
            f.write(' ')

        
def connect_to_training_data(talendid, appid):
    from datetime import datetime
    now = datetime.now().strftime('20%y-%-m')

    from pymongo import MongoClient
    
    dbserver = '172.20.9.29'
    dbname = 'nccloud-analysis-' + talendid
    collname = talendid + '-' + appid + '-' + now
    
    client = MongoClient(dbserver)
    client.admin.authenticate("ublinker","nash2017")
    
    db = client[dbname]
    collection = db[collname]
    return collection

def connect_to_origin_data(talendid, appid, time):  # from-time需要传入格式2018-3

    from pymongo import MongoClient
    dbserver = '172.20.9.29'
    dbname = 'nccloud-robot-brain'
    collname = 'chatlog-' + talendid + '-' + appid + '-' + time
    client = MongoClient(dbserver)
    client.admin.authenticate("ublinker", "nash2017")
    db = client[dbname]
    collection = db[collname]

    return collection


def mongo_connect_local():
    from datetime import datetime
    now = datetime.now().strftime('20%y%m%d')
    from pymongo import MongoClient
    
    dbname = 'nccloud-analysis-diwork'
    collname = 'diwork_' + now
    
    connection = MongoClient('localhost',27017)
    db = connection[dbname]
    c = db[collname]
    return c

#if __name__ == '__main__':
#    import pandas as pd
#    maxLen = 30
#    raw_data = pd.read_csv('raw_data')
#    raw_data.dropna(axis=0,inplace = True)
#    model_w2v = read_w2v()
#    word_to_index = word_to_index(model_w2v)
#    X_train_indices, Y_train_oh = training_set_generation(raw_data, word_to_index, maxLen)
#    word_to_vec_map = word_to_vec_map(model_w2v)
#    keras_model = keras_model((maxLen,), word_to_vec_map, word_to_index)
#    keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#    keras_model.fit(X_train_indices, Y_train_oh, epochs = 100, batch_size = 16, shuffle=True)
