import pandas as pd
import numpy as np
from sklearn import preprocessing
import gensim
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding

model_path = {
    'raw_data': 'raw_data.csv',
    'model_w2v': 'w2v_model_50',
    'model_nn': 'model.json',
    'model_nn_h5': 'model.h5',
    'w2v_corpus': 'train_corpus',
}

para = {
    'epoch': 100
}


class SemanticIntentionClassify():

    def __init__(self, model_path, para):
        self.raw_data = pd.read_csv(model_path['raw_data'])
        self.model_w2v = model_path['model_w2v']
        self.model_nn = model_path['model_nn']
        self.model_nn_h5 = model_path['model_nn_h5']
        self.w2v_corpus = model_path['w2v_corpus']
        self.epoch = para['epoch']

    def _load_word2vec_model(self):
        """载入词向量模型和词的索引"""

        import gensim
        model_w2v = gensim.models.Word2Vec.load(self.model_w2v)
        i = 0
        word_to_index = {}
        for word in model_w2v.wv.vocab.keys():
            word_to_index[word] = i
            i += 1
        return model_w2v, word_to_index

    def _load_nn_model(self):
        """载入预测模型"""

        from keras.models import model_from_json
        json_file = open(self.model_nn, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.model_nn_h5)
        return loaded_model

    def _sentences_to_indices(self, x, word_to_index, max_len):
        """批量句子转换为词向量字典对应的索引"""

        m = x.shape[0]
        x_indices = np.zeros((m, max_len))
        for i in range(m):
            sentence_words = x[i]
            j = 0
            for w in sentence_words:
                if w != ' ' and w in word_to_index.keys():
                    x_indices[i, j] = word_to_index[w]
                    j = j + 1
        return x_indices

    def data_generate(self):
        """
        功能：
        批量转化原始数据，用来预测或者训练

        用法：
        model = SemanticIntentionClassify(model_path)
        x,y = model.data_generate()
        """

        raw_data = self.raw_data
        _, word_to_index = self._load_word2vec_model()
        x = []
        for index in raw_data.index:
            x_raw = raw_data.iloc[index]['X']
            sentence = []
            for each in x_raw:
                if each == ' ':
                    continue
                else:
                    sentence.append(each)
            x.append(sentence)
        x = np.array(x)
        x = self._sentences_to_indices(x, word_to_index, max_len=30)
        le = preprocessing.LabelEncoder()
        le.fit(raw_data['Y'])
        y_word_to_indice = {}
        for i in range(len(le.classes_)):
            y_word_to_indice[i] = le.classes_[i]
        y = le.transform(raw_data['Y'])
        return x, y, y_word_to_indice

    def error_analysis(self, x, y):
        """
        功能：
        用于对输入的数据做误差检查，返回预测不对的数据索引

        参数：
        x：方法data_generate生成的x
        y：方法data_generate生成的y

        用法：
        model = SemanticIntentionClassify(model_path)
        x,y = model.data_generate()
        fault_indice = model.error_analysis(x,y)
        """

        nn_model = self._load_nn_model()
        model_w2v, _ = self._load_word2vec_model()
        fault_indice = []
        num = 0
        for i in range(len(x)):
            pred = nn_model.predict(np.array([x[i]]))
            pred_y = np.argmax(pred)
            if y[i] == pred_y:
                num += 1
            else:
                fault_indice.append(i)
        return fault_indice

    def predict(self, x):
        """
        功能：
        用于预测句子的意图

        参数：
        x：是要预测句子的字符串

        返回：
        句子的意图

        用法：
        model = SemanticIntentionClassify(model_path)
        x = "你的库里面有悦阅的部门吗？"
        num = model.error_analysis(x,y)
        model.predict(x)
        """

        # _,_,y_word_to_indice = self.data_generate()
        y_word_to_indice = {0: '找职位', 1: '找邮箱', 2: '找部门', 3: '找领导'}
        nn_model = self._load_nn_model()
        model_w2v, word_to_index = self._load_word2vec_model()
        inputs_list = []
        for each in x:
            if each in model_w2v.wv.vocab:
                inputs_list.append(each)
        x_test = np.array([inputs_list])
        x_test_indices = self._sentences_to_indices(x_test, word_to_index, 30)
        pred = nn_model.predict(x_test_indices)
        class_ = y_word_to_indice[np.argmax(pred)]

        return class_

    def train_w2v(self):
        """
        作用：
        训练语料的词向量模型

        例子：
        model = SemanticIntentionClassify(model_path)
        model_w2v = model.train_w2v()
        model_w2v.save("{}".format(path_of_model_you_want_to_save))
        """
        from gensim.models import word2vec
        from gensim import models
        import gensim
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.Text8Corpus(self.w2v_corpus)
        model = gensim.models.Word2Vec(size=50, window=5, min_count=1)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=100)
        # model.save("{}".format('w2v_model_50'))

    def _pretrained_embedding_layer(self, word_to_vec_map, word_to_index):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

        Arguments:
        word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

        Returns:
        embedding_layer -- pretrained layer Keras instance
        """

        vocab_len = len(word_to_index) + 1
        emb_dim = word_to_vec_map['i'].shape[0]
        emb_matrix = np.zeros((vocab_len, emb_dim))
        for word, index in word_to_index.items():
            emb_matrix[index, :] = word_to_vec_map[word]
        embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
        embedding_layer.build((None,))
        embedding_layer.set_weights([emb_matrix])
        return embedding_layer

    def _keras_model(self, input_shape, word_to_vec_map, word_to_index):
        """
        Function creating the Emojify-v2 model's graph.

        Arguments:
        input_shape -- shape of the input, usually (max_len,)
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

        Returns:
        model -- a model instance in Keras
        """

        sentence_indices = Input(input_shape, dtype='int32')
        embedding_layer = self._pretrained_embedding_layer(word_to_vec_map, word_to_index)
        embeddings = embedding_layer(sentence_indices)
        X = LSTM(128, return_sequences=True)(embeddings)
        X = Dropout(0.5)(X)
        X = LSTM(128, return_sequences=False)(X)
        X = Dropout(0.5)(X)
        X = Dense(4)(X)
        X = Activation('softmax')(X)
        keras_model = Model(inputs=sentence_indices, outputs=X)
        return keras_model

    def _embedding_dict(self, model_w2v):
        embedding_dict = {}
        for i in range(len(model_w2v.wv.vocab)):
            embedding_vector = model_w2v.wv[model_w2v.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_dict[model_w2v.wv.index2word[i]] = embedding_vector
        return embedding_dict

    def _convert_to_one_hot(self, y, c):
        y = np.eye(c)[y.reshape(-1)]
        return y

    def train_classifier(self):
        """
        作用:
        训练语义意图理解的分类器

        例子：

        model = SemanticIntentionClassify(model_path,para)
        model.train_classifier()
        from keras.models import model_from_json
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        """

        x, y, _ = self.data_generate()
        y = self._convert_to_one_hot(y, len(set(y)))
        model_w2v, word_to_index = self._load_word2vec_model()
        word_to_vec_map = self._embedding_dict(model_w2v)
        maxLen = 30
        model = self._keras_model((maxLen,), word_to_vec_map, word_to_index)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x, y, epochs=self.epoch, shuffle=True)

