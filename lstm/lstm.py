import collections
import os
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Activation, SpatialDropout1D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt


def log():
    import logging
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    logging.debug('debug 信息')
    logging.info('info 信息')
    logging.warning('warning 信息')
    logging.error('error 信息')
    logging.critical('critial 信息')

    
def getWord2Vec(w2vpath, word2id=None):
    with open(w2vpath, 'r', encoding="utf-8") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        print("词向量的维度为%d" % (vector_size))
        lenword2id = len(word2id)
        embeddings = np.zeros((lenword2id, vector_size))
        for _ in range(vocab_size):
            word_list = f.readline().split(' ')
            word = word_list[0]
            if word in word2id.keys():
                embeddings[word2id[word]] = np.array(word_list[1:-1])
        return embeddings

        
def prepareData(most_common_words=8000, textNum=8000, savePath="../data/textClassifyInput.csv"):
#     文本分类数据在E:\THUCNews_cut中
#     主要的类别有三种：
#     每个类别的文本数量为5000
    text_class_name = {'财经':0, '科技':1, '时政':2}
    path = r"E:\\THUCNews_cut"
    data = []
    conter = collections.Counter()
    for tc, index in text_class_name.items():
        subpath = os.path.join(path, tc)
        for i, file_name in enumerate(os.listdir(subpath)):
            with open(os.path.join(subpath, file_name), encoding="utf-8") as f:
                text = f.read()
                text = text.split(' ')
                conter.update(text)
                data.append([text, index, len(text)])
                if i >= textNum:
                    break
    print("总词语数量 %d" % len(conter))
    wordfreqs = conter.most_common(most_common_words)    
    df = pd.DataFrame(data, columns=["文本", "类别", "文本长度"])
    df.to_csv(savePath, encoding="utf-8")
    word2id = {}
    for i, word in enumerate(wordfreqs):
        word2id[word[0]] = i
    word2id["UNK"] = most_common_words
    word2id["PAD"] = most_common_words + 1
    return word2id


def text2ids(word2id, textlist, pad_maxlen=150):
#     将文本转化为其中词语对应的id:
    X = []
    for text in textlist["文本"]:
        rest = []
        text = text[2:-2].split("', '")
        for word in text:
            if word in word2id.keys():
                rest.append(word2id[word])
            else:
                rest.append(word2id["UNK"])
        X.append(rest)
    textlist.describe()
    pad_value = word2id["PAD"]
    X = pad_sequences(X, maxlen=pad_maxlen, dtype='int32',
                  padding='post', truncating='post', value=pad_value)
    Y = []
    for classIndex in textlist["类别"]:
        Y.append(classIndex)
    return X, Y

    
def buildModel(X, Y, modelConfig):
#     建立RNN分类模型
    Xtrain, Xtest, Ytrain, Ytext = train_test_split(X, Y, test_size=modelConfig["test_size"])
    Ytrain = keras.utils.to_categorical(Ytrain, modelConfig["num_classes"])
    Ytext = keras.utils.to_categorical(Ytext, modelConfig["num_classes"])
    model = Sequential()
    model.add(Embedding(modelConfig["vocab_size"],
                        modelConfig["embed_size"],
                        input_length=modelConfig["input_len"],
                        weights=[modelConfig["embedding"]]
                        ))
    model.add(SpatialDropout1D(modelConfig["dropout"]))
    model.add(LSTM(modelConfig["hidden_size"],
                               dropout=modelConfig["dropout"],
                               recurrent_dropout=modelConfig["dropout"]))
    model.add(Dense(modelConfig["num_classes"]))
    model.add(Activation("sigmoid"))
    
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    history = model.fit(Xtrain,
                        Ytrain,
                        batch_size=modelConfig["batch_size"],
                        epochs=modelConfig["epoch"],
                        validation_data=(Xtest, Ytext))
    
    showTrainResult(history)
    return model

def showTrainResult(history):
    plt.subplot(211)
    plt.plot(history.history["acc"], color="g",
             label="Train")
    plt.plot(history.history["val_acc"], color="b",
             label="Validation")
    plt.legend("best")
    
    plt.subplot(212)
    plt.plot(history.history["loss"], color="g",
             label="Train")
    plt.plot(history.history["val_loss"], color="b",
             label="Validation")
    plt.legend("best")
    
    plt.tight_layout()
    plt.show()


def main():
    pad_maxlen = 250
    word2id = prepareData()
    data = pd.read_csv("../data/textClassifyInput.csv", encoding="utf-8")
    X, Y = text2ids(word2id, data, pad_maxlen)
    embeddings = getWord2Vec("../data/sgns.merge.word", word2id)
    modelConfig = {"num_classes":3,
                   "hidden_size":32,
                   "batch_size":32,
                   "epoch":5,
                   "test_size":0.2,
                   "input_len" : pad_maxlen,
                   "vocab_size" : len(word2id),
                   "embed_size" : len(embeddings[0]),
                   "embedding" : embeddings,
                   "dropout" : 0.2}
    
    buildModel(X, Y, modelConfig) 


if __name__ == '__main__':
    log()
    main()
