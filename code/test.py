#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import pickle
from config import config
import numpy as np
from modelAsServer import textPreprocessing
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
config = config["10"]
# 加载模型
def load_model(path_to_model = config["model_save_path"]+"model-3-6-7.pb"):
    if not os.path.exists(path_to_model):
        raise ValueError(path_to_model + " is not exist.")

    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return model_graph
def index_map(i):
    indexmap = {
        1:1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 3,
        7: 3,
        8: 6,
        9: 7,
        10: 8,
        11: 9
    }
    return indexmap[i]
def submodel(test_x, test_y):
    model_graph = load_model(config["model_save_path"] + "model-12-25-9.pb")
    max_sentence_num = model_graph.get_tensor_by_name('placeholder/max_sentence_num:0')
    max_sentence_length = model_graph.get_tensor_by_name('placeholder/max_sentence_length:0')
    # x的shape为[1, 句子数， 句子长度(单词个数)]，_balance但是每个样本的数据都不一样，，所以这里指定为空
    input_x = model_graph.get_tensor_by_name('placeholder/input_x:0')
    prediction = model_graph.get_tensor_by_name('doc_classification/predict:0')
    # sect_dict = get_dict()
    with tf.Session(graph=model_graph) as sess:
        pred_y = []
        real_y = []
        for i, y in tqdm(enumerate(test_y)):
            # if not y in sect_dict.keys():
            #     continue

            doc = textPreprocessing(test_x[i])
            if doc is None: continue
            feed_dict = {
                input_x: [doc],
                max_sentence_num: 30,
                max_sentence_length: 30
            }
            pred = sess.run(prediction, feed_dict=feed_dict)
            pred = np.argmax(pred)
            pred_y.append(pred+1)

            real = np.argmax(y)
            real_y.append(index_map(real+1))
        return real_y, pred_y

def get_dict():
    dict = pd.read_excel("./12-18sectiontitle2classindex.xlsx")
    sheet = dict.values.tolist()
    sectTitle2classindex_dict = {}
    for item in sheet:
        sectTitle2classindex_dict[item[0]] = item[1]
    return sectTitle2classindex_dict
def TestModel():

    with open("../traindata/new_train/test_data_4000", 'rb') as f:
        test_x, test_y = pickle.load(f)
        print(np.sum(test_y, axis=0))
        return submodel(test_x, test_y)


def get_accuracy_perClass(test_y,pred_y,classNum ):
    target_names = ["class" + str(i+1) for i in range(0,classNum)]
    res = np.zeros((classNum,classNum))
    for i,real in enumerate(test_y):
        temp =  res[real-1,pred_y[i]-1]
        res[real - 1, pred_y[i]-1] = temp +1
    print(classification_report(test_y,
                                pred_y,
                                target_names=target_names))

    data = pd.DataFrame(res.tolist(),
                        index=["真实"+str(i) for i in range(1,classNum+1)],
                        columns = ["预测"+str(i) for i in range(1,classNum+1)] )
    pd.set_option('display.max_columns',1000)
    print(data)
if __name__ == "__main__":
    test_y, pred_y = TestModel()
    get_accuracy_perClass(test_y,pred_y,9)