#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import pickle
from config import config

config = config["104"]

# 加载模型
def load_model(path_to_model="/home/wangxin/PycharmProjects/mayq/codemodel.pb"):
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

def TestModel():
    with open("/home/wangxin/PycharmProjects/mayq/code/mysql/han_type1_data_1", 'rb') as f:
        data_x, data_y = pickle.load(f)
        length = len(data_x)
        test_x = data_x[int(length * 0.99) + 1:]
        test_y = data_y[int(length * 0.99) + 1:]
        # 加载模型
        model_graph = load_model()

        max_sentence_num = model_graph.get_tensor_by_name('placeholder/max_sentence_num:0')
        max_sentence_length = model_graph.get_tensor_by_name('placeholder/max_sentence_length:0')
        # x的shape为[1, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
        input_x = model_graph.get_tensor_by_name('placeholder/input_x:0')

        prediction = model_graph.get_tensor_by_name('doc_classification/predict:0')
        pred_y = []
        with model_graph.as_default():

            with tf.Session(graph=model_graph) as sess:

                acc = 0
                for i, real_y in enumerate(test_y):
                    feed_dict = {
                        input_x: [test_x[i]],
                        max_sentence_num: 30,
                        max_sentence_length: 30
                    }
                    pred = sess.run(prediction, feed_dict=feed_dict)
                    pred_y.append(pred)
                    if real_y[pred[0]] == 1:
                        acc = acc + 1
                print("acc  " + str(acc / len(test_y)))
        return test_y,pred_y
def get_accuracy_perClass(test_y,pred_y):
    totalNum = [[0]*config["num_classes"]]
    accuracyNum = [[[0]*config["num_classes"]]]
    for i,pred in enumerate(test_y):
        pass
    for i,tNum in enumerate(totalNum):
        print("类别 " + str(i+1) + " 正确率为-" + str(accuracyNum[i]/tNum))


