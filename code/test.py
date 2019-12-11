#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import pickle
from config import config
import numpy as np
config = config["10"]

# 加载模型
def load_model(path_to_model="../codemodel-12-11.pb"):
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
    with open("../traindata/test_data", 'rb') as f:
        test_x, test_y = pickle.load(f)
        length = len(test_y)
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
        return test_y,pred_y
def get_accuracy_perClass(test_y,pred_y):
    totalNum = [0]*config["num_classes"]
    accuracyNum = [0]*config["num_classes"]
    for i,pred in enumerate(pred_y):
        real = np.argmax(test_y[i])
        if np.argmax(pred) == real:
            temp = accuracyNum[real]
            accuracyNum[real] = temp+1
        totalNum[real] = totalNum[real] + 1
    for i,tNum in enumerate(totalNum):
        if not tNum == 0:
            print("类别 " + str(i+1) + " 正确率为 " + str(accuracyNum[i]/tNum) + " 类别总数 " + str(tNum))


if __name__ == "__main__":
    test_y, pred_y = TestModel()
    get_accuracy_perClass(test_y,pred_y)