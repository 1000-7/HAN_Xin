#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import nltk
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import os
import numpy as np
import pickle
from config import config,index2class_name
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = config["10"]
numClasses = config["num_classes"]
# 分句，分词加载
sent_tokenizer = nltk.data.load(config["sent_tokenizer_path"])  # 加载英文的划分句子的模型
word_tokenizer = WordPunctTokenizer()  # 加载英文的划分单词的模型
# 加载词-index字典
with open('../traindata/word2index_dict', 'rb') as f:
    vocab = pickle.load(f)
    UNKNOWN = 0
sess = tf.Session()
pb_file_path = os.getcwd()

def textPreprocessing(text, max_sent_in_doc=30, max_word_in_sent=30):
    '''
    对输入的文本进行预处理
    分句，分词，
    最后基于词-index字典 将其转换为一个矩阵
    :param doc:
    :return:
    '''
    doc = np.zeros((max_sent_in_doc, max_word_in_sent), dtype=np.int32)
    paperSection = text.replace("- ", "")
    sentences = sent_tokenizer.tokenize(paperSection)  # 将评论分句
    words = word_tokenizer.tokenize(paperSection)
    if len(words) < 50 or len(sentences) < 2:
        return None
    for i, sent in enumerate(sentences):
        words = word_tokenizer.tokenize(sent)
        if i < max_sent_in_doc:
            word_to_index = []
            words = word_tokenizer.tokenize(sent)
            for j, word in enumerate(words):
                if j < max_word_in_sent:
                    doc[i][j] = vocab.get(word, UNKNOWN)
    return doc


# 加载模型
def load_model(path_to_model="../codemodel.pb"):
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


def predict(doctest):
    model_graph = load_model()

    max_sentence_num = model_graph.get_tensor_by_name('placeholder/max_sentence_num:0')
    max_sentence_length = model_graph.get_tensor_by_name('placeholder/max_sentence_length:0')
    # x的shape为[1, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
    input_x = model_graph.get_tensor_by_name('placeholder/input_x:0')
    doc = textPreprocessing(doctest)
    feed_dict = {
        input_x: [doc],
        max_sentence_num: 30,
        max_sentence_length: 30
    }

    prediction = model_graph.get_tensor_by_name('doc_classification/predict:0')

    with model_graph.as_default():
        with tf.Session(graph=model_graph) as sess:
            pred = sess.run(prediction, feed_dict=feed_dict)
            pred = pred[0]
    print(pred)
    return pred
def argmax(predict_result_list):
    return np.argmax(predict_result_list)+1

if __name__ == "__main__":
    sectionText = "Multilevel models refer to analytic models that contain variables mea- D Methods sured at different levels of the hierarchy (Kreft & De Leeuw, ). Research propositions tested in multilevel analysis refer to relationships between variables that are measured at different hierarchical levels. Because multilevel models acknowledge hierarchical data, researchers should not move aggregation or disaggregation variables to a single level (Hox, ). Thus, there are conceptual and statistical advantages in multilevel analysis; variables are analyzed at the level that they were defined and measured. For example, if job satisfaction is conceptualized and measured at the individual nurse level, it is theoretically correct to analyze the variable at the nurse level, not at a higher level (e.g., care unit, hospital). These advantages are particularly meaningful for comparing patient outcomes across hospitals because risk adjustment can be conducted at the patient level without aggregating risk factors at the hospital level. The traditional approach to dealing with multilevel data in nursing outcomes research is to aggregate individual-level variables at the higher level. For example, patients who had pneumonia after surgery are aggregated at the hospital level by calculating an overall pneumonia rate for each hospital. Patient-level predictors (e.g., age, severity of illness) are also aggregated at the hospital level. Hos- Using Multilevel Analysis in Patient and Organizational. Outcomes Research. Sung-Hyun Cho Nursing Research January/February Vol , No . Sung-Hyun Cho, PhD, MPH, RN, is a. researcher at Korea Institute for Health and Social Affairs, Seoul, Korea. pital characteristics are then included in a multivariate model with the hospital as the unit of analysis. The literature reports several problems related to data aggregation. The ecological fallacy, which occurs when relationships between variables are examined using data aggregated at the group level, but conclusions are drawn at the individual level. Robinson () concluded that correlations between aggregated variables cannot be used as substitutes for individual correlations. For example, when the mean age of patients in each hospital has an association with the overall pneumonia rate of hospitals, this relationship does not allow any inferences about the effect of age on the occurrence of pneumonia at the patient level. In this example, the problem of “shift of meaning” also arises (Snijders & Bosker, ). The shift of meaning holds that when a variable of individuals is aggregated to the group level, the meaning of the variable does not directly refer to the individual, but rather to the group. The mean age of patients may be an implication for the patient population of the hospital; and further, this meaning may be distinct from that of age at an individual level. The statistical issue may be another potential problem of data aggregation. In this instance, the process of aggregating to the higher level may inflate the estimates of the true relationship between variables because aggregated data eliminates within-hospital variance (Kreft & De Leew, ). An alternative approach to nested data would be using a single regression model with patients as the unit of analysis, including dummy variables for hospitals in the model. This"
    res = predict(sectionText)
    index = argmax(res)
    res = index2class_name[index]
    print(res)
