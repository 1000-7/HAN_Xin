#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import pickle
from config import config
import numpy as np
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
import nltk
from tqdm import tqdm
from sqlalchemy import create_engine

config = config["10"]
max_sent_in_doc = config["max_sent_in_doc"]
max_word_in_sent = config["max_word_in_sent"]
UNKNOWN = 0
sent_tokenizer = nltk.data.load(config["sent_tokenizer_path"])  # 加载英文的划分句子的模型
word_tokenizer = WordPunctTokenizer()  # 加载英文的划分单词的模型
# 加载模型
data_save_path = "../traindata/3_6_7_submodel/"
# 初始化数据库连接，使用pymysql模块
# MySQL的用户：root, 密码:147369, 端口：3306,数据库：test
engine = create_engine("mysql+pymysql://root:aixuexi109@localhost:3306/graduate?charset=utf8")
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
def text_trans(paperSection):
    doc = np.zeros((max_sent_in_doc,
                    max_word_in_sent),
                   dtype=np.int32)
    sentences = sent_tokenizer.tokenize(paperSection)  # 将评论分句
    words = word_tokenizer.tokenize(paperSection)  # 将评论分词
    if len(words) > 50 and len(sentences) >= 2:
        for i, sent in enumerate(sentences):
            if i < max_sent_in_doc:
                words = word_tokenizer.tokenize(sent)
                for j, word in enumerate(words):
                    if j < max_word_in_sent:
                        doc[i][j] = vocab.get(word, UNKNOWN)
        return [doc.tolist()]
    else:
        return None
def split_text(paperSection):
    sentences = sent_tokenizer.tokenize(paperSection)  # 将评论分句
    if len(sentences) > 1: return [" ".join(sentences[0:int(len(sentences)/2)])," ".join(sentences[int(len(sentences)/2):])]
    else: return [" ".join(sentences)]
def insert_data(data):
    df = pd.DataFrame(data,columns=["paperSection","paperSectionType","type1","sign"])
    df.to_sql('semanticScholar_train_data', engine, if_exists = "append",index=False)
    print('write to Mysql table successfully!')
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def index_map(index):
    if index == 1 : return 3
    if index == 2 : return 6
    if index == 3 : return 7
dict = pd.read_excel("3_6_7.xlsx")
sheet = dict.values.tolist()
sectTitle2classindex_dict = {}
for item in sheet:
    sectTitle2classindex_dict[item[0]] = item[1]
vocab = pickle.load(open('../traindata/word2index_dict', 'rb'))
model_graph = load_model(config["model_save_path"]+"model-3-6-7.pb")
max_sentence_num = model_graph.get_tensor_by_name('placeholder/max_sentence_num:0')
max_sentence_length = model_graph.get_tensor_by_name('placeholder/max_sentence_length:0')
# x的shape为[1, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
input_x = model_graph.get_tensor_by_name('placeholder/input_x:0')
prediction = model_graph.get_tensor_by_name('doc_classification/predict:0')


text_x, text_y = pickle.load( open(data_save_path+'text_data', 'rb'))

with tf.Session(graph=model_graph) as sess:
    temp = []
    para_len = []
    cnt = 0
    total = 0
    for i,text in tqdm(enumerate(text_x)):
        for para in split_text(text):
            doc = text_trans(para)
            if doc is None:
                continue
            feed_dict = {
                input_x: doc,
                max_sentence_num: 30,
                max_sentence_length: 30
            }
            pred = sess.run(prediction, feed_dict=feed_dict)
            pred = softmax(pred)
            pred = pred[0]
            result = np.argmax(pred) + 1
            real = sectTitle2classindex_dict[text_y[i]]
            if (not real == result) and pred[result-1] > 0.70:

                temp.append([para,text_y[i],index_map(result),3])
                cnt = cnt+1
            else:
                temp.append([para, text_y[i], index_map(real),3])
                pass
            total = total+1
            if len(temp) %20000 == 0 and len(temp) > 0:
                insert_data(temp)
                temp = []
    if len(temp) > 0:
        insert_data(temp)
    print(cnt,total)
    print("avg" + str(np.sum(para_len,axis=0)))
    print("avg" + str(len(para_len)))
    print(" end ")