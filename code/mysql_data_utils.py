#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pymysql
import json
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
import numpy as np

# 打开数据库连接
db = pymysql.connect("localhost", "root", "wx1996", "graduate", charset='utf8')

# 使用cursor()方法获取操作游标
cursor = db.cursor()

sent_tokenizer = nltk.data.load('/Users/unclewang/nltk_data/tokenizers/punkt/english.pickle')  # 加载英文的划分句子的模型
word_tokenizer = WordPunctTokenizer()  # 加载英文的划分单词的模型

# 记录每个单词及其出现的频率
word_freq = defaultdict(int)  # Python默认字典

# SQL 查询语句
sql = "SELECT * FROM semanticScholar_filter"
try:
    # 执行SQL语句
    cursor.execute(sql)
    # 获取所有记录列表
    results = cursor.fetchall()
    for row in results:
        paperSection = row[2].replace("- ", "")
        type1 = row[6]
        words = word_tokenizer.tokenize(paperSection)
        if len(words) < 50:
            continue
        for word in words:
            word_freq[word] += 1
    print("load finish")

    with open('word_freq.pickle', 'wb') as g:
        pickle.dump(word_freq, g)
        print(len(word_freq))  # 159654
        print("word_freq save finished")

    num_classes = 5
    sort_words = list(sorted(word_freq.items(), key=lambda x: -x[1]))  # 按出现频数降序排列
    print(sort_words[:10], sort_words[-10:])  # 打印前十个和倒数后十个

    # 构建vocablary，并将出现次数小于5的单词全部去除，视为UNKNOW
    vocab = {}
    i = 1
    vocab['UNKNOW_TOKEN'] = 0
    for word, freq in word_freq.items():
        if freq > 5:
            vocab[word] = i
            i += 1
    print(i)  # 6651

    UNKNOWN = 0
    data_x = []
    data_y = []
    max_sent_in_doc = 30
    max_word_in_sent = 30
    # 将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
    # 多余的删除，并保存到最终的数据集文件之中
    # 注意，少于30的并没有进行补零操作

    totalSentenceNum = 0
    totalWordNum = 0

    for row in results:
        # doc = []
        doc = np.zeros((30, 30), dtype=np.int32)
        paperSection = row[2].replace("- ", "")
        type1 = row[6]
        sentences = sent_tokenizer.tokenize(paperSection)  # 将评论分句
        totalSentenceNum += len(sentences)

        for i, sent in enumerate(sentences):
            words = word_tokenizer.tokenize(sent)
            totalWordNum += len(words)
            # if i < max_sent_in_doc:
            #     word_to_index = []
            #     words = word_tokenizer.tokenize(sent)
            #     totalWordNum += len(words)
    print("sentenceTotalNum:%s, wordTotalNum:%s, sentenceLen:%s, sectionLen:%s" % (
        totalSentenceNum, totalWordNum, totalWordNum / totalSentenceNum, totalSentenceNum / len(results)))
    #
    #             for j, word in enumerate(words):
    #                 if j < max_word_in_sent:
    #                     # word_to_index.append(np.array(vocab.get(word, UNKNOWN)))
    #                     doc[i][j] = vocab.get(word, UNKNOWN)
    #             doc.append(np.array(word_to_index))
    #
    #     label = type1
    #     labels = [0] * num_classes
    #     labels[label - 1] = 1
    #     data_y.append(labels)
    #     data_x.append(doc.tolist())
    # pickle.dump((data_x, data_y), open('yel_data', 'wb'))
    # print("data_X:", data_x[10])
    # print("data_y:", data_y[10])
    # print(len(data_x))  # 229907
    # length = len(data_x)
    # train_x, dev_x = data_x[:int(length*0.9)], data_x[int(length*0.9)+1 :]
    # train_y, dev_y = data_y[:int(length*0.9)], data_y[int(length*0.9)+1 :]

except:
    print("Error: unable to fetch data")

# 关闭数据库连接
db.close()
