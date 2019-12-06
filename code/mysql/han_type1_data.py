#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import nltk
import numpy as np
import pymysql
from nltk.tokenize import WordPunctTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
db = pymysql.connect("localhost", "root", "irlab2017", "graduate", charset='utf8')
# db = pymysql.connect("localhost", "root", "wx1996", "graduate", charset='utf8')

# 使用cursor()方法获取操作游标
cursor = db.cursor()

# sent_tokenizer = nltk.data.load('/Users/unclewang/nltk_data/tokenizers/punkt/english.pickle')  # 加载英文的划分句子的模型
sent_tokenizer = nltk.data.load('/home/wangxin/sshFile/nltk_data/tokenizers/punkt/english.pickle')  # 加载英文的划分句子的模型
word_tokenizer = WordPunctTokenizer()  # 加载英文的划分单词的模型

# 记录每个单词及其出现的频率
word_freq = pickle.load(open('../word_freq.pickle', 'rb'))
logger.info("word_freq load finished")
# SQL 查询语句
try:
    for iter in range(1, 3):
        logger.info("now batch is:" + str(iter))
        sql = "SELECT * FROM semanticScholar_filter limit " + str(iter * 200000) + "," + str(200000) + ";"
        logger.info(sql)
        # for i in range(0)
        # 执行SQL语句
        cursor.execute(sql)

        # 获取所有记录列表
        results = cursor.fetchall()

        num_classes = 5

        logger.info("cursor fetch finished" + str(iter))
        # 构建vocabulary，并将出现次数小于5的单词全部去除，视为UNKNOW
        vocab = {}
        i = 1
        vocab['UNKNOW_TOKEN'] = 0
        for word, freq in word_freq.items():
            if freq > 5:
                vocab[word] = i
                i += 1

        UNKNOWN = 0
        data_x = []
        data_y = []
        max_sent_in_doc = 30
        max_word_in_sent = 30

        # 将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
        # 多余的删除，并保存到最终的数据集文件之中
        # 注意，少于30的并没有进行补零操作
        finish_num = 0
        for row in results:
            doc = np.zeros((30, 30), dtype=np.int32)
            paperSection = row[2].replace("- ", "")
            type1 = row[6]
            sentences = sent_tokenizer.tokenize(paperSection)  # 将评论分句
            words = word_tokenizer.tokenize(paperSection)
            if len(words) < 50 or len(sentences) < 2:
                continue

            if finish_num % 10000 == 0:
                logger.info("%s /200000 has finished" % str(finish_num))

            finish_num += 1
            for i, sent in enumerate(sentences):
                words = word_tokenizer.tokenize(sent)
                if i < max_sent_in_doc:
                    word_to_index = []
                    words = word_tokenizer.tokenize(sent)
                    for j, word in enumerate(words):
                        if j < max_word_in_sent:
                            doc[i][j] = vocab.get(word, UNKNOWN)

            label = type1
            labels = [0] * num_classes
            labels[label - 1] = 1
            data_y.append(labels)
            data_x.append(doc.tolist())
        logger.info("results has iterated" + str(iter))
        pickle.dump((data_x, data_y), open('han_type1_data_' + str(iter), 'wb'))
    logger.info("pickle dump end")

except:
    print("Error: unable to fetch data")

# 关闭数据库连接
db.close()
