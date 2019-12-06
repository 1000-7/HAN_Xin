#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
from collections import defaultdict

import nltk
import numpy as np
import pymysql
from nltk.tokenize import WordPunctTokenizer
import logging
import pandas as pd

config = {
    "db_addr": "localhost",
    "db_user": "root",
    "db_password": "irlab2017",
    "db_name": "graduate",
    "db_charSet": 'utf8',
    "sent_tokenizer_path": r'D:\Program Files (x86)\anaconda\nltkdata\english.pickle',
    "num_classes": 11,
}
max_sent_in_doc = 30
max_word_in_sent = 30
iterNum = 200000
path = "sectionTitle2ClassIndex.xlsx"
# 加载标签类别字典
dict = pd.read_excel(path)
sheet = dict.values.tolist()
sectTitle2classindex_dict = {}
for item in sheet:
    sectTitle2classindex_dict[item[0]] = item[1]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
db = pymysql.connect(config["db_addr"],
                     config["db_user"],
                     config["db_password"],
                     config["db_name"],
                     charset=config["db_charSet"])

sent_tokenizer = nltk.data.load('/home/wangxin/sshFile/nltk_data/tokenizers/punkt/english.pickle')  # 加载英文的划分句子的模型
word_tokenizer = WordPunctTokenizer()


def genPickleData():
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # 加载英文的划分单词的模型

    # 记录每个单词及其出现的频率
    word_freq = pickle.load(open('../word_freq.pickle', 'rb'))
    vocab = {}
    i = 1
    for word, freq in word_freq.items():
        if freq > 5:
            if word not in vocab.keys():
                vocab[word] = i
                i += 1
    logger.info("vocab len " + str(len(vocab)))
    pickle.dump(vocab, open('word2index_dict', 'wb'))

    logger.info("vocab build finished")

    # SQL 查询语句
    try:

        vocab['UNKNOW_TOKEN'] = 0
        for iter in range(1, 11):
            logger.info("now batch is:" + str(iter))
            sql = "SELECT * FROM semanticScholar_filter limit " + str(iter * iterNum) + "," + str(iterNum)
            logger.info(sql)
            # 执行SQL语句
            cursor.execute(sql)

            # 获取所有记录列表
            results = cursor.fetchall()

            logger.info("cursor fetch finished" + str(iter))
            # 构建vocabulary，并将出现次数小于5的单词全部去除，视为UNKNOW

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
                sentType = row[5]
                sentences = sent_tokenizer.tokenize(paperSection)  # 将评论分句
                words = word_tokenizer.tokenize(paperSection)
                if len(words) < 50 or len(sentences) < 2:
                    continue

                if finish_num % 10000 == 0:
                    logger.info(str(finish_num) + " / " + str(iterNum) + " has finished")

                finish_num += 1
                for i, sent in enumerate(sentences):
                    words = word_tokenizer.tokenize(sent)
                    if i < max_sent_in_doc:
                        word_to_index = []
                        words = word_tokenizer.tokenize(sent)
                        for j, word in enumerate(words):
                            if j < max_word_in_sent:
                                doc[i][j] = vocab.get(word, UNKNOWN)
                if sentType in sectTitle2classindex_dict.keys():
                    # sectiontitle2classindex.xlsx只是对高频type进行了标注
                    label = sectTitle2classindex_dict[sentType]
                    labels = [0] * config["num_classes"]
                    labels[label - 1] = 1
                    data_y.append(labels)
                    data_x.append(doc.tolist())
                else:
                    continue

            logger.info("results has iterated" + str(iter))
            pickle.dump((data_x, data_y), open('han_type1_data_' + str(iter), 'wb'))
        logger.info("pickle dump end")

    except:
        print("Error: unable to fetch data")
        logging.exception('Got exception on main handler')

    # 关闭数据库连接
    db.close()


if __name__ == "__main__":
    genPickleData()
