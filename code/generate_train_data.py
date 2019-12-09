#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
import nltk
import numpy as np
import pymysql
from nltk.tokenize import WordPunctTokenizer
import logging
import pandas as pd
from code.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = config["10"]
max_sent_in_doc = config["max_sent_in_doc"]
max_word_in_sent = config["max_word_in_sent"]
iterNum = config["gen_pickle_iter_num"]
# 加载标签类别字典
dict = pd.read_excel(config["section_title_class_index_path"])
sheet = dict.values.tolist()
sectTitle2classindex_dict = {}
sampleNumControler = {}
for item in sheet:
    sectTitle2classindex_dict[item[0]] = item[1]
    sampleNumControler[item[0]] = [item[3],0]

vocab = pickle.load(open('word2index_dict', 'rb'))
UNKNOWN = 0
db = pymysql.connect(config["db_addr"],
                     config["db_user"],
                     config["db_password"],
                     config["db_name"],
                     charset=config["db_charSet"])

sent_tokenizer = nltk.data.load(config["sent_tokenizer_path"])  # 加载英文的划分句子的模型
word_tokenizer = WordPunctTokenizer()  # 加载英文的划分单词的模型

def data_trans(paperSection,sentType):
    """
    将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
    多余的删除，并保存到最终的数据集文件之中
    注意，少于30的并没有进行补零操作
    :param paperSection:
    :param sentType:
    :return:
    """
    if sampleNumControler(sentType):
        return None
    doc = np.zeros((max_sent_in_doc,
                    max_word_in_sent),
                   dtype=np.int32)

    sentences = sent_tokenizer.tokenize(paperSection)  # 将评论分句
    words = word_tokenizer.tokenize(paperSection) # 将评论分词
    if len(words) > 50 and len(sentences) >= 2:
        for i, sent in enumerate(sentences):
            if i < max_sent_in_doc:
                words = word_tokenizer.tokenize(sent)
                for j, word in enumerate(words):
                    if j < max_word_in_sent:
                        doc[i][j] = vocab.get(word, UNKNOWN)

        label = sectTitle2classindex_dict[sentType]
        labels = [0] * config["num_classes"]
        labels[label - 1] = 1
        return [doc.tolist(),labels]
    return None

def sampleNumControler(sentType):
    """
    对每一类的训练样本数量进行控制
    :param sentType:
    :return:
    """
    temp = sampleNumControler[sentType]
    max_sample_num = temp[0]
    now_sample_num = temp[1]+1
    if now_sample_num <= max_sample_num:
        return True
    else:
        return False

def genPickleData():
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
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
            data_x = []
            data_y = []

            finish_num = 0
            for row in results:
                sentType = row[5]
                if sentType in sectTitle2classindex_dict.keys():
                    paperSection = row[2].replace("- ", "")
                    temp  = data_trans(paperSection,sentType)
                    if not temp is None:
                        data_y.append(temp[1])
                        data_x.append(temp[0])
                        if finish_num % 10000 == 0:
                            logger.info(str(finish_num) + " / " + str(iterNum) + " has finished")
                        finish_num += 1

            logger.info("results has iterated" + str(iter))
            pickle.dump((data_x, data_y), open('../tarindata/train_data_' + str(iter), 'wb'))
        logger.info("pickle dump end")
    except:
        print("Error: unable to fetch data")
        logging.exception('Got exception on main handler')
    # 关闭数据库连接
    db.close()
if __name__ == "__main__":
    genPickleData()
