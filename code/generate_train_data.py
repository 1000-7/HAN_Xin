#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
import nltk
import numpy as np
import pymysql
from nltk.tokenize import WordPunctTokenizer
import logging
import pandas as pd
from config import config
import random
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

dict = pd.read_excel(config["test_num_control"])
sheet = dict.values.tolist()
testNumControler = {}
for item in sheet:
    testNumControler[item[0]] = [item[1],0]

vocab = pickle.load(open('../traindata/word2index_dict', 'rb'))
UNKNOWN = 0
db = pymysql.connect(config["db_addr"],
                     config["db_user"],
                     config["db_password"],
                     config["db_name"],
                     charset=config["db_charSet"])

sent_tokenizer = nltk.data.load(config["sent_tokenizer_path"])  # 加载英文的划分句子的模型
word_tokenizer = WordPunctTokenizer()  # 加载英文的划分单词的模型

def data_trans(paperSection,sentType,train_data = True):
    """
    将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
    多余的删除，并保存到最终的数据集文件之中
    注意，少于30的并没有进行补零操作
    :param paperSection:
    :param sentType:
    :return:
    """

    if not sampleNumControl(sentType,train_data):
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

def sampleNumControl(sentType,train_data=True):
    """
    对每一类的训练样本数量进行控制
    :param sentType:
    :return:
    """
    if train_data:
        temp = sampleNumControler[sentType]
        max_sample_num = temp[0]
        now_sample_num = temp[1]+1
        sampleNumControler[sentType][1] = now_sample_num
        if now_sample_num <= max_sample_num:
            return True
        else:
            return False
    else:
        temp = testNumControler[sectTitle2classindex_dict[sentType]]
        max_sample_num = temp[0]
        now_sample_num = temp[1] + 1
        testNumControler[sectTitle2classindex_dict[sentType]][1] = now_sample_num
        if now_sample_num <= max_sample_num:
            return True
        else:
            return False

        pass

def gen_train_data_pickle():
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句
    try:
        vocab['UNKNOW_TOKEN'] = 0
        for iter in range(0, 10):
            logger.info("now batch is:" + str(iter))
            sql = "SELECT * FROM semanticScholar_filter limit " + str(iter * iterNum) + "," + str(iterNum)
            logger.info(sql)
            # 执行SQL语句
            cursor.execute(sql)
            # 获取所有记录列表
            results = cursor.fetchall()
            logger.info("cursor fetch finished " + str(iter+1 ))
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
            pickle.dump((data_x, data_y), open('../traindata/train_data_' + str(iter), 'wb'))
        logger.info("pickle dump end")
    except:
        print("Error: unable to fetch data")
        logging.exception('Got exception on main handler')
    # 关闭数据库连接
    db.close()
def shuffle_and_balance_data():
    data_x = []
    data_y = []
    for iter in range(0, 10):
        x, y = pickle.load(open('../traindata/train_data_' + str(iter), 'rb'))
        data_x.extend(x)
        data_y.extend(y)
    # classNum = np.sum(data_y,axis=0)
    # print(classNum)
    # [19760 19361 19678 19929 19092 19937 19926  8261  4097  4080  6591]
    index_list = list(range(len(data_y)))
    iter = 0
    random.shuffle(index_list)
    linspace =np.linspace(0, len(data_y)-1,11,dtype=np.int64)
    for i,index in enumerate(linspace[0:len(linspace)-1]):
        x = []
        y = []
        for il in index_list[index:linspace[i+1]]:
            x.append(data_x[il])
            y.append(data_y[il])
        print(np.sum(y,axis=0))
        pickle.dump((x, y), open('../traindata/train_data_balance_' + str(iter), 'wb'))
        iter = iter+1
    logger.info("pickle dump end")
    # 每个pickle中 各个类别对应的数量
    # [1939 1980 1973 1966 1859 2071 1987  841  418  387  650]
    # [2011 1928 2058 2002 1905 1970 2029  828  354  378  608]
    # [2034 1924 1929 2005 1958 1957 2006  782  393  384  699]
    # [2031 1934 1982 1978 1922 1893 2003  808  397  420  703]
    # [1974 1933 1944 1940 1909 2081 1988  827  436  398  641]
    # [1946 1997 1886 2021 1939 1986 1979  840  405  416  656]
    # [1934 1971 1943 2082 1840 1984 1950  843  427  427  670]
    # [1983 1968 1989 1983 1823 2008 1995  803  438  427  654]
    # [1968 1888 1944 1967 1948 2017 1985  842  426  426  660]
    # [1940 1838 2030 1985 1989 1970 2004  847  403  416  650]


def gen_test_data_pickle():
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # SQL 查询语句
    try:
        data_x = []
        data_y = []
        vocab['UNKNOW_TOKEN'] = 0
        iterList = list(range(0, 10))
        iterList = iterList[::-1]
        for iter in iterList:
            logger.info("now batch is:" + str(iter))
            sql = "SELECT * FROM semanticScholar_filter limit " + str(iter * iterNum) + "," + str(iterNum)
            logger.info(sql)
            # 执行SQL语句
            cursor.execute(sql)
            # 获取所有记录列表
            results = cursor.fetchall()
            logger.info("cursor fetch finished " + str(iter + 1))
            # 构建vocabulary，并将出现次数小于5的单词全部去除，视为UNKNOW


            finish_num = 0
            for row in results:
                sentType = row[5]
                if sentType in sectTitle2classindex_dict.keys():
                    paperSection = row[2].replace("- ", "")
                    temp = data_trans(paperSection, sentType,False)
                    if not temp is None:
                        data_y.append(temp[1])
                        data_x.append(temp[0])
                        if finish_num % 10000 == 0:
                            logger.info(str(finish_num) + " / " + str(iterNum) + " has finished")
                        finish_num += 1

            logger.info("results has iterated" + str(iter))
        pickle.dump((data_x, data_y), open('../traindata/test_data', 'wb'))
        print(np.sum(data_y, axis=0))
        logger.info("pickle dump end")
    except:
        print("Error: unable to fetch data")
        logging.exception('Got exception on main handler')
    # 关闭数据库连接
    db.close()

if __name__ == "__main__":
    gen_test_data_pickle()
