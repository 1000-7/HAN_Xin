#!/usr/bin/python
# -*- coding: UTF-8 -*-
import logging
import nltk
import pymysql
from nltk.tokenize import WordPunctTokenizer
from config import config
import pickle
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
config = config["104"]
db = pymysql.connect(config["db_addr"],
                     config["db_user"],
                     config["db_password"],
                     config["db_name"],
                     charset=config["db_charSet"])

# 使用cursor()方法获取操作游标
cursor = db.cursor()
sent_tokenizer = nltk.data.load(config["sent_tokenizer_path"])  # 加载英文的划分句子的模型
word_tokenizer = WordPunctTokenizer()  # 加载英文的划分单词的模型

def paper_section_stat():
    paper_section_dict = {}

    for i in range(0, 10):
        logger.info("now batch is:" + str(i))
        sql = "SELECT * FROM semanticScholar_filter limit " + str(i * 200000) + "," + str(200000) + ";"
        cursor.execute(sql)

        # 获取所有记录列表
        results = cursor.fetchall()
        for row in results:
            paper_section_title = row[5]
            if paper_section_title in paper_section_dict:
                paper_section_dict[paper_section_title] += 1
            else:
                paper_section_dict[paper_section_title] = 1

    logger.info("begin print to file")
    fileName = 'paper_section_dict.txt'
    with open(fileName, 'a+', encoding='utf-8') as f:
        for key, value in paper_section_dict.items():
            f.write(key + "###" + str(value) + '\n')
    print("paper_section_dict 生成成功")
    # 关闭数据库连接
    db.close()
def genVocabulary():
    """
    主要功能
    1、统计所有的word的词频
    2、保留词频大于 min_word_freq 的词语，构建word - index的字典。
    :return:
    """
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
            print("word_freq save finished")
        # 保存word-index字典
        vocab = {}
        i = 1
        for word, freq in word_freq.items():
            if freq > config["min_word_freq"]:
                if word not in vocab.keys():
                    vocab[word] = i
                    i += 1
        logger.info("vocab len " + str(len(vocab) + 1))
        pickle.dump(vocab, open('word2index_dict', 'wb'))
        logger.info("vocab build finished")

    except:
        print("Error: unable to fetch data")

    # 关闭数据库连接
    db.close()
def data_stat():
    """
    统计每个文本平均句子数量，每个句子的平均词语数量
    :return:
    """
    # SQL 查询语句
    sql = "SELECT * FROM semanticScholar_filter"
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        i = 0
        totalSentenceNum = 0
        totalWordNum = 0
        for row in results:
            paperSection = row[2].replace("- ", "")
            type1 = row[6]
            words = word_tokenizer.tokenize(paperSection)
            if len(words) < 50:
                continue
            i += 1
            sentences = sent_tokenizer.tokenize(paperSection)  # 将评论分句
            totalSentenceNum += len(sentences)
            totalWordNum += len(words)

        print("i:%s, sentenceTotalNum:%s, wordTotalNum:%s, sentenceLen:%s, sectionLen:%s" % (i,
                                                                                             totalSentenceNum,
                                                                                             totalWordNum,
                                                                                             totalWordNum / totalSentenceNum,
                                                                                             totalSentenceNum / i))
    except:
        print("Error: unable to fetch data")

    # 关闭数据库连接
    db.close()
