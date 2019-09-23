#!/usr/bin/python
# -*- coding: UTF-8 -*-
from collections import defaultdict

import nltk
import pymysql
from nltk.tokenize import WordPunctTokenizer

# 统计基本信息
# 打开数据库连接
db = pymysql.connect("localhost", "root", "irlab2017", "graduate", charset='utf8')

# 使用cursor()方法获取操作游标
cursor = db.cursor()

sent_tokenizer = nltk.data.load('/home/wangxin/sshFile/nltk_data/tokenizers/punkt/english.pickle')  # 加载英文的划分句子的模型
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
