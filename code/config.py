#!/usr/bin/python
# -*- coding: UTF-8 -*-
config = {
"109": {
    "db_addr": "localhost",
    "db_user": "root",
    "db_password": "irlab2017",
    "db_name": "graduate",
    "db_charSet": "utf8",
    "sent_tokenizer_path": r"D:/Program Files (x86)/anaconda/nltkdata/english.pickle",
    "num_classes": 11,
    "vocab_size": 625293,
    "min_word_freq":5,
    "max_sent_in_doc":30,
    "max_word_in_sent":30,
    "gen_pickle_iter_num":200000,
    "section_title_class_index_path":"sectionTitle2ClassIndex.xlsx"
  },
"104": {
    "db_addr": "localhost",
    "db_user": "root",
    "db_password": "irlab2017",
    "db_name": "graduate",
    "db_charSet": "utf8",
    "sent_tokenizer_path": r"'/home/wangxin/sshFile/nltk_data/tokenizers/punkt/english.pickle'",
    "num_classes": 11,
    "vocab_size": 625293,
    "min_word_freq":5,
    "max_sent_in_doc":30,
    "max_word_in_sent":30,
    "gen_pickle_iter_num":200000,
    "section_title_class_index_path":"sectionTitle2ClassIndex.xlsx"
  },
"10": {
    "db_addr": "localhost",
    "db_user": "root",
    "db_password": "aixuexi109",
    "db_name": "graduate",
    "db_charSet": "utf8",
    "sent_tokenizer_path": r"/home/aixuexi/PycharmProjects/AZR/code/english.pickle",
    "num_classes": 11,
    "vocab_size": 625293,
    "min_word_freq":5,
    "max_sent_in_doc":30,
    "max_word_in_sent":30,
    "gen_pickle_iter_num":200000,
    "section_title_class_index_path":"./sectiontitle2classindex.xlsx"
  }

}

index2class_name = {
    1:'引言 Introduction',
    2:'背景 Background',
    3:'方法 Method',
    4:'实验结果 Result',
    5:'结论 Conclusion',
    6:'实验材料 Material',
    7:'实验病人 Patients',
    8:'病例相关 Case',
    9:'实验 Experiment',
    10:'参考文献 Literature',
    11:'研究不足 Limatation'
}