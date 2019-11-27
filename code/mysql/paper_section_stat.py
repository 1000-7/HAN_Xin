import pymysql
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
db = pymysql.connect("localhost", "root", "irlab2017", "graduate", charset='utf8')

# 使用cursor()方法获取操作游标
cursor = db.cursor()

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
