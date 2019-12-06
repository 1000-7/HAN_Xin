import pandas as pd

fileName = 'paper_section_dict.csv'
paper_section_dict = {}
paper_section_dict["aaa"] = 1
paper_section_dict["abb"] = 2
with open(fileName, 'a+', encoding='utf-8') as f:
    for key, value in paper_section_dict.items():
        f.write(key + "###" + str(value) + '\n')
