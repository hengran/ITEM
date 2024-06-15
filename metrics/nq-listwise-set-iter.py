import json
import os
import sys
sys.path.append(".")
import re
from openpyxl import Workbook
import matplotlib.pyplot as plt
from utils.utils import EM_compute, has_answer, F1_compute
import numpy as np
import random

wb = Workbook()
ws = wb.active
ws.append(['file', 'em',"a_f1", 'ha'])
 
def extract_numbers(text):
    numbers = re.findall('\d+', text)
    return numbers

def extract_substrings(input_string):
    pattern = re.compile(r'\[\d+\]')
    substrings = pattern.findall(input_string)
    extracted_string = ''.join(substrings)
    return extracted_string

def clean_response(sp: str):
    response = extract_substrings(sp.lower())
    if len(response) == 0:
        response = extract_numbers(sp.lower())
        new_response = []
        for res in response:
            if int(res)>20 or int(res)<=0:
                continue
            else:
                new_response.append(int(res))
        return new_response
    else:
        new_response = ''
        for c in response:
            if not c.isdigit():
                new_response += ' '
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response
def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response
file_list = [
]
def get_pres(response, ground_truth):
    acc = len([value for value in response if value in ground_truth])
    return acc/len(response) if len(response) > 0 else 0
def get_recs(response, ground_truth):
    acc = len([value for value in response if value in ground_truth])
    return acc/len(ground_truth) if len(ground_truth) > 0 else 0
# 打开本地的json文件
with open('data/non_factoid_trec1920.json') as f:
    data_non_fact = json.load(f)
query_answer = {}
file1 = open("data/nq.json", "r", encoding="utf-8")
for line in file1:
    js = json.loads(line)
    query_answer[js["question"]] = js["answer"]
data_non_fact = list(data_non_fact.keys())
data = []
for file_name in file_list:
    querys = []
    for k in [0,1,2,3,4,5]:  
        query_set = set()
        file = open(file_name, "r", encoding="utf-8")
        pre_pass = []
        ems = []
        a_f1s = []
        has = []
        num = 0
        for line in file:
            js = json.loads(line)
            passages = js["passage"]
            acc_pas = 0
            min_len = len(js["answer_generation_ems"])
            ground_answers = query_answer[js["question"]]
            ems.append(EM_compute(ground_answers, js["answer_generation_ems"][min(min_len-1, k)]))
            has.append(has_answer(ground_answers, js["answer_generation_ems"][min(min_len-1, k)]))
            a_f1s.append(F1_compute(ground_answers, js["answer_generation_ems"][min(min_len-1, k)]))
            num += 1 
        print("iter ", k, " times: ")
        print(len(ems))
        print(file_name)
        em = 100*sum(ems)/len(ems)
        ha = 100*sum(has)/len(has)
        a_f1 = 100*sum(a_f1s)/len(a_f1s)
        print("em: ", em)
        print("a_f1: ", a_f1)
        print("ha: ", ha)
        data.append({"file_name":file_name,"em":em, "a_f1":a_f1, "ha": ha})
    print("-------------------------------------------")
    # print("original: ", 100*sum(original)/len(original))

# 将数据写入工作表
for d in data:
    ws.append([d['file_name'], d['em'], d["a_f1"], d['ha']])

# 保存工作簿
wb.save('mistral_relevance_listwise-iter-output.xlsx')
