import json
import os
import sys
sys.path.append(".")
import re
from openpyxl import Workbook
from utils.utils import EM_compute, has_answer, F1_compute
import matplotlib.pyplot as plt
import numpy as np
import random

wb = Workbook()
ws = wb.active
ws.append(['file','pre', 'rec', 'f1', 'em',"a_f1", 'ha', 'pre@', 'rec@', 'f1@'])
 
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
file3 = open("data/NQ_qa_label.json", "r", encoding="utf-8")
query_labels = {}
for line in file3:
    js= json.loads(line)
    query_labels[js["question"]] = js["dense_label_qa"]
def get_pres(response, ground_truth):
    acc = len([value for value in response if value in ground_truth])
    return acc/len(response) if len(response) > 0 else 0
def get_recs(response, ground_truth):
    acc = len([value for value in response if value in ground_truth])
    return acc/len(ground_truth) if len(ground_truth) > 0 else 0
data = [] 
# 打开本地的json文件
with open('data/non_factoid_trec1920.json') as f:
    data_non_fact = json.load(f)
data_non_fact = list(data_non_fact.keys())
# 打开本地的json文件
file1 = open("data/nq.json", "r", encoding="utf-8")
query_answer = {}
for line in file1:
    js = json.loads(line)
    query_answer[js["question"]] = js["answer"]
data = []
for file_name in file_list:
    i_rounds = {}
    x_data = [1,2,3,4,5,6]
    y_per = []
    y_rec = []
    y_f1 = []
    yy_pre = []
    yy_rec = []
    yy_f1 = []
    querys = []
    for k in [1,2,3,4,5,6]:
        query_set = set()
        file = open(file_name, "r", encoding="utf-8")
        pres = []
        pres_releavnce = []
        recs_releance = []
        recs = []
        f1s = []
        acc = []
        has = []
        a_f1s = []
        ems = []
        original = []
        kk = 6
        for line in file:
            js = json.loads(line)
            min_len = len(js["output_all"])
            # if js["question"] in data_non_fact: ##只有非事实类
            #     continue
            i_rounds[js["question"]] = js["i_round"]
            if js["question"] in query_set:
                continue
            query_set.add(js["question"])
            sp = js["output_all"][min(min_len-1, k-1)]
            querys.append(js["question"])
            response = clean_response(sp)
            if isinstance(response,list):
                response = [int(x)-1 for x in response]
                response = remove_duplicate(response)
            else:
                response = [int(x)-1 for x in response.split()]
                response = remove_duplicate(response)
            ground_truth_label = js["ground_truth_label"][min(k-1, len(js["ground_truth_label"])-1)]
            # print(ground_truth_label)
            # assert len(ground_truth_label) == 20
            # print(ground_truth_label)
            if "nq" in file_name:
                max_label = 1
            else:
                max_label = 3
            ground_truth = []
            for index, label in enumerate(ground_truth_label):
                if label == max_label:
                    ground_truth.append(index)
            original.append(len(ground_truth)/len(ground_truth_label))
            pres.append(get_pres(response, ground_truth))
            recs.append(get_recs(response, ground_truth))
            passages = js["passage"]
            temp = ground_truth_label[:kk]
            pres_releavnce.append(temp.count(max_label)/len(temp))
            if ground_truth_label.count(max_label) == 0:
                recs_releance.append(0)
            else:
                recs_releance.append(temp.count(max_label)/ground_truth_label.count(max_label))
            if "nq" in file_name:
                ground_answers = query_answer[js["question"]]
                ems.append(EM_compute(ground_answers, js["answer_generations"][min(min_len-1, k-1)]))
                has.append(has_answer(ground_answers, js["answer_generations"][min(min_len-1, k-1)]))
                a_f1s.append(F1_compute(ground_answers, js["answer_generations"][min(min_len-1, k-1)]))
        file.close()
        print("iter ", k, " times: ")
        print(len(pres))
        pre = 100*sum(pres)/len(pres)
        rec = 100*sum(recs)/len(recs)
        f1 = 2*pre*rec/(pre+rec)
        relevace_pre = 100*sum(pres_releavnce)/len(pres_releavnce)
        releance_rec = 100*sum(recs_releance)/len(recs_releance)
        relevance_f1 = 2*relevace_pre*releance_rec/(relevace_pre+releance_rec)
        # acc = sum(acc)/len(acc)
        print(file_name.split("/")[-1])
        print("pre: ", pre)
        print("rec: ", rec)
        print("macro-f1: ", 2*pre*rec/(pre+rec))
        # print("relevance@",kk,": ", relevace_pre)
        # print("rec@",kk,": ", releance_rec)
        # print("f1@",kk,": ", relevance_f1)
        y_per.append(pre)
        y_rec.append(rec)
        y_f1.append(2*pre*rec/(pre+rec))
        if "nq" in file_name:
            ha = 100*sum(has)/len(has)
            em = 100*sum(ems)/len(ems)
            a_f1 = 100*sum(a_f1s)/len(a_f1s)
            print("em: ", em)
            print("a_f1: ", a_f1)
            print("ha: ", ha)
        else:
            em = 0
            ha = 0
            a_f1 = 0
        print("-------------------------------------------")
        # print("original: ", 100*sum(original)/len(original))
        data.append({"file_name":file_name.split("/")[-1],"pre": pre, "rec":rec, "f1": f1, "em":em,"a_f1": a_f1, "ha": ha, 'pre@': relevace_pre, 'rec@': releance_rec, 'f1@':relevance_f1})
    print("-------------------------------------------")
    # print("original: ", 100*sum(original)/len(original))

# 将数据写入工作表
for d in data:
    ws.append([d['file_name'], d['pre'], d['rec'], d['f1'], d['em'], d["a_f1"], d['ha'], d['pre@'], d['rec@'], d['f1@']])

# 保存工作簿
wb.save('metrics_listwise-relevance-output.xlsx')
