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
ws.append(['file','pre', 'rec', 'f1', 'em',"a_f1", 'ha'])
 
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
def Get_rouge(pre_answers, answer_generation):
    rouge = Rouge()
    rouge_score = rouge.get_scores(pre_answers, answer_generation)
    return rouge_score[0]["rouge-l"]['f']
for file_name in file_list:
    i_rounds = {}
    x_data = [1,2,3,4,5]
    y_per = []
    y_rec = []
    y_f1 = []
    yy_pre = []
    yy_rec = []
    yy_f1 = []
    querys = []
    for k in [1,2,3,4,5]:  
        query_set = set()
        file = open(file_name, "r", encoding="utf-8")
        pres, recs, f1s, acc, has, original, pre_pass, ems, a_f1s = [], [], [] ,[] ,[] ,[] ,[], [], []
        num = 0
        for line in file:
            js = json.loads(line)
            min_len = len(js["output_all"])
            # if js["question"] in data_non_fact:
            #     continue
            if js["question"] in query_set:
                continue
            query_set.add(js["question"])
            sp = js["output_all"][min(min_len-1, k-1)]
            querys.append(js["question"])
            response = clean_response(sp)
            if isinstance(response, list):
                response = [int(x)-1 for x in response]
                response = remove_duplicate(response)
            else:
                response = [int(x)-1 for x in response.split()]
                response = remove_duplicate(response)
            if "nq" in file_name:
                ground_truth_label = query_labels[js["question"]][:10]
                if max(ground_truth_label) < 1:
                    continue
            else:
                ground_truth_label = js["ground_truth_label"]
            if "nq" in file_name:
                max_label = 1
            else:
                max_label = 3
            ground_truth = []
            # assert max_label == max(ground_truth_label)
            for index, label in enumerate(ground_truth_label):
                if label == max_label:
                    ground_truth.append(index)
            original.append(len(ground_truth)/len(ground_truth_label))
            pres.append(get_pres(response, ground_truth))
            recs.append(get_recs(response, ground_truth))
            passages = js["passage"]
            acc_pas = 0
#             if "nq" in file_name:
#                 ground_answers = query_answer[js["question"]]
#                 for passage in passages:
#                     if has_answer(ground_answers, passage) == 1:
#                         acc_pas +=1
                        
#                 if "Answer:" in js["answer_generations"][min(min_len-1, k-1)]:
#                     answer = js["answer_generations"][min(min_len-1, k-1)].split("Answer:")[1]
#                 else:
#                     answer = js["answer_generations"][min(min_len-1, k-1)]
#                 ems.append(EM_compute(ground_answers,answer))
#                 has.append(has_answer(ground_answers, answer))
#                 a_f1s.append(F1_compute(ground_answers, answer))
            # pre_pass.append(acc_pas/len(passages))
            num += 1
        print("iter ", k, " times: ")
        print(len(pres))
        pre = 100*sum(pres)/len(pres)
        rec = 100*sum(recs)/len(recs)
        f1 = 2*pre*rec/(pre+rec)
        # acc = sum(acc)/len(acc)
        print(file_name)
        print("pre: ", pre)
        print("rec: ", rec)
        print("macro-f1: ", 2*pre*rec/(pre+rec))
        y_per.append(pre)
        y_rec.append(rec)
        y_f1.append(2*pre*rec/(pre+rec))
        print("-------------------------------------------")
        print("original: ", 100*sum(original)/len(original))
        # print("has_original: ", 100*sum(pre_pass)/len(pre_pass))
        # if "nq" in file_name:
        #     em = 100*sum(ems)/len(ems)
        #     ha = 100*sum(has)/len(has)
        #     a_f1 = 100*sum(a_f1s)/len(a_f1s)
        #     print("em: ", em)
        #     print("a_f1: ", a_f1)
        #     print("ha: ", ha)
        # else:
        em = 0
        a_f1 = 0
        ha = 0
        data.append({"file_name":file_name,"pre": pre, "rec":rec, "f1": f1, "em":em, "a_f1":a_f1, "ha": ha})
    print("-------------------------------------------")
    # print("original: ", 100*sum(original)/len(original))

# 将数据写入工作表
for d in data:
    ws.append([d['file_name'], d['pre'], d['rec'], d['f1'], d['em'], d["a_f1"], d['ha']])

# 保存工作簿
wb.save('trec-code/mistral_webap_listwise-iter-output.xlsx')
