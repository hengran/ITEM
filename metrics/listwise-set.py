import json
import re
from openpyxl import Workbook
import sys
import os
sys.path.append(".")
# 创建一个工作簿
wb = Workbook()

# 激活工作表
ws = wb.active

# 写入表头
ws.append(['file','pre', 'rec', 'f1'])

 
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
data = [] 

for file_name in file_list:
    file = open(file_name, "r", encoding="utf-8")
    pres = []
    recs = []
    f1s = []
    acc = []
    original = []
    original_pre, original_rec, original_f1 = [], [], []
    num = 0
    kk = 1
    for line in file:
        # if num >= 500:
        #     continue
        js = json.loads(line)
        sp = js["LLM_output_all"].lower()
        response = clean_response(sp)
        if isinstance(response,list):
            response = [int(x)-1 for x in response]
            response = remove_duplicate(response)
        else:
            response = [int(x)-1 for x in response.split()]
            response = remove_duplicate(response)
        ground_truth_label = js["ground_truth_label"]
        ground_truth = []
        if "nq" in file_name:
            max_label = 1
        elif "antique" in file_name:
            max_label = 4
        else:
            max_label = 3
        ground_truth = []
        for index, label in enumerate(ground_truth_label):
            if label >= max_label:
                ground_truth.append(index)
        temp = ground_truth[:kk]
        original.append(len(ground_truth)/len(ground_truth_label))
        # original_pre.append(sum(temp)/len(temp))
        # original_rec.append(sum(temp)/sum(ground_truth))
        pres.append(get_pres(response, ground_truth))
        recs.append(get_recs(response, ground_truth))
        acc.append(response == ground_truth)
        num += 1
    print(len(pres))
    pre = 100*sum(pres)/len(pres)
    rec = 100*sum(recs)/len(recs)
    f1 = 2*pre*rec/(pre+rec)
    acc = sum(acc)/len(acc)
    print(file_name.split("/")[-1])
    print("pre: ", pre)
    print("rec: ", rec)
    print("macro-f1: ", 2*pre*rec/(pre+rec))
    data.append({"file_name":file_name.split("/")[-1],"pre": pre, "rec":rec, "f1": f1})
    # print("original_pre: ", op)
    # print("original_pre: ", orr)
    # print("original_f1: ", of1)
    print("-------------------------------------------")
    print("original: ", 100*sum(original)/len(original))

# 将数据写入工作表
for d in data:
    ws.append([d['file_name'], d['pre'], d['rec'], d['f1']])

# 保存工作簿
wb.save('listwise-set-output.xlsx')
