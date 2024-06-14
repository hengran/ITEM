import json
from openpyxl import Workbook
import json
import os
import sys
sys.path.append(".")
from utils.utils import EM_compute, has_answer, F1_compute
# 创建一个工作簿
wb = Workbook()

# 激活工作表
ws = wb.active

# 写入表头
ws.append(['file','pre','rec', 'f1', "em", "a_f1", "ha"])
file_list = [
      
]
data = []
with open('data/non_factoid_trec1920.json') as f:
    data_non_fact = json.load(f)
query_answer = {}
file1 = open("data/nq.json", "r", encoding="utf-8")
for line in file1:
    js = json.loads(line)
    query_answer[js["question"]] = js["answer"]
data_non_fact = list(data_non_fact.keys())
for file_name in file_list:
    for k in [1,2,3,4,5]:
        query_set = set()
        file = open(file_name, "r", encoding="utf-8")
        pre_all = 0
        rec_all = 0
        num = 0
        has = []
        i_num = 0
        a_f1s = []
        ems = []
        for line in file:
            js = json.loads(line)
            # if js["question"] not in data_non_fact:
            #     continue
            question = js["question"]
            if js["question"] in query_set:
                continue
            query_set.add(js["question"])
            min_len = len(js["model_out_labels"])
            model_answer = js["model_out_labels"][min(min_len-1, k-1)]
            if "nq" in file_name:
                max_label = 1
            elif "antique" in file_name:
                max_label = 4
            else:
                max_label = 3
            ground_truth_label = [1 if i==max_label else 0 for i in js["ground_truth_label"]]
            ground_num = sum(ground_truth_label)
            pre_num = sum(model_answer)
            acc = 0
            for i in range(len(model_answer)):
                if model_answer[i] == ground_truth_label[i] and ground_truth_label[i]==1:
                    acc += 1
            rec = acc/ground_num
            pre = acc/pre_num if pre_num != 0 else 0
            pre_all += pre
            rec_all += rec
            num += 1
            i_num += 1
            if "nq" not in file_name:
                continue
            else:
                # continue
                ground_answers = query_answer[js["question"]]
                ems.append(EM_compute(ground_answers, js["answer_generations"][min(min_len-1, k-1)]))
                has.append(has_answer(ground_answers, js["answer_generations"][min(min_len-1, k-1)]))
                a_f1s.append(F1_compute(ground_answers, js["answer_generations"][min(min_len-1, k-1)]))
        print("\n")
        print("iter ", k, " times: ")
        print("num: ", num)
        pre = 100*pre_all / num
        rec = 100*rec_all / num
        
        f1 = 2 * (pre * rec) / (pre + rec)
        print(file_name.split("/")[-1])
        print("pre is: {pre}".format(pre=pre))
        print("rec is: {rec}".format(rec=rec))
        print("f1 is: {f1}".format(f1=f1))
        if "nq" not in file_name:
            em = 0
            ha = 0
            a_f1 = 0
        else:
            ha = 100*sum(has)/len(has)
            em = 100*sum(ems)/len(ems)
            a_f1 = 100*sum(a_f1s)/len(a_f1s)
            print("em: ", em)
            print("a_f1: ", a_f1)
            print("ha: ", ha)
        data.append({"file_name":file_name,"pre": pre, "rec":rec, "f1": f1, "em":em, "a_f1": a_f1, "ha": ha})
    print("-------------------------------------------")
    # print("original: ", 100*sum(original)/len(original))

# 将数据写入工作表
for d in data:
    ws.append([d['file_name'], d['pre'], d['rec'], d['f1'], d['em'], d['a_f1'], d['ha']])

# 保存工作簿
wb.save('trec-code/mistral_metrics_pointwise-iter-output.xlsx')