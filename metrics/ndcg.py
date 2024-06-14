import numpy as np
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
ws.append(['file',"ndcg1", "ndcg3", "ndcg5", "ndcg10", "ndcg20"])

# Example usage
if __name__ == "__main__":
    # Example relevance scores
    relevance_scores = [3, 2, 3, 0, 1, 2]
    k = 5  # Number of results to consider

    # Compute nDCG
    ndcg_score = ndcg_at_k(relevance_scores, k)
    print(f"nDCG@{k} = {ndcg_score:.4f}")


file_list = []
data = []
def get_ndcg(question, labels):
    qrel[question] = {}
    for i in range(len(labels)):
        qrel[question]['d'+str(i)] = labels[i]
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut.1'})
    ndcg_at_1 = 0
    ndcg_at_5 = 0
    ndcg_at_10 = 0
    ndcg_at_20 = 0
    run = {}
    run[question] = {}
    for i in range(len(labels)):
        run[question]['d'+str(i)] = len(labels)-i
    results = evaluator.evaluate(run)
    for query in evaluator.evaluate(run).keys():
        ndcg_at_1 = results[query]["ndcg_cut_1"]

    evaluator = pytrec_eval.RelevanceEvaluator(
            qrel,  pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)
    for query in evaluator.evaluate(run).keys():
        ndcg_at_10 = results[query]["ndcg_cut_10"]
        ndcg_at_5 = results[query]["ndcg_cut_5"]
        ndcg_at_20 = results[query]["ndcg_cut_20"]
    return ndcg_at_1, ndcg_at_1, ndcg_at_10, ndcg_at_20
labels_list = []
for file_name in file_list:
    file = open(file_name, "r", encoding = "utf-8")
    ndcg_1s = []
    ndcg_3s = []
    ndcg_5s = []
    ndcg_10s = []
    ndcg_20s = []
    key = "relevance_labels"
    # key = "ground_truth_label"
    # key = "labels"
    print(key)
    for line in file:
        js = json.loads(line)
        # if max(js[key]) < 3:
        #     if len(js["gold_ctxs"]) == 0:
        #         continue
        #     else:
        #         js[key][-1] = 3  
        # labels_list.append(js[key])
        labels = js[key]
        question = js["question"]
        ndcg_1 = []
        ndcg_3 = []
        ndcg_5 = []
        ndcg_10 = []
        ndcg_20 = []
        for label_list in labels:
            
            label = [0 if i==-1 else i for i in label_list]
            # print(label)
            ndcg_score = ndcg_at_k(label, 1)
            ndcg_1.append(ndcg_score)
            ndcg_score = ndcg_at_k(label, 3)
            ndcg_3.append(ndcg_score)
            ndcg_score = ndcg_at_k(label, 5)
            ndcg_5.append(ndcg_score)
            ndcg_score = ndcg_at_k(label, 10)
            ndcg_10.append(ndcg_score)
            ndcg_score = ndcg_at_k(label, 20)
            ndcg_20.append(ndcg_score)
        ndcg_1s.append(ndcg_1)
        ndcg_3s.append(ndcg_3)
        ndcg_5s.append(ndcg_5)
        ndcg_10s.append(ndcg_10)
        ndcg_20s.append(ndcg_20)
    for i in [0, 1, 2, 3, 4]:
        ndcg_1 = []
        ndcg_3 = []
        ndcg_5 = []
        ndcg_10 = []
        ndcg_20 = []
        for j in range(len(ndcg_1s)):
            ndcg_1.append(ndcg_1s[j][min(i,len(ndcg_1s[j])-1)])
            ndcg_3.append(ndcg_3s[j][min(i,len(ndcg_3s[j])-1)])
            ndcg_5.append(ndcg_5s[j][min(i,len(ndcg_5s[j])-1)])
            ndcg_10.append(ndcg_10s[j][min(i,len(ndcg_10s[j])-1)])
            ndcg_20.append(ndcg_20s[j][min(i,len(ndcg_20s[j])-1)])
        ndcg1 =  100*sum(ndcg_1)/len(ndcg_1)
        ndcg3 =   100*sum(ndcg_3)/len(ndcg_3)
        ndcg5 =   100*sum(ndcg_5)/len(ndcg_5)
        ndcg10 =   100*sum(ndcg_10)/len(ndcg_10)
        ndcg20 =   100*sum(ndcg_20)/len(ndcg_20)
        print(file_name)
        print(len(ndcg_1))
        print("ndcg_1: ", ndcg1)
        print("ndcg_3: ", ndcg3)
        print("ndcg_5: ", ndcg5)
        print("ndcg_10: ", ndcg10)
        print("ndcg_20: ", ndcg20)
        data.append({"file_name":file_name,"ndcg1": ndcg1, "ndcg3":ndcg3, "ndcg5": ndcg5, "ndcg10":ndcg10, "ndcg20":ndcg20})
        print("-------------------------------------------")
    # print("original: ", 100*sum(original)/len(original))

# 将数据写入工作表
for d in data:
    ws.append([d['file_name'], d['ndcg1'], d['ndcg3'], d['ndcg5'], d['ndcg10'], d["ndcg20"]])

# 保存工作簿
wb.save('trec-code/mistral_webap_relevance_ncdg.xlsx')
# print(labels_list)
        
        
