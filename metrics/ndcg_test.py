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
from sklearn.metrics import ndcg_score
wb = Workbook()
ws = wb.active
ws.append(['file',"ndcg1", "ndcg3", "ndcg5", "ndcg10", "ndcg20"])
def dcg_at_k(r, k, method=0):
    """
    Computes the Discounted Cumulative Gain (DCG) at rank k.
    Args:
    r (list): Relevance scores (list or numpy) in rank order (first element is the top-ranked item).
    k (int): Number of results to consider.
    method (int): If 0, use the standard method. If 1, use the alternative method.
    
    Returns:
    float: DCG value.
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    """
    Computes the Normalized Discounted Cumulative Gain (nDCG) at rank k.
    Args:
    r (list): Relevance scores (list or numpy) in rank order (first element is the top-ranked item).
    k (int): Number of results to consider.
    method (int): If 0, use the standard method. If 1, use the alternative method.
    
    Returns:
    float: nDCG value.
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

file_list = []
data = []

labels_list = []
for file_name in file_list:
    file = open(file_name, "r", encoding = "utf-8")
    ndcg_1s = []
    ndcg_3s = []
    ndcg_5s = []
    ndcg_10s = []
    ndcg_20s = []
    # key = "relevance_labels"
    key = "utility_labels"
    # key = "ground_truth_label"
    # key = "labels"
    print(key)
    for line in file:
        js = json.loads(line)
        # print(js)
        # labels_list.append(js[key])
        # if max(js[key]) < 3:
        #     if len(js["gold_ctxs"]) == 0:
        #         continue
        #     else:
        #         js[key][-1] = 3  
        labels = js[key]
        question = js["question"]
        ndcg_1 = []
        ndcg_3 = []
        ndcg_5 = []
        ndcg_10 = []
        ndcg_20 = []
        # if max(js[key]) < 3 and len(js["gold_ctxs"]) == 0:
        #     continue
        for label_list in labels:
            # print(label_list)
            label = [0 if i==-1 else i for i in label_list]
            true_relevance = np.asarray([label])
            score = []
            for i in range(len(label)):
                score.append(len(label)-i+1)
            scores = np.asarray([score])
            # print(label)
            ndcg = ndcg_score(true_relevance, scores, k=1)
            ndcg_1.append(ndcg)
            ndcg = ndcg_score(true_relevance, scores, k=3)
            ndcg_3.append(ndcg)
            ndcg = ndcg_score(true_relevance, scores, k=5)
            ndcg_5.append(ndcg)
            ndcg = ndcg_score(true_relevance, scores, k=10)
            ndcg_10.append(ndcg)
            ndcg = ndcg_score(true_relevance, scores, k=20)
            ndcg_20.append(ndcg)
        ndcg_1s.append(ndcg_1)
        ndcg_3s.append(ndcg_3)
        ndcg_5s.append(ndcg_5)
        ndcg_10s.append(ndcg_10)
        ndcg_20s.append(ndcg_20)
    for i in [0,1,2,3,4]:
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
wb.save('trec-code/mistral-sun-ndcg_test_relevance_ncdg.xlsx')
# print(labels_list)
        
        
