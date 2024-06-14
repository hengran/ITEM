import json
from openpyxl import Workbook

# 创建一个工作簿
wb = Workbook()

# 激活工作表
ws = wb.active

# 写入表头
ws.append(['file','pre','rec', 'f1'])
file_list = [
]
data = []
for file_name in file_list:
    file = open(file_name, "r", encoding="utf-8")
    pre_all = 0
    rec_all = 0
    num = 0
    query_set = set()
    if "nq" in file_name:
        max_label = 1
    elif "antique" in file_name:
        max_label = 4
    else:
        max_label = 3
    print(max_label)
    for line in file:
        # print(line)
        # if num >= 500:
        #     continue
        js = json.loads(line)
        if js["question"] in query_set:
                continue
        query_set.add(js["question"])
        question = js["question"]
        model_answer = js["model_out_label"]
        if "nq" in file_name:
            max_label = 1
        elif "antique" in file_name:
            max_label = 4
        else:
            max_label = 3
        ground_truth_label = [1 if i>=max_label else 0 for i in js["ground_truth_label"]]
        acc = 0
        for i in range(len(model_answer)):
            if model_answer[i] == ground_truth_label[i] and ground_truth_label[i]==1:
                acc += 1
        ground_num = sum(ground_truth_label)
        pre_num = sum(model_answer)
        if pre_num == 0:
            pre = 0
        else:
            pre = acc/pre_num
        if ground_num == 0 :
            rec = 0
        else:
            rec = acc/ground_num
        pre_all += pre
        rec_all += rec
        num += 1
    print("\n")
    print("num: ", num)
    pre = 100*pre_all / num
    rec = 100*rec_all / num
    f1 = 2 * (pre * rec) / (pre + rec)
    print(file_name.split("/")[-1])
    print("pre is: {pre}".format(pre=pre))
    print("rec is: {rec}".format(rec=rec))
    print("f1 is: {f1}".format(f1=f1))
    data.append({"file_name":file_name.split("/")[-1],"pre": pre, "rec":rec, "f1": f1})
    print("-------------------------------------------")

# 将数据写入工作表
for d in data:
    ws.append([d['file_name'], d['pre'], d['rec'], d['f1']])

# 保存工作簿
wb.save('pointwise-output.xlsx')



