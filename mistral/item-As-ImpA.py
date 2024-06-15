import os
import sys
sys.path.append(".")
from tqdm import tqdm
from utils.template import get_conv_template
import copy
import json
import logging
import argparse
import random
import re
from utils.utils import load_source, EM_compute, F1_compute, get_pre, get_rec, has_answer
from utils.prompt import get_prompt_multi_docs_all_pair
from vllm import LLM, SamplingParams
import transformers
from rouge import Rouge

def extract_numbers(text):
    numbers = re.findall('\d+', text)
    return numbers

def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response
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

def generate_answer_prompt(question, passages):
    pas = '\n'.join(passages)
    return [{'role': 'user', 'content': f"You are a faithful question and answer assistant. Given a question and references. To answer the question, output which information is necessary to answer the question based on the references."}, 
            {'role': 'assistant', 'content': 'Yes, i am the faithful question and answer assistant.'}, 
            {'role': 'user', 'content': f"References: {pas}\n Question: {question}\n\n To answer the question, output which information is necessary to answer the question based on the references. Do not mention references when printing out necessary information. The format of the output is: 'Necessary information: [xxx]'." },]

def get_prefix_prompt_relevance(query, num):
    return [{'role': 'user',
            'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevance to the query."},
            {'role': 'assistant',
            'content': "Yes, i am RankGPT."},
            {'role': 'user',
            'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
def get_post_prompt_relevance(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [] > [] > ..., e.g., [i] > [j] > [k] > ... Only response the ranking results, do not say any word or explain."


def get_prompt_relevance(query, passages):
    num = len(passages)
    messages = get_prefix_prompt_relevance(query, num)
    rank = 0
    for passage in passages:
        rank += 1
        content = passage
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt_relevance(query, num)})
    return messages

def get_relevance_passages(relevance_generation, passages, labels):
    def extract_substrings(input_string):
        pattern = re.compile(r'\[\d+\]')
        substrings = pattern.findall(input_string)
        extracted_string = ''.join(substrings)
        return extracted_string

    def clean_response(response: str):
        response = extract_substrings(response)
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

    response = clean_response(relevance_generation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    new_passages = []
    new_labels = []
    for i in  response:
        if i>=len(passages) or i <0:
            continue
        else:
            new_passages.append(passages[i])
            new_labels.append(labels[i])
    for i in range(20):
        if i not in response:
            new_labels.append(labels[i])
            new_passages.append(passages[i])
    return new_passages, new_labels
    
def get_args(file_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/nq.json')
    parser.add_argument('--type', type=str, default="") 
    parser.add_argument('--outfile', type=str, default='')
    args = parser.parse_args()
    return args

def get_prefix_direct_judge_point(query):
    return [{'role': 'user',
             'content': "You are the utility judger, an intelligent assistant that can judge whether a passage has utility in answering the question or not."},
            {'role': 'assistant', 'content': 'Yes, i am the utility judger.'},
            {'role': 'user',
             'content': f"I will provide you with a passage and tell you what necessary information is needed to answer the question. \n Judge whether the passage has utility in answering the question based on the necessary information or not: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passage and the necessary information.'}]

def get_post_direct_judge_point(query, passage,answer, instruct):
    return f"Question: {query}. \n {answer}. \n\n Passage: {passage} \n\n The requirements for judging whether a passage has utility in answering the question are: The passage has utility in answering the question, meaning that the passage not only be relevant to the question, but also be useful in generating a correct, reasonable and perfect answer to the question. \n."+instruct

def get_direct_judge_point(question, instruct, passage, answer):
    messages = get_prefix_direct_judge_point(question)
    messages.append({'role': 'user', 'content': get_post_direct_judge_point(question, passage, answer, instruct)})
    return messages

def get_prefix_direct_judge_list(query, num):
    return [{'role': 'user',
             'content': "You are the utility judger, an intelligent assistant that can select the passages that have utility in answering the question from all given passages."},
            {'role': 'assistant', 'content': 'Yes, i am the utility judger.'},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \n I will also tell you what necessary information is needed to answer the question. \nSelect the passages that have utility in answering the following question based on the necessary information from the {num} passages: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages and the necessary information.'}]
def get_post_direct_judge_list(query, instruct, answer):
    return f"Question: {query}. \n {answer}.\n The requirements for judging whether a passage has utility in answering the question are: The passage has utility in answering the question, meaning that the passage not only be relevant to the question, but also be useful in generating a correct, reasonable and perfect answer to the question. \n"+instruct
def get_direct_judge_list(question, instruct, passages, answer):
    messages = get_prefix_direct_judge_list(question, len(passages))
    rank = 0
    for content in passages:
        rank += 1
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list(question, instruct, answer)})
    return messages


def Get_Acc(pre_model_out_label, model_out_label):
    ac = 0
    for i, j in zip(pre_model_out_label, model_out_label):
        if i==j:
            ac += 1
    return ac/len(pre_model_out_label)

def Get_rouge(pre_answers, answer_generation):
    rouge = Rouge()
    rouge_score = rouge.get_scores(pre_answers, answer_generation)
    return rouge_score[0]["rouge-l"]['f']

def main_list(file_type, llm, tokenizer, instruct, types, number, sampling_params):
    args = get_args(file_type)
    if file_type == "nq":
        args.source = 'data/nq.json'
    elif file_type == "trec":
        args.source = 'data/trec.json'
    else:
        args.source = 'data/webap.json'
    path = 'trec-code/level4_cot/'+file_type+'/listwise/'
    args.outfile = path + file_type + types + str(number) +'.json'
    if not os.path.exists(path):
        os.makedirs(path)
    begin = 0
    if os.path.exists(args.outfile):
        outfile = open(args.outfile, 'r', encoding='utf-8')
        for line in outfile.readlines():
            if line != "":
                begin += 1
        outfile.close()
        outfile = open(args.outfile, 'a', encoding='utf-8')
    else:
        outfile = open(args.outfile, 'w', encoding='utf-8')
    all_data = load_source(args.source)  ## read the file and load it
    num_output = 0
    total_round = 0
    print(len(all_data))
    all_num = 0
    try:
        for sample in tqdm(all_data[begin:len(all_data)], desc="Filename: %s" % args.outfile):
            passages = sample["passages"]   
            question = sample["question"]
            labels = sample["labels"]
            passage_ids = {}
            for index, passage in enumerate(passages):
                passage_ids[passage] = index
            print("question: ", question)
            if "nq" in args.source:
                max_label = 1
            elif "trec" in args.source:
                max_label = 3
            else:
                max_label = 3
            if max(labels) < max_label:
                if len(sample["gold_ctxs"]) == 0:
                    continue
                else:
                    index = len(passages)
                    passages[index-1] = sample["gold_ctxs"][0]
                    labels[index-1] = max_label
            all_num += 1
            temp_passages = []
            model_out_label = []
            pre_model_out_label = [-1]*20
            i_round = 0
            model_out_labels = []
            pres = []
            pre_answers = ""
            recs = []
            ems = []
            has = []
            f1s = []
            ress = []
            pre_answer = ""
            answer_generations = []  
            change_labels = [1 if i==1 else 0 for i in labels]
            while (i_round < 5):
                i_round += 1
                print("-----------------------------------the "+ str(i_round)+" round-------------------------------")
                if temp_passages == []:
                    prompt = generate_answer_prompt(question, passages)
                else:
                    prompt = generate_answer_prompt(question, temp_passages)    
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
                outputs = llm.generate([prompt,], sampling_params)
                answer_generation = outputs[0].outputs[0].text
                print("generating answers: ", answer_generation) 
                answer_generations.append(answer_generation)
                if  number >= 0:
                    if pre_answers!="" and Get_rouge(pre_answers, answer_generation)>=number:
                        break
                pre_answers = answer_generation
                messages = get_direct_judge_list(question, instruct, passages, answer_generation)
                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                outputs = llm.generate([prompt,], sampling_params)
                res = outputs[0].outputs[0].text
                ress.append(res)
                response = clean_response(res)
                if isinstance(response,list):
                    response = [int(x)-1 for x in response]
                    response = remove_duplicate(response)
                else:
                    response = [int(x)-1 for x in response.split()]
                    response = remove_duplicate(response)
                print(res)
                print(response)
                for i in range(20):
                    if i in response:
                        model_out_label.append(1)
                    else:
                        model_out_label.append(0)
                print("preee_out_label: ", pre_model_out_label)
                print("model_out_label: ", model_out_label)
                print("ground_out_labe: ", change_labels)
                temp_passages = []
                for i in range(len(model_out_label)):
                    if model_out_label[i] == 1:
                        temp_passages.append(passages[i])
                # for i in range(len(model_out_label)):
                #     if model_out_label[i] == 0:
                #         if random.random() > 0.5:
                #             temp_passages.append(passages[i])
                pre = get_pre(change_labels, model_out_label)
                rec = get_rec(change_labels, model_out_label)
                pres.append(pre)
                recs.append(rec)
                model_out_labels.append(model_out_label)
                if number == -1:
                    if Get_Acc(pre_model_out_label, model_out_label)==1.0:
                        break
                # for i in range(len(model_out_label)):
                #     if model_out_label[i] == 0:
                #         if random.random() > 0.8:
                #             temp_passages.append(passages[i])
                
                pre_model_out_label = [i for i in model_out_label]
                model_out_label = [] 
                # pre_answer = answer_generation.split(":")[1]
            outfile.write(json.dumps({
                "question": sample["question"],
                "passage": passages,
                "prompts_exmper": messages,
                "output_all": ress,
                "LLM_output_all": model_out_label,
                "ground_truth_label": labels,
                "i_round": i_round,
                "model_out_labels":model_out_labels,
                "answer_generations": answer_generations,
                "pres": pres,
                "recs": recs,
            }) + "\n")
        print("all_num: ", all_num)
    except Exception as e:
        logging.exception(e)

    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        outfile.close()

def main(file_type, llm, tokenizer, instruct, types, number, sampling_params):
    args = get_args(file_type)
    if file_type == "nq":
        args.source = 'data/nq.json'
    elif file_type == "trec":
        args.source = 'data/trec.json'
    else:
        args.source = 'data/webap.json'
    path = 'trec-code/level4_cot/'+file_type+'/pointwise/'
    args.outfile = path + file_type + types + str(number) +'.json'
    if not os.path.exists(path):
        os.makedirs(path)
    begin = 0
    if os.path.exists(args.outfile):
        outfile = open(args.outfile, 'r', encoding='utf-8')
        for line in outfile.readlines():
            if line != "":
                begin += 1
        outfile.close()
        outfile = open(args.outfile, 'a', encoding='utf-8')
    else:
        outfile = open(args.outfile, 'w', encoding='utf-8')

    all_data = load_source(args.source)  ## read the file and load it
    num_output = 0
    total_round = 0
    try:
        for sample in tqdm(all_data[begin:len(all_data)], desc="Filename: %s" % args.outfile):
            passages = sample["passages"]   
            question = sample["question"]
            labels = sample["labels"]
            print("question: ", question)
            if "nq" in args.source:
                max_label = 1
            elif "trec" in args.source:
                max_label = 3
            else:
                max_label = 3
            if max(labels) < max_label:
                if len(sample["gold_ctxs"]) == 0:
                    continue
                else:
                    index = len(passages)
                    passages[index-1] = sample["gold_ctxs"][0]
                    labels[index-1] = max_label
            change_labels = [1 if i==max_label else 0 for i in labels]
            temp_passages = []
            model_out_label = []
            pre_model_out_label = [-1]*20
            i_round = 0
            model_out_labels = []
            pres = []
            pre_answer = ""
            recs = []
            ems = []
            has = []
            answer_generations = []
            f1s = []
            pre_answers = ""
            while (i_round < 5):
                i_round += 1
                print("-----------------------------------the "+ str(i_round)+" round------------------------------") 
                if temp_passages == []:
                    prompt = generate_answer_prompt(question, passages)
                else:
                    prompt = generate_answer_prompt(question, temp_passages)    
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
                outputs = llm.generate([prompt,], sampling_params)
                answer_generation = outputs[0].outputs[0].text
                print("answer_generation: ", answer_generation)
                answer_generations.append(answer_generation)
                if  number >= 0:
                    if pre_answers!="" and Get_rouge(pre_answers, answer_generation)>=number:
                        break
                pre_answers = answer_generation
                prompts_list = []
                for i in range(len(passages)):
                    pair = passages[i]
                    messages = get_direct_judge_point(question, instruct, pair, answer_generation)
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                    prompts_list.append(prompt)
                outputs = llm.generate(prompts_list, sampling_params)
                assert len(outputs) == len(prompts_list)
                ress = []
                for output in outputs:
                    res = output.outputs[0].text
                    ress.append(res)
                    print(res)
                    if "*yes, " in res.lower():
                        model_out_label.append(1)
                        print(1)
                    elif " yes, " in res.lower():
                        model_out_label.append(1)
                        print(1)
                    else:
                        print(0)
                        model_out_label.append(0)
                print("preee_out_label: ", pre_model_out_label)
                print("model_out_label: ", model_out_label)
                print("ground_out_labe: ", change_labels)
                temp_passages = []
                for i in range(len(model_out_label)):
                    if model_out_label[i] == 1:
                        temp_passages.append(passages[i])
                pre = get_pre(change_labels, model_out_label)
                rec = get_rec(change_labels, model_out_label)
                pres.append(pre)
                recs.append(rec)
                model_out_labels.append(model_out_label)
                if number == -1:
                    if Get_Acc(pre_model_out_label, model_out_label)>=1.0:
                        break
                pre_answers = answer_generation
                pre_model_out_label = [i for i in model_out_label]
                model_out_label = []
                pre_answer = answer_generation
                
            outfile.write(json.dumps({
                "question": sample["question"],
                "passage": passages,
                "prompts_exmper": prompts_list[0],
                "output_all": ress,
                "LLM_output_all": model_out_label,
                "ground_truth_label": labels,
                "i_round": i_round,
                "model_out_labels":model_out_labels,
                "answer_generations": answer_generations,
                "pres": pres,
                "recs": recs,
            }) + "\n")
    except Exception as e:
        logging.exception(e)

    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        outfile.close()

    
if __name__ == '__main__':
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    tokenizer = transformers.AutoTokenizer.from_pretrained("models/Mistral_7B_Instruct_v0-2/")
    llm = LLM(model="models/Mistral_7B_Instruct_v0-2/")
    
    instruct = """
    Directly output whether the passage has utility in answering the question according to the information that is needed to answer the question or not. If the passage has utility, output 'My judgment: Yes, the passage has utility in answering the question.'; otherwise, output 'My judgment: No, the passage has no utility in answering the question.'.
    """
    main("trec", llm, tokenizer, instruct, "-new-iter-passages-pointfinal14-", -1, sampling_params)
    main("webap", llm, tokenizer, instruct, "-new-iter-passages-pointfinal14-", -1, sampling_params)

    
    instruct = """
    Directly output the passages you selected that have utility in answering the question according to the information that is needed to answer the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
    """
    main_list("trec", llm, tokenizer, instruct, "-note-iter-passages-listfianl13-", -1, sampling_params)
    main_list("webap", llm, tokenizer, instruct, "-note-iter-passages-listfinal13-", -1, sampling_params)
   
    
    
    
    
    
    
  





