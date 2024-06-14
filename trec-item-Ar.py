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

def extract_substrings(input_string):
    pattern = re.compile(r'\[\d+\]')
    substrings = pattern.findall(input_string)
    extracted_string = ''.join(substrings)
    return extracted_string

def clean_response(sp: str, max_len):
    response = extract_substrings(sp.lower())
    if len(response) == 0:
        response = extract_numbers(sp.lower())
        new_response = []
        for res in response:
            if int(res)> max_len or int(res)<=0:
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

def generate_answer_prompt_passages(question, passages):
    pas = '\n'.join(passages)
    return [{'role': 'user', 'content': f"You are a faithful question and answer assistant. Answer the question based on the given information with one or few words without the source."}, 
            {'role': 'assistant', 'content': 'Yes, i am the faithful question and answer assistant.'}, 
            {'role': 'user', 'content': f"Given the information: \n{pas}\n Answer the following question based on the given information with one or few words without the source.\n Question: {question}\n\n Answer:"},]

def get_prefix_prompt_relevance(query, num, answer):
    return [{'role': 'user',
            'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevance to the query."},
            {'role': 'assistant',
            'content': "Yes, i am RankGPT."},
            {'role': 'user',
            'content': f"I will provide you with {num} passages, each indicated by number identifier [].  I will also give you a reference answer to the query. \nRank the passages based on their relevance to the query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages and the reference answer.'}]

def get_post_prompt_relevance(query, num, answer):
    return f"Query: {query}. \n\n Reference answer: {answer}\n\n Rank the {num} passages above based on their relevance to the query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [] > [] > ..., e.g., [i] > [j] > [k] > ... Only response the ranking results, do not say any word or explain."


def get_prompt_utility(query, passages, answer):
    num = len(passages)
    messages = get_prefix_prompt_utility(query, num, answer)
    rank = 0
    for passage in passages:
        rank += 1
        content = passage
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt_utility(query, num, answer)})
    return messages

def get_prefix_prompt_utility(query, num, answer):
    return [{'role': 'user',
            'content': "You are RankGPT, an intelligent assistant that can rank passages based on their utility in generating the given reference answer to the question."},
            {'role': 'assistant',
            'content': "Yes, i am RankGPT."},
            {'role': 'user',
            'content': f"I will provide you with {num} passages, each indicated by number identifier [].  I will also give you a reference answer to the question. \nRank the passages based on their utility in generating the reference answer to the question: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages and the reference answer.'}]

def get_post_prompt_utility(query, num, answer):
    return f"Question: {query}. \n\n Reference answer: {answer}\n\n Rank the {num} passages above based on their utility in generating the reference answer to the question. The passages should be listed in utility descending order using identifiers.  The passages that have utility generating the reference answer to the question should be listed first. The output format should be [] > [] > [] > ..., e.g., [i] > [j] > [k] > ... Only response the ranking results, do not say any word or explain."





def get_prompt_relevance(query, passages, answer):
    num = len(passages)
    messages = get_prefix_prompt_relevance(query, num, answer)
    rank = 0
    for passage in passages:
        rank += 1
        content = passage
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt_relevance(query, num, answer)})
    return messages

def get_relevance_passages(relevance_generation, passages, labels):
    response = clean_response(relevance_generation, len(passages))
    if isinstance(response,list):
        response = [int(x)-1 for x in response]
        response = remove_duplicate(response)
    else:
        response = [int(x)-1 for x in response.split()]
        response = remove_duplicate(response)
    print(response)
    new_passages = []
    new_labels, all_index = [], set()
    for i in response:
        if i>=len(passages) or i <0:
            continue
        else:
            new_passages.append(passages[i])
            new_labels.append(labels[i])
            all_index.add(i)
    for i in range(20):
        if i not in all_index:
            new_labels.append(labels[i])
            new_passages.append(passages[i])
    return new_passages, new_labels




def get_utility_passages(relevance_generation, passages, labels):
    response = clean_response(relevance_generation, len(passages))
    if isinstance(response,list):
        response = [int(x)-1 for x in response]
        response = remove_duplicate(response)
    else:
        response = [int(x)-1 for x in response.split()]
        response = remove_duplicate(response)
    print(response)
    new_passages = []
    new_labels, all_index = [], set()
    for i in response:
        if i>=len(passages) or i <0:
            continue
        else:
            new_passages.append(passages[i])
            new_labels.append(labels[i])
            all_index.add(i)
    for i in range(20):
        if i not in all_index:
            new_labels.append(labels[i])
            # new_passages.append(passages[i])
    return new_passages, new_labels
    
def get_args(file_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/dl.json')
    parser.add_argument('--type', type=str, default="") 
    parser.add_argument('--outfile', type=str, default='')
    args = parser.parse_args()
    return args



def get_direct_judge_point(question, instruct, passage, answer):
    messages = get_prefix_direct_judge_point(question)
    messages.append({'role': 'user', 'content': get_post_direct_judge_point(question, passage, answer, instruct)})
    return messages

def get_prefix_direct_judge_list(query, num):
    return [{'role': 'user',
             'content': "You are the utility judger, an intelligent assistant that can select the passages that have utility in answering the question."},
            {'role': 'assistant', 'content': 'Yes, i am the utility judger.'},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \n I will also provide you with a reference answer to the question. \nSelect the passages that have utility in generating the reference answer to the following question from the {num} passages: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages and the reference answer.'}]
def get_post_direct_judge_list(query, instruct, answer):
    return f"Question: {query}. \n Reference answer: {answer}. \n\n The requirements for judging whether a passage has utility in answering the question are: The passage has utility in answering the question, meaning that the passage not only be relevant to the question, but also be useful in generating a correct, reasonable and perfect answer to the question. \n"+instruct


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

def main_list(file_type, llm, tokenizer, instruct, types, number, sampling_params, answer_number, nw, kk):
    args = get_args(file_type)
    if file_type == "nq":
        args.source = 'data/nq.json'
    elif file_type == "trec":
        args.source = 'data/trec.json'
    else:
        args.source = 'data/webap.json'
    path = 'trec-code/level4_utility_ranking/'+file_type+'/listwise/'
    args.outfile = path + file_type + types + str(number) +'final-'+str(answer_number)+str(kk)+'.json'
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
                passages = sample["passages"][:10]   
                question = sample["question"]
                labels = sample["labels"][:10]
            elif "trec" in args.source:
                max_label = 3
            else:
                max_label = 3
            if "nq" not in args.source:
                if max(labels) < max_label:
                    if len(sample["gold_ctxs"]) == 0:
                        continue
                    else:
                        index = len(passages)
                        passages[index-1] = sample["gold_ctxs"][0]
                        labels[index-1] = max_label
            copy_passages = passages.copy()
            all_num += 1
            temp_passages, ress, model_out_label, utility_labels, relevance_labels = [], [], [], [], []
            pre_model_out_label = [-1]*20
            i_round = 0
            model_out_labels = []
            pre_answers = ""
            answer_generations = []
            total_labels = []
            dirct = {}
            for i in range(len(copy_passages)):
                dirct[copy_passages[i]] = i 
            while (i_round < 5):
                i_round += 1
                print("-----------------------------------the "+ str(i_round)+" round-------------------------------")     
                if temp_passages == []:
                    prompt = generate_answer_prompt_passages(question, passages)
                else:
                    prompt = generate_answer_prompt_passages(question, temp_passages)  
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
                outputs = llm.generate([prompt,], sampling_params)
                answer_generation = outputs[0].outputs[0].text
                print("generating answers: ", answer_generation)              
                answer_generations.append(answer_generation)
                pre_answers = answer_generation
                ground_truth_ids = []
                for index, label in enumerate(labels):
                    if label == max_label:
                        if index >= len(passages):
                            continue
                        ground_truth_ids.append(dirct[passages[index]])
                messages_relevance = get_prompt_utility(question, passages, answer_generation)
                prompt = tokenizer.apply_chat_template(messages_relevance, tokenize=False)
                outputs = llm.generate([prompt,], sampling_params)
                relevance_generation = outputs[0].outputs[0].text
                print(relevance_generation)
                passages, labels = get_relevance_passages(relevance_generation, passages, labels)  
                utility_labels.append(labels)
                response = []
                for i in range(kk):
                    response.append(i)
                model_out_ids = []
                temp_passages = []
                model_out_label = []
                passages_id = {}
                for i in range(20):
                    if i in response and i < len(passages):
                        temp_passages.append(passages[i])
                        passages_id[passages[i]] = dirct[passages[i]]
                        model_out_ids.append(dirct[passages[i]])
                        model_out_label.append(1)
                    else:
                        model_out_label.append(0)
                # passages_id = sorted(passages_id.items(),key = lambda x:x[1],reverse = False)
                # for passage, id in passages_id:
                #     temp_passages.append(passage)
                print("preee_out_label: ", pre_model_out_label)
                print("model_out_label: ", model_out_ids)
                print("ground_out_labe: ", ground_truth_ids)
                model_out_labels.append(model_out_label)
                if answer_number == -1:
                    if set(pre_model_out_label) == set(model_out_ids):
                        break
                pre_model_out_label = [i for i in model_out_ids]
            outfile.write(json.dumps({
                "question": sample["question"],
                "passage": passages,
                "prompts_exmper": prompt,
                "output_all": ress,
                "LLM_output_all": model_out_label,
                "ground_truth_label": labels,
                "i_round": i_round,
                "relevance_labels": relevance_labels,
                "model_out_labels":model_out_labels,
                "answer_generations": answer_generations,
                "utility_labels": utility_labels
            }) + "\n")
        print("all_num: ", all_num)
    except Exception as e:
        logging.exception(e)

    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        outfile.close()



    
if __name__ == '__main__':
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    tokenizer = transformers.AutoTokenizer.from_pretrained("/home/gomall/models/Mistral_7B_Instruct_v0-2/")
    llm = LLM(model="/home/gomall/models/Mistral_7B_Instruct_v0-2/")

    instruct = """
    Directly output the passages you selected that have utility in generating the reference answer to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
    """
  
    # main_list("webap", llm, tokenizer, instruct, "-iter-passages-list-new-no-def-final-", 0, sampling_params, -1, 0, 5)
    # main_list("trec", llm, tokenizer, instruct, "-iter-passages-list-new-no-def-final-", 0, sampling_params, -1, 0, 1)
    # main_list("trec", llm, tokenizer, instruct, "-iter-passages-list-new-no-def-final-", 0, sampling_params, -1, 0, 3)
    main_list("trec", llm, tokenizer, instruct, "-iter-passages-list-new-no-def-final-", 0, sampling_params, -1, 0, 10)
#     main_list("trec", llm, tokenizer, instruct, "-iter-passages-list-new-no-defQuestionnew-", 0, sampling_params, -1, 0, 3)
#     main_list("trec", llm, tokenizer, instruct, "-iter-passages-list-new-no-defQuestionnew-", 0, sampling_params, -1, 0, 1)
#     main_list("trec", llm, tokenizer, instruct, "-iter-passages-list-new-no-defQuestionnew-", 0, sampling_params, -1, 0, 10)
    
    # main_list("webap", llm, tokenizer, instruct, "-iter-passages-list-new-no-defQuestionnew-", 0, sampling_params, -1, 0, 5)
    # main_list("webap", llm, tokenizer, instruct, "-iter-passages-list-new-no-defQuestionnew-", 0, sampling_params, -1, 0, 3)
    # main_list("webap", llm, tokenizer, instruct, "-iter-passages-list-new-no-defQuestionnew-", 0, sampling_params, -1, 0, 1)
    # main_list("webap", llm, tokenizer, instruct, "-iter-passages-list-new-no-defQuestionnew-", 0, sampling_params, -1, 0, 10)
   
    
  





