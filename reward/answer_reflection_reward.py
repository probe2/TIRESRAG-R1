import argparse
import re
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import datasets
import random
from openrlhf.utils.logging_utils import init_logger
from transformers import AutoTokenizer
# from symeval import EvaluatorMathBatch
import sys
import ujson as json
import re
import string
from collections import Counter
import pickle
import copy
import requests
import math

import numpy as np
np.set_printoptions(suppress=True, precision=4)  

logger = init_logger(__name__)


import torch
import math

def compute_consistency_penalty(answer_reward: None,
                                 thinking_reward: None,
                                 sufficient_reward: None,
                                 global_step: int,
                                 total_training_steps: int,
                                 lambda_penalty: float = 0.1,
                                 group_size: int = 1):
    """
    Compute delta_a, delta_t, delta_s and return penalty when their product is negative.
    """
    # Dynamic weight
    tau = 10
    progress = (global_step - total_training_steps * 0.9) / tau
    w = 1 / (1 + math.exp(progress))  # scalar
    answer_reward = np.array(answer_reward).reshape(-1, group_size)
    thinking_reward = np.array(thinking_reward).reshape(-1, group_size)
    sufficient_reward = np.array(sufficient_reward).reshape(-1, group_size)
    # Normalize std from sufficient_reward
    std_a = np.std(answer_reward, axis=1, keepdims=True) + 1e-8
    std_t = np.std(thinking_reward, axis=1, keepdims=True) + 1e-8
    std_s = np.std(sufficient_reward, axis=1, keepdims=True) + 1e-8

    # Compute deltas
    delta_a = (answer_reward - answer_reward.mean(axis=1, keepdims=True)) / std_a
    delta_t = (w * 0.6 * thinking_reward - (w * 0.6 * thinking_reward).mean(axis=1, keepdims=True)) / std_t
    delta_s = (w * 0.3 * sufficient_reward - (w * 0.3 * sufficient_reward).mean(axis=1, keepdims=True)) / std_s

    # Compute product of three deltas
    product = delta_a * delta_t * delta_s

    # Apply penalty when product < 0
    penalty = -lambda_penalty * product * (product < 0).astype(float)
    return  penalty

def load_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        if data_path.endswith('.jsonl'):
            data = []
            for line in f:
                data.append(json.loads(line.strip()))
        else:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    return data
def difficulty_weight(rho_q, *, A=0.4, B=1.5, rho0=0.75, k=10.0):
    return A + (B - A) / (1.0 + np.exp(k * (rho_q - rho0)))

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    if s == "True": return "yes"
    elif s == "False": return "no"
    else: return s

def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )

def cover_exact_match_score_1(prediction, ground_truth):

    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")
    if all(ground in pre_list for ground in ground_list):
        return 1
    else:
        return 0

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    if (normalized_prediction in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC
    if (normalized_ground_truth in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
        
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def is_valid_sequence(text):
    content = text

    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    state = "start"
    last_tag = None  

    for part in parts:
        if not part.strip():
            continue

        if re.match(r"</?(?:think|search|information|answer)>", part):
            last_tag = part  

            if part == "<think>" and state in ["start", "information", "after_answer"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state in ["after_think", "information", "after_answer"]:
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "after_answer"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                continue
            elif state in ["start", "after_think", "after_search", "information", "after_answer"]:
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    if last_tag != "</answer>":
        return False, f"Final tag must be </answer>, but got {last_tag}"

    return True, "Valid sequence format"

def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def extract_answer_math(s):
    return s.split("<answer>")[-1].split("</answer>")[0].strip()

def extract_all_answers(s):
    return [ans.strip() for ans in re.findall(r"<answer>(.*?)</answer>", s, re.DOTALL)]
format_punishment_dic = {True: 1, False: 0}
class MathRuleProxy:
    def __init__(self, args):
        eval_dataset = load_dataset(args.data_path)
        self.eval_data_dict = self.get_answer_dict(eval_dataset)
        print(len(self.eval_data_dict))
        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain, trust_remote_code=True, use_fast=True)
        self.log_file = args.log_file
        
        self.avg_length_dict = []
        self.cnt = 0
        self.avg_len = 5000
        
        self.key_words = [
            "wait", "double check", "what", "how", "why",
            "alternatively", "think", "rethink", "?",
            "change", "try", "check",
        ]
        self.args = args
    def get_answer_dict(self, eval_dataset):
        eval_data_dict = {}
        for item in eval_dataset:
            eval_data_dict[normalize_text(item["question"])] = item["answer"]
        return eval_data_dict

    def get_qa(self, query):
        if self.args.template_type == 'qwen_base':
            remove_prefix = " ".join(query.split("Question:")[1:])
            question = remove_prefix.split("\n<think>")[0]
            solution = " ".join(query.split("\n<think>")[1:]).split("<|endoftext|>")[0].strip()
        elif self.args.template_type == 'qwen_chat':
                question = query.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
                solution = query.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
        return question, solution

    def get_query_answer(self, query):
        query = normalize_text(query)
        return self.eval_data_dict[query]

    def get_query_pred(self, query):
        return extract_answer_math(query), extract_all_answers(query)

    def get_reward(self, queries, group_size, exit_flags =  None, total_training_steps = None, global_step = None ):
        preds = [] 
        answers = []  
        questions = [] 
        solutions = []  
        finished_lst = []  
        all_preds = [] 

        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
            question, solution = self.get_qa(queries[i])
            x1, x2 = self.get_query_pred(solution)
            preds.append(x1)
            all_preds.append(x2)
            answers.append(self.get_query_answer(question))
            questions.append(question)
            solutions.append(solution)

        scores = []
        for t in range(len(queries)):
            f1_score_now, _, _ = f1_score(preds[t], answers[t])
            scores.append(float(f1_score_now))
        premilinary_scores = copy.deepcopy(scores)
        format_punishments = []
        cover_scores = [] 
        for t in range(len(queries)):
            cover_scores.append(cover_exact_match_score_1(preds[t], answers[t]))

        previous_right_final_incorrect_results_wait_enough_information = [0 for _ in range(len(queries))]
        previous_right_final_incorrect_results_wait_not_enough_information = [0 for _ in range(len(queries))]
        previous_incorrect_final_right_results_wait_enough_information = [0 for _ in range(len(queries))]
        previous_incorrect_final_right_results_wait_not_enough_information = [0 for _ in range(len(queries))]
        previous_right_final_incorrect = []
        previous_incorrect_final_right = []
        reflect_results = []
        no_change_results = []
        reflect_no_change_results = []
        for i, query in enumerate(queries):
            temp_cem  = []
            temp = []
            for j in range(len(all_preds[i])):
                f1_score_now, _, _ = f1_score(all_preds[i][j], answers[i])
                cem_score_now = cover_exact_match_score_1(all_preds[i][j], answers[i])
                temp_cem.append(cem_score_now)
                temp.append(f1_score_now)
            if(len(temp) > 1):
                if temp[0] > 0.5 or temp_cem[0] == 1:
                    if temp[-1] > 0.5 or temp_cem[-1] == 1:
                        if temp[-1] >= temp[0]:
                            reflect_results.append(0)
                            reflect_no_change_results.append(1)
                            previous_right_final_incorrect.append(0)
                            previous_incorrect_final_right.append(0)
                        else:
                            reflect_results.append(-1)
                            reflect_no_change_results.append(0)
                            previous_right_final_incorrect.append(1)
                            previous_incorrect_final_right.append(0)
                    else:
                        reflect_results.append(-1)
                        reflect_no_change_results.append(0)
                        previous_right_final_incorrect.append(1)
                        previous_incorrect_final_right.append(0)
                else:
                    if temp[-1] > 0.5 or temp_cem[-1] == 1:
                        reflect_results.append(0)
                        reflect_no_change_results.append(0)
                        previous_incorrect_final_right.append(1)
                        previous_right_final_incorrect.append(0)
                    else:
                        reflect_results.append(-1)
                        reflect_no_change_results.append(1)
                        previous_incorrect_final_right.append(0)
                        previous_right_final_incorrect.append(0)
                no_change_results.append(0)
            elif(len(temp) == 1):
                if temp[-1] > 0.5 or temp_cem[-1] == 1: 
                    reflect_results.append(0)
                else:
                    reflect_results.append(-1)
                no_change_results.append(1)
                reflect_no_change_results.append(0)
                previous_right_final_incorrect.append(0)
                previous_incorrect_final_right.append(0)
            else:
                reflect_results.append(0)
                no_change_results.append(0)
                reflect_no_change_results.append(0)
                previous_right_final_incorrect.append(0)
                previous_incorrect_final_right.append(0)

        for i, query in enumerate(queries):
            format_punishment, error_info = is_valid_sequence(solutions[i]) 
            format_punishments.append(format_punishment_dic[format_punishment])
        correct_index = [i for i, _ in enumerate(cover_scores) if cover_scores[i] == 1]
        wrong_index = [i for i, _ in enumerate(cover_scores) if cover_scores[i] == 0]
        format_score = [0.2 if format_punishments[i] else 0 for i, _ in enumerate(format_punishments)]
        sufficient_queries = [] 
        for index, i in enumerate(questions):
            temp = {}
            temp['question'] = i
            temp['answer'] = answers[index]
            temp['context'] = re.sub(r'<answer>.*?</answer>', '', solutions[index], flags=re.DOTALL)
            sufficient_queries.append(temp)

        response = requests.post(sufficient_url, json={"data": sufficient_queries}) 
        if response.status_code == 200:
            result = response.json()
            sufficient_res = result['results']
            thinking_res = result['thinking_results']
        chongfen_qingkuang = {'correct_sufficient':0, 'correct_insufficient':0, 'incorrect_sufficient':0, 'incorrect_insufficient':0}

        
        sufficient_res_group = np.array(sufficient_res).reshape(-1, group_size)
        sufficient_premilinary_match = np.zeros(sufficient_res_group.shape)
        x_premilinary_scores = np.array(premilinary_scores).reshape(-1, group_size)
        for i in range(len(sufficient_res_group)):
            for j in range(group_size):
                if(sufficient_res_group[i][j] == 1):
                    if(x_premilinary_scores[i][j] > 0.6 or (i*group_size + j) in correct_index):
                        sufficient_premilinary_match[i][j] = 1
                else:
                    if(x_premilinary_scores[i][j] < 0.6 and (i*group_size + j) in wrong_index):
                        sufficient_premilinary_match[i][j] = 1
        group_level_ratio = np.sum(sufficient_res_group, axis=1) / group_size
        resufficient_res_group = np.repeat(group_level_ratio[:, np.newaxis], group_size, axis=1)
        difficulty = difficulty_weight(resufficient_res_group).reshape(-1)
        penalty = compute_consistency_penalty(scores, thinking_res, sufficient_res, global_step, total_training_steps, group_size = group_size)
        penalty = penalty.reshape(-1)
        for index, i in enumerate(scores):
            scores[index] += 1 / (1 + math.exp((global_step - total_training_steps * 0.9) / 10))  * (0.6 *  thinking_res[index] + 0.3 * sufficient_res[index] + 0.3 * reflect_results[index])
            
        for index, diff in enumerate(difficulty):
            is_correct = index in correct_index
            is_sufficient = sufficient_res[index] == 1

            if is_correct and is_sufficient:
                chongfen_qingkuang['correct_sufficient'] += 1

            elif is_correct and not is_sufficient:
                chongfen_qingkuang['correct_insufficient'] += 1

            elif not is_correct and is_sufficient:
                chongfen_qingkuang['incorrect_sufficient'] += 1

            else:
                chongfen_qingkuang['incorrect_insufficient'] += 1
        x_flags = np.array(exit_flags).reshape(-1,group_size)
        x_scores = np.array(scores).reshape(-1,group_size)
        exit_lose_reward = []
        for i in range(len(x_flags)):
            mean_score = np.min(x_scores[i])
            for j in range(group_size):
                if(x_flags[i][j] == 1): 
                    if(x_scores[i][j] <= mean_score): 
                        exit_lose_reward.append(1)
                    else:
                        exit_lose_reward.append(0)
                else:
                    exit_lose_reward.append(0)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for q, a, s, f_f in zip(questions, solutions, scores, finished_lst):
                    record = {
                        "question": q,
                        "solution": a,
                        "score": s,
                        "finished": f_f,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {"总的分数":scores,"纯answer分数":premilinary_scores,"纯format分数":format_score,"前面对了后面错了有足够信息":previous_right_final_incorrect_results_wait_enough_information, \
                "前面错了后面对了有足够信息":previous_incorrect_final_right_results_wait_enough_information, "前面对了后面错了没有足够信息":previous_right_final_incorrect_results_wait_not_enough_information, 
                "前面错了后面对了没有足够信息":previous_incorrect_final_right_results_wait_not_enough_information, "没有反思的":no_change_results,"没有改变的reflect": reflect_no_change_results,
                "反思前错误的，后面正确的":previous_incorrect_final_right, "反思前正确的，后面错误的":previous_right_final_incorrect,
                "exit_lose_reward": exit_lose_reward, "sufficient_premilinary_match": sufficient_premilinary_match.reshape(-1).tolist(),
                "thinking_reward": thinking_res,
                "difficulty": difficulty.tolist(),
                "penalty": penalty.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--port", type=int, default=5001, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--log_file", type=str, default=None, help="Path to JSONL log file")
    parser.add_argument("--template_type", type=str, default='qwen_base', help="template type")
    args = parser.parse_args()
    sufficient_url = 'http://127.0.0.1:5006/generate'
    # server
    reward_model = MathRuleProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        group_size = data.get("group_size")
        exit_flags = data.get("exit_flags")
        total_training_steps = data.get("total_training_steps")
        global_step = data.get("global_step")
        rewards = reward_model.get_reward(queries, group_size, exit_flags, total_training_steps = total_training_steps, global_step = global_step)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


