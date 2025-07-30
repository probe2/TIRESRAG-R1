import argparse
from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams
import os
import re
import threading
import json 
from transformers import AutoTokenizer
def get_result(s):
    match = re.search(r'(\{\s*"sufficient context to the given answer"\s*:\s*[^}]+\})', s.lower(), re.DOTALL)
    if match:
        json_str = match.group(1)
        json_str = re.sub(r'\s+', ' ', json_str)
        tempx = json.loads(json_str)
    else:
        global m 
        m += 1
        return 0
    return tempx["sufficient context to the given answer"]
def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def extract_information_and_others(text):
    
    info_pattern = r"<information>(.*?)</information>"
    info_contents = re.findall(info_pattern, text, re.DOTALL)

    other_pattern = r"(?:</information>)(.*?)(?=<information>|$)"
    other_contents = re.findall(other_pattern, text, re.DOTALL)

    start_pattern = r"^(.*?)(?=<information>|$)"
    start_content = re.findall(start_pattern, text, re.DOTALL)
    if start_content and start_content[0].strip():
        other_contents = [start_content[0]] + other_contents

    info_contents = [c.strip() for c in info_contents]
    other_contents = [c.strip() for c in other_contents if c.strip()]
    return "\n".join(info_contents), "\n".join(other_contents)

def extract_question(question):
    match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', question, re.DOTALL)
    if match:
        content = match.group(1)
    return content
app = Flask(__name__)
sys_prompt  =  """
You are an expert LLM evaluator that excels at evaluating a QUESTION, ANSWER and REFERENCES.
Consider the following criteria:
Sufficient Context To The Given Answer: 1 IF the CONTEXT is sufficient to infer the ANSWER to the question and 0
IF the CONTEXT cannot be used to infer the ANSWER to the question. Make the sufficiency judgment based solely on the context, without relying on your memory to determine whether the question can be answered from the context.
First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION. Include questions about any mathematical calculations or arithmetic that would be required.
Next, answer each of the questions. Please note that you may answer these questions only on the basis of the given context; do not use your own outside knowledge. Make sure to work step by step through any required mathematical calculations or arithmetic.
Finally, use these answers to evaluate the criteria.
EVALUATION (JSON)
EXAMPLE:
### QUESTION
In which year did the publisher of Roald Dahl’s Guide to Railway Safety cease to exist?
### ANSWER
2001
### References
Roald Dahl’s Guide to Railway Safety was published in 1991 by the British Railways Board.
The British Railways Board had asked Roald Dahl to write the text of the booklet, and
Quentin Blake to illustrate it, to help young people enjoy using the railways safely. The
British Railways Board (BRB) was a nationalised industry in the United Kingdom that
operated from 1963 to 2001. Until 1997 it was responsible for most railway services in Great
Britain, trading under the brand name British Railways and, from 1965, British Rail. It
did not operate railways in Northern Ireland, where railways were the responsibility of the
Government of Northern Ireland.
### EXPLANATION
The context mentions that Roald Dahl’s Guide to Railway Safety was published by the
British Railways Board. It also states that the British Railways Board operated from 1963 to
2001, meaning the year it ceased to exist was 2001. Therefore, the context does provide a
precise answer to the question.
### JSON
{{"Sufficient Context To The Given Answer": 1}}
Remember the instructions: You are an expert LLM evaluator that excels at evaluating a
QUESTION, ANSWER and REFERENCES. Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the ANSWER to the question and 0
IF the CONTEXT cannot be used to infer the ANSWER to the question. Make the sufficiency judgment based solely on the context, without relying on your memory to determine whether the question can be answered from the context.
First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION. Include questions about any mathematical calculations or arithmetic that would be required.
Next, answer each of the questions. Please note that you may answer these questions only on the basis of the given context; do not use your own outside knowledge. Make sure to work step by step through any required mathematical calculations or arithmetic.
Finally, use these answers to evaluate the criteria.
Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ###
EVALUATION (JSON)
"""

thinking_sys_prompt = """
You are an expert reasoning evaluator for Retrieval-Augmented Generation (RAG) tasks.
Your goal is to judge the reasoning quality of the model's thinking process based on the retrieved context and question.
You will assign a reward score between 0 and 1. This score reflects only the quality of the reasoning process, not whether the final answer is correct.

Evaluation Criteria:
1. Logical Soundness – Is the reasoning coherent and structured?
2. Contextual Alignment – Does it use retrieved evidence correctly?
3. Error Awareness – Does it avoid unsupported assumptions?
4. Clarity and Precision – Is it concise, relevant, and non-redundant?

Scoring:
0.0: Completely flawed reasoning
1.0: Perfect reasoning
Intermediate (e.g., 0.3, 0.7) are allowed.

Important:
- Judge only the thinking process, not the answer.
- Reward accurate, grounded, and structured reasoning.

Your Output:
A single float-type score from {{0.0, 0.1, 0.2, ..., 1.0}}.
No explanation. Only the score.
"""
@app.route("/generate", methods=["POST"])
def generate():
    data1 = request.json
    search_data_len = len(data1['data'])
    data = []
    for i in data1['data']:
        data.append(i)
    # for i in data1['normal_data']:
    #     data.append(i['normal'])
    # prompt = data.get("prompt")
    max_tokens =  4048
    temperature =  0.6
    top_p = 0.95
    top_k = 20
    # if not prompt:
    #     return jsonify({"error": "Missing prompt"}), 400
    global m 
    m = 0 
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        top_k=top_k,
        stop=["<|im_end|>", "</s>", "<|assistant|>"],
    )
    process_data = []
    for i in data:
        user_prompt = '''
        ### QUESTION
        {question}
        ### ANSWER
        {answer}    
        ### REFERENCES
        {context}'''.format(question=i['question'], answer=i['answer'], context=i['context'])
        messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        process_data.append(text)
    process_thinking_data = []
    for i in data:
        information, thinking_process = extract_information_and_others(i['context'])
        user_prompt = '''
        INPUT:
        ### CONTEXT
        {information}
        ### QUESTION
        {question}
        ### MODEL THINKING PROCESS
        {thinking_process}    
        '''.format(question=i['question'], thinking_process=thinking_process, information=information)
        messages = [
        {"role": "system", "content": thinking_sys_prompt},
        {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        process_thinking_data.append(text)
    with lock:
        outputs = llm.generate(process_data, sampling_params)
        thinking_outputs = llm.generate(process_thinking_data, sampling_params)
    res = []
    for i in range(len(outputs)):
        res.append(get_result(outputs[i].outputs[0].text))
    res_thinking = []
    for i in range(len(thinking_outputs)):
        try:
            res_thinking.append(float(remove_think_tags(thinking_outputs[i].outputs[0].text.strip()).strip()))
        except:
            res_thinking.append(0)
    return jsonify({
        "results": res,
        "thinking_results": res_thinking
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    lock = threading.Lock()

    print("Loading vLLM model...")
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", tensor_parallel_size = 1, trust_remote_code=True, gpu_memory_utilization = 0.45)  # dtype="auto" 自动选择精度
    print("Model loaded.")
    sufficient_dic  = {'sufficient': 1, 'insufficient': 0}
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port, debug=False)