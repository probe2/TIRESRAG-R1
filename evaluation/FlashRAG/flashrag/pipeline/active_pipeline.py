import re
from tqdm import tqdm
from typing import List, Tuple
import math
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from flashrag.utils import get_retriever, get_generator, selfask_pred_parse, ircot_pred_parse
from flashrag.pipeline import BasicPipeline
from flashrag.dataset.utils import get_batch_dataset, merge_batch_dataset
from flashrag.prompt import PromptTemplate
from flashrag.dataset.dataset import Dataset, Item
import copy
import re
import json
import os
import re 

def clean_text(text):
    # print(text)
    doc_pattern = r'<documents>.*?</documents>'
    if re.search(doc_pattern, text, flags=re.DOTALL):
        m = re.sub(doc_pattern, '', text, flags=re.DOTALL)
    else:
        m = text
    # print('打印',m)
    info_pattern = r'<information>.*?</information>'
    if re.search(info_pattern, m, flags=re.DOTALL):
        m = re.sub(info_pattern, '', m, flags=re.DOTALL)
    info_pattern = r'<result>.*?</result>'
    if re.search(info_pattern, m, flags=re.DOTALL):
        m = re.sub(info_pattern, '', m, flags=re.DOTALL)
    # print('打印1',m)
    pattern = r'<\|begin_of_documents\|>.*?<\|end_of_documents\|>'
    if re.search(pattern, m, flags=re.DOTALL):
        return re.sub(pattern, '', m, flags=re.DOTALL)

    return m 

class ItSearchPipeline(BasicPipeline):
    def __init__(self, config, retriever=None, generator=None):
        super().__init__(config)
        if retriever is None:
            retriever = get_retriever(config)
        if generator is None:
            generator = get_generator(config)
        self.retriever = retriever
        self.generator = generator
        if(self.config.get("prompt_template_type", "default") == "qwen-qwq"): 
            from flashrag.prompt.prompt_template1_qwen import single_demons_high_reasoning_and_high_retrieval_effort, one_demons_high_reasoning_high_retrieval_prompt, single_demons_high_reasoning_and_low_retrieval_effort, \
single_demons_low_reasoning_and_high_retrieval_effort,single_demons_low_reasoning_and_low_retrieval_effort, single_demons_normal_prompt
            from flashrag.prompt.demons import example

            self.single_demons_prompts  = {
                    "high_reasoning_and_high_retrieval_effort": single_demons_high_reasoning_and_high_retrieval_effort,
                    "high_reasoning_and_low_retrieval_effort": single_demons_high_reasoning_and_low_retrieval_effort,
                    "low_reasoning_and_high_retrieval_effort": single_demons_low_reasoning_and_high_retrieval_effort,
                    "low_reasoning_and_low_retrieval_effort": single_demons_low_reasoning_and_low_retrieval_effort,
                    "normal": single_demons_normal_prompt,

            }
            self.demons = {'normal': example, 'high_reasoning_and_high_retrieval_effort': example, 'high_reasoning_and_low_retrieval_effort': example, 'low_reasoning_and_high_retrieval_effort': example, 'low_reasoning_and_low_retrieval_effort': example}
        elif(self.config.get("prompt_template_type", "default") == "pure_llama"): 
            from flashrag.prompt.prompt_template1_llama import single_demons_high_reasoning_and_high_retrieval_effort, one_demons_high_reasoning_high_retrieval_prompt, single_demons_high_reasoning_and_low_retrieval_effort, \
single_demons_low_reasoning_and_high_retrieval_effort,single_demons_low_reasoning_and_low_retrieval_effort, single_demons_normal_prompt
            from flashrag.prompt.demons import example

            self.single_demons_prompts  = {
                    "high_reasoning_and_high_retrieval_effort": single_demons_high_reasoning_and_high_retrieval_effort,
                    "high_reasoning_and_low_retrieval_effort": single_demons_high_reasoning_and_low_retrieval_effort,
                    "low_reasoning_and_high_retrieval_effort": single_demons_low_reasoning_and_high_retrieval_effort,
                    "low_reasoning_and_low_retrieval_effort": single_demons_low_reasoning_and_low_retrieval_effort,
                    "normal": single_demons_normal_prompt,

            }
            self.demons = {'normal': example }
        elif(self.config.get("prompt_template_type") == "none"):
            from flashrag.prompt.demons import example
            self.demons = {'normal': example , 'openai': example}
            from flashrag.prompt.demons_deepseek_r1 import example as example_deepseek
            self.demons['deepseek'] = example_deepseek
            from flashrag.prompt.demons_r1_search import example as example_r1_search
            self.demons['r1-searcher'] = example_r1_search
            self.demons['research'] = example_r1_search
            self.demons['ours'] = example_r1_search
            self.demons['ours-base'] = example_r1_search
            self.demons['search-refine'] = example_r1_search
            self.demons['simpledeep'] = example_r1_search
    def extract_search_content(self, text: str) -> str:
        try:
            if(self.config.get("prompt_type", "default") != "r1-searcher" and self.config.get("prompt_type", "default") != "ours" and self.config.get("prompt_type", "default") != "ours-base"):
                start_tag = '<search>'
                end_tag = '</search>'
            elif(self.config.get("prompt_type", "default") == "ours" or self.config.get("prompt_type", "default") == "ours-base"):
                start_tag = self.key_token["ours"][0]
                end_tag = self.key_token["ours"][1]
            else:
                start_tag = '<|begin_of_query|>'
                end_tag = '<|end_of_query|>'
            assert text.strip().endswith(end_tag)
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""

    def run_batch(self, items):
        print("=Begin="*10)
        # Assuming config contains necessary parameters similar to args in the original code
        config = self.config
        max_iterarion = config.get("max_iterations", 100)

        # Extract parameters from config
        temp = config.get("temperature", 0.7)
        topk = 7 #config.get("topk", 7)
        max_tokens = config.get("max_tokens", 512)
        prompt_type = config.get("prompt_type", "default")
        model_path = config.get("model_path", "")
        
        key_token = {"r1-searcher": ["<|begin_of_query|>", "<|end_of_query|>", "<|begin_of_documents|>", "<|end_of_documents|>"], \
        "default": ["<search>", "</search>","<documents>","</documents>"],
        "high_reasoning_and_high_retrieval_effort": ["<search>","</search>","<documents>","</documents>"], #["<|begin_search_query|>", "<|end_search_query|>", "<|begin_search_result|>", "<|end_search_result|>"], #["<search>","</search>","<documents>","</documents>"],
        "high_reasoning_and_low_retrieval_effort": ["<search>","</search>","<documents>","</documents>"],
        "low_reasoning_and_high_retrieval_effort": ["<search>","</search>","<documents>","</documents>"],
        "low_reasoning_and_low_retrieval_effort": ["<search>","</search>","<documents>","</documents>"],
        "normal": ["<search>","</search>","<documents>","</documents>"],
        "search-r1": ["<search>","</search>","<information>","</information>"],
        "openai": ["<search>","</search>","<documents>","</documents>"],
        "deepseek": ["<search>","</search>","<documents>","</documents>"],
        "research": ["<search>","</search>","<result>","</result>"],
        # "ours": ["<|begin_of_query|>", "<|end_of_query|>", "<|begin_of_documents|>", "<|end_of_documents|>"],
        "ours": ["<search>","</search>","<information>","</information>"],
        "search-refine": ["<search>","</search>","<documents>","</documents>"],
        "ours-base": ["<search>","</search>","<information>","</information>"],
        "simpledeep": ["<search>","</search>","<information>","</information>"],
        }
        self.key_token = key_token
        # Get the generator (LLM) and tokenizer from class attributes
        llm = self.generator
        tokenizer = self.generator.tokenizer if hasattr(self.generator, "tokenizer") else None
        
        # Process items in chunks if needed
        chunk_size = config.get("chunk_size", 20000)
        chunk_num = len(items) // chunk_size
        if len(items) % chunk_size != 0:
            chunk_num += 1
        
        all_results = []
        
        # Prepare output file path
        if not config.get("output_file", None):
            # Create default output file name based on input file if provided
            # src_file = config.get("src_file", "output")
            model_name = model_path.split("/")[-2] + model_path.split("/")[-1] if model_path else "model"
            # output_file = src_file.replace(".jsonl", f"-{model_name}_base_temp{temp}_type{prompt_type}.jsonl")
            output_file = os.path.join(self.config.save_dir, f"{model_name}_base_temp{temp}_type{prompt_type}.jsonl")
        for h in range(chunk_num):
            print("=="*80)
            print(f"Begin Chunk: {h} All: {chunk_num}")
            chunk_items = items[h*chunk_size:(h+1)*chunk_size]
            print("chunk_items 数据类型:", type(chunk_items))

            # Create a dataset from the items
            if tokenizer:
                # Process the items with tokenizer if available
                processed_items = []

                for item in chunk_items:
                    # Assuming item has question and answer keys
                    processed_item = {
                        "question": getattr(item, "question", ""),  # 使用 getattr 安全地访问属性
                        "answer": getattr(item, "golden_answers", ""), # this is the gold answer
                        "gen_text_store": "",  # Initialize empty storage for generated text
                    }
                    processed_items.append(processed_item)
                    
                # Define stop tokens
                # stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>", "</search>"]
                # sampling_params = {"temperature": temp, "top_p": 0.95, "max_tokens": max_tokens, "stop": stop_tokens}
                
                finished_all_list = []
                continued_items = copy.deepcopy(processed_items)
                
                for k in range(max_iterarion):
                    if len(continued_items) == 0:
                        print("All items processed.")
                        break
                    
                    prompts = [self._prepare_prompt(item) for item in continued_items]
                    if(self.config.get("framework") != "openai"):
                        outputs = llm.generate(input_list = prompts, return_raw_output = True) #可以更新sampling params
                    else:
                        outputs = llm.generate(input_list = prompts) #可以更新sampling params
                    
                    finished_items = []
                    next_continued_items = []
                    query_list = []
                    for i in range(1):
                        print(f"输出第{i}条")
                        print(outputs[i])
                        print('\n')
                    for i, output in enumerate(outputs):
                        prompt = output.prompt if hasattr(output, "prompt") else prompts[i]
                        item = continued_items[i]
                        answer = item.get("answer", "")
                        question = item.get("question", "")
                        gen_text_store = item.get("gen_text_store", "")
                        
                    
                        # Extract generated text and stop reason
                        if hasattr(output, "outputs") and output.outputs:
                            stop_reason = output.outputs[0].stop_reason
                            generated_text = output.outputs[0].text
                        else:
                            print('deepseek 输出',output)
                            # stop_reason = None
                            generated_text = output if isinstance(output, str) else ""
                            if(key_token[prompt_type][0] in generated_text):
                                stop_reason = key_token[prompt_type][1] 
                                generated_text += key_token[prompt_type][1]
                            elif('<answer>' in generated_text):
                                print('deepseek debug')  #因为这里是给openai这种 不包含end token的处理的，然后只有deepseek 有包含<answer>的格式，openai使用的是the final answer is 的格式。
                                stop_reason = '</answer>'
                                generated_text += '</answer>'
                            else:
                                stop_reason = None
                        # Handle maximum iterations reached
                        # if k == max_iterarion - 1:   
                        #     finished_items.append({
                        #         "question": question,
                        #         "gold_answer": answer,
                        #         "generated_text": generated_text,
                        #         "stop_reason_final": "many_retrieve",
                        #         "pred": "I don't know."
                        #     })
                        #     continue
                        
                        # Handle completed generation
                        #if(prompt_type in ['qwen','llama', ])
                        #if(prompt_type not in ['r1-searcher','search-r1','deepseek']:
                        if('The final answer is' in self.demons[prompt_type]):
                            # Handle search/retrieval requests
                            if key_token[prompt_type][0] in generated_text and  stop_reason == key_token[prompt_type][1]  :
                                try:
                                    query = self.extract_search_content(generated_text)
                                    if query:
                                        query_list.append(query)
                                        next_continued_items.append({
                                            "chat_prompt": prompt + generated_text.strip(),
                                            "answer": answer,
                                            "question": question,
                                            "stop_reason": stop_reason,
                                            "gen_text_store": gen_text_store + generated_text.strip()
                                        })
                                    else:
                                        finished_items.append({
                                            "question": question,
                                            "gold_answer": answer,
                                            "gen_text_store": gen_text_store + generated_text.strip(),
                                            "generated_text": generated_text,
                                            "stop_reason_final": "query_inst_error",
                                            "pred": "I don't know."
                                        })
                                except Exception as e:
                                    finished_items.append({
                                        "question": question,
                                        "gold_answer": answer,
                                        "gen_text_store": gen_text_store + generated_text.strip(),
                                        "generated_text": generated_text,
                                        "stop_reason_final": f"extraction_error: {str(e)}",
                                        "pred": "I don't know."
                                    })
                            elif "the final answer is" in generated_text.lower():
                                pred_ans = generated_text.lower().split("the final answer is")[-1]
                                finished_items.append({
                                    "question": question,
                                    "gold_answer": answer,
                                    "pred": pred_ans.replace(':','').strip(),
                                    "stop_reason_final": "finished",
                                    "gen_text_store": gen_text_store + generated_text,
                                })

                            
                            # Handle other cases (incomplete, error)
                            else:
                                print('打印其他情况', generated_text,"\n停止原因",stop_reason)
                                pred_ans = generated_text.lower().split(".")[-1]
                                finished_items.append({
                                    "question": question,
                                    "gold_answer": answer,
                                    "stop_reason_final": "shot_down",
                                    "pred": pred_ans,
                                    "gen_text_store": gen_text_store + generated_text,
                                })

                        elif('The final answer is' not in self.demons[prompt_type]):
                            # print(key_token[prompt_type][0],key_token[prompt_type][1])
                            if "<answer>" in generated_text and stop_reason == "</answer>":
                                pred_ans = generated_text.split("<answer>")[-1].split("</answer>")[0]
                                finished_items.append({
                                    "question": question,
                                    "gold_answer": answer,
                                    "pred": pred_ans,
                                    "stop_reason_final": "finished",
                                    "gen_text_store": gen_text_store + generated_text,
                                })
                            elif "<answer>" in generated_text and stop_reason == "<|im_end|>":
                                pred_ans = generated_text.split("<answer>")[-1].split("</answer>")[0]
                                finished_items.append({
                                    "question": question,
                                    "gold_answer": answer,
                                    "pred": pred_ans,
                                    "stop_reason_final": "finished",
                                    "gen_text_store": gen_text_store + generated_text,
                                })
                            elif "<answer>" in generated_text and stop_reason == "<|endoftext|>":
                                pred_ans = generated_text.split("<answer>")[-1].split("</answer>")[0]
                                finished_items.append({
                                    "question": question,
                                    "gold_answer": answer,
                                    "pred": pred_ans,
                                    "stop_reason_final": "finished",
                                    "gen_text_store": gen_text_store + generated_text,
                                })
                            # Handle search/retrieval requests
                            elif (key_token[prompt_type][0] in generated_text and  stop_reason == key_token[prompt_type][1]): #\
                              #  or (key_token[prompt_type][0] in generated_text):
                                try:
                                    #print('打印检索query',generated_text)
                                    query = self.extract_search_content(generated_text)
                                   # print('打印检索query2',query)
                                    if query:
                                        query_list.append(query)
                                        next_continued_items.append({
                                            "chat_prompt": prompt + generated_text.strip(),
                                            "answer": answer,
                                            "question": question,
                                            "stop_reason": stop_reason,
                                            "gen_text_store": gen_text_store + generated_text.strip()
                                        })
                                    else:
                                        finished_items.append({
                                            "question": question,
                                            "gold_answer": answer,
                                            "gen_text_store": gen_text_store + generated_text.strip(),
                                            "generated_text": generated_text,
                                            "stop_reason_final": "query_inst_error",
                                            "pred": "I don't know."
                                        })
                                except Exception as e:
                                    finished_items.append({
                                        "question": question,
                                        "gold_answer": answer,
                                        "gen_text_store": gen_text_store + generated_text.strip(),
                                        "generated_text": generated_text,
                                        "stop_reason_final": f"extraction_error: {str(e)}",
                                        "pred": "I don't know."
                                    })
                            
                            # Handle other cases (incomplete, error)
                            else:
                                print('打印其他情况', generated_text,"\n停止原因",stop_reason)
                                if "<answer>" in generated_text:
                                    pred_ans = generated_text.split("<answer>")[-1].split("</answer>")[0]
                                else:
                                    pred_ans = generated_text.lower().split(".")[-1]
                                finished_items.append({
                                    "question": question,
                                    "gold_answer": answer,
                                    "stop_reason_final": "shot_down",
                                    "pred": pred_ans,
                                    "gen_text_store": gen_text_store + generated_text,
                                })
                                # finished_items.append({
                                #     "question": question,
                                #     "gold_answer": answer,
                                #     "stop_reason_final": "shot_down",
                                #     "pred": "I don't know."
                                # })
                    print("=="*80)
                    print(f"Queries to process: {len(query_list)}")
                    
                    # Process retrieval if there are queries
                    if len(query_list) > 0:
                        assert len(query_list) == len(next_continued_items), "Error in len of query_list and next_continued_items"
                        
                        # Use the retriever to get documents
                        retrieved_docs_batch = self.retriever.batch_search(query_list, num = topk)
                        
                        for i, retrieved_docs in enumerate(retrieved_docs_batch):
                            item = next_continued_items[i]
                            
                            if retrieved_docs and len(retrieved_docs) > 0:
                                doc_content_list = []
                                for j, doc in enumerate(retrieved_docs):
                                    # Format document content
                                    doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)
                                    doc_text = re.sub(r'^\d+\s+', '', doc_text)  # Remove leading numbers
                                    doc_content_list.append(f"({j+1}){doc_text}\n")
                                
                                doc_content = ''.join(doc_content_list)
                            else:
                                doc_content = "None"
                            
                            # Add the retrieved documents to the prompt
                            item["chat_prompt"] = (item["chat_prompt"] + "\n" + #f"{key_token[prompt_type][1]}\n\n" +  
                                                f"{key_token[prompt_type][2]}\n" + doc_content + f"{key_token[prompt_type][3]}\n\n")
                            item["gen_text_store"] = (item["gen_text_store"] + "\n" + #f"{key_token[prompt_type][1]}\n\n" + 
                                                    f"{key_token[prompt_type][2]}\n" + doc_content + f"{key_token[prompt_type][3]}\n\n")
                    
                    # Add finished items to the final list
                    finished_all_list.extend(finished_items)
                    
                    # Write finished items to file for this iteration
                    if finished_items:
                        print(f"Begin Writing Epoch: {k}",output_file, finished_items[0].keys(), finished_items[0])
                        with open(output_file, "a") as f:
                            for item in finished_items:
                                # Create a copy of the item with transformed field names for file output
                                output_item = item.copy()
                                # Transform any field names to match the expected output format
                                if "gold_answer" in output_item:
                                    output_item["answer"] = output_item.pop("gold_answer")
                                if "pred" in output_item:
                                    output_item["pred_ans"] = output_item.pop("pred")
                                f.write(json.dumps(output_item) + "\n")
                    
                    # If no items are continuing, we're done
                    if len(next_continued_items) == 0:
                        print("No more items to continue processing.")
                        break
                    
                    # Otherwise, prepare for the next iteration
                    continued_items = next_continued_items
                    
                    print("=="*80)
                    print(f"Iteration: {k}, New Finished: {len(finished_items)}, "
                        f"All Finished: {len(finished_all_list)}, Continued: {len(next_continued_items)}")
                
                # Add all results from this chunk
                all_results.extend(finished_all_list)
        
        # Prepare the final items with consistent field names
        # final_items = []
        # for result in all_results:
        #     final_item = {
        #         "question": result.get("question", ""),
        #         "gold_answer": result.get("gold_answer", result.get("answer", "")),  # Use gold_answer if available, otherwise answer
        #         "pred": result.get("pred", result.get("pred_ans", ""))  # Use pred if available, otherwise pred_ans
        #     }
        #     final_items.append(final_item)
        totol_thinking = []
        total_retrieval = []
        think_length = []
        for i in all_results: 
            if('gen_text_store' in i):
                # totol_thinking.append(i['gen_text_store'].count('</think>'))
                if('<search>' in i['gen_text_store']):
                    total_retrieval.append(i['gen_text_store'].count('<search>'))
                else:
                    total_retrieval.append(i['gen_text_store'].count('<|begin_of_query|>'))
                text = clean_text(i['gen_text_store'])
                think_length.append(len(tokenizer.encode(text)))
        with open(self.config.result_output_file, "a") as f:
            f.write(json.dumps({
                "prompt_type": self.config.prompt_type,
                "save_dir": self.config.save_dir,
                "dataset_name": self.config.dataset_name,
                "generator_model_path": self.config.generator_model_path,
                "think的长度": sum(think_length)/len(think_length),
                "检索的步数": sum(total_retrieval)/len(total_retrieval),
            }) + "\n")
            
            # print('think的长度', sum(think_length)/len(think_length), file=f)
            # print('检索的步数', sum(total_retrieval)/len(total_retrieval), file=f)

        dataset_items = []
        for result in all_results:
            item_dict = {
                "question": result.get("question", ""),
                "golden_answers": result.get("gold_answer", result.get("answer", "")), 
                "output": {
                    "pred": result.get("pred", result.get("pred_ans", ""))
                }
            }
            # 创建 Item 实例
            dataset_items.append(Item(item_dict))
        
        # 创建 Dataset 实例
        final_dataset = Dataset(data=dataset_items)

        return final_dataset

    def _prepare_prompt(self, item):
        """Helper method to prepare prompts for the model"""
        # Customize this based on your specific requirements
        if "chat_prompt" in item:

            return item["chat_prompt"]
        
        # Create a prompt from question
        question = item['question'] if 'question' in item  else ""
        example = self.demons[self.config.get("prompt_type", "default")]
        # define the prompt template
        # 需要根据prompt_type来选择不同的prompt模板
        if self.config.get("prompt_type", "default") == "r1-searcher":
                prompt_template="""The User asks a question, and the Assistant solves it.
                The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
                The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
                During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
                Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>""".format(question=question)
        elif self.config.get("prompt_type", "default") == "ours":
                messages_chat=[
            {"role": "system", "content":  "You are a helpful assistant. Answer the given question. \
            You must reason **clearly and completely** inside <think> and </think> before providing any final answer. \
            Always identify and verify all key entities (e.g., person names, locations, dates, awards) mentioned in the question. \
            If you are uncertain about an entity or fact, or if the question requires external knowledge, you may use \
            <search>your query</search>, and the top search results will be returned between <information> and </information>. Carefully read and reflect on each newly retrieved piece of information. \
            You can search as many times as you want. \
            When reasoning, you must ensure your reasoning path aligns strictly with the evidence. \
            After reasoning, before providing your final answer, rethink it to make sure the answer is exactly correct for the original question. \
            Use the most accurate span from the evidence when possible. \
            Only after satisfying all the above, give the final answer inside <answer> and </answer>. For example, <answer> Beijing </answer>. \
            After outputting the final answer in <answer>  </answer>, You have one chance to reflect on the answer. If you choose not to reflect, nothing further needs to be done.\
            Otherwise, you can then re-examine your thinking process, the information obtained, and even search for more information to verify your previous answer or correct the previous answer. Remember, after reflection ends, you should output the  answer in <answer>  </answer>."
            }, 
                    {"role": "user", "content":question}
                ]
                prompt_template = self.generator.tokenizer.apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True) + "<think>"
        elif self.config.get("prompt_type", "default") == "ours-base":
            base_prompt = """Answer the given question. \
            You must reason **clearly and completely** inside <think> and </think> before providing any final answer. \
            Always identify and verify all key entities (e.g., person names, locations, dates, awards) mentioned in the question. \
            If you are uncertain about an entity or fact, or if the question requires external knowledge, you may use \
            <search>your query</search>, and the top search results will be returned between <information> and </information>. Carefully read and reflect on each newly retrieved piece of information. \
            You can search as many times as you want. \
            When reasoning, you must ensure your reasoning path aligns strictly with the evidence. \
            After reasoning, before providing your final answer, rethink it to make sure the answer is exactly correct for the original question. \
            Use the most accurate span from the evidence when possible. \
            Only after satisfying all the above, give the final answer inside <answer> </answer>. For example, <answer> Beijing </answer>. \
            After outputting the final answer in <answer> </answer>, You have one chance to reflect on the answer. If you choose not to reflect, nothing further needs to be done.\
            Otherwise, you can then re-examine your thinking process, the information obtained, and even search for more information to verify your previous answer or correct the previous answer. Remember, after reflection ends, you should output the  answer in <answer>  </answer>.
            
            Question:{question}
            
            \n<think>"""
            prompt_template = base_prompt.format(question=question)
        elif self.config.get("prompt_type", "default") == "search-r1":
                if('qwen' in self.config.get('generator_model_path')):
                    prompt_template = """
                        <|im_start|>system
                        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
                        <|im_start|>user
                        Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
                        <|im_end|>
                        <|im_start|>assistant
                        <think>""".format(question=question)
                elif('llama' in self.config.get('generator_model_path')):
                    prompt_template = """
                        You are a helpful assistant.
                        Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
                        After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. 
                        You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. 
                        Question: {question}""".format(question=question)
        elif self.config.get("prompt_type", "default") == "research":
                if('qwen' in self.config.get('generator_model_path').lower()):
                    prompt_template = """
                    <im_start>system 
                    You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. 
                    Given a question, you need to first think about the reasoning process in the mind and then provide the answer. 
                    During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. 
                    The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, 
                    and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. 
                    For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> 
                    <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>. 
                    In the last part of the answer, the final exact answer is enclosed within \\boxed{{}} with latex format.                    
                    <im_start>user
                    {question}
                    <im_end>
                    <|im_start|>assistant <think>
                    """.format(question=question)
        elif self.config.get("prompt_type", "default") == "openai":
                    example = """
                    Question:\n What nationality is the director of film Age Of Consent (Film)? <question_end>
                    assistant:\n   To solve this question, I need to find out the nationality of the director of the film \"Age of Consent (Film)\". 
                    \n\nnStep 1: Identify the director of \"Age of Consent (Film)\".\n\nStep 2: Determine the nationality of the director.
                    \n\nLet's start with Step 1:\n\nStep 1: Identify the director of \"Age of Consent (Film)\".\n\nI'll use a query to search for information about the director of \"Age of Consent (Film)\": 
                    \n\n<search> Age of Consent (Film) director </search>
                    <documents>\n(1)Age of Consent (film)\nAge of Consent (also known as Norman Lindsay's Age of Consent) is a 1969 Australian film which was the penultimate feature film directed by British director Michael Powell. 
                    The romantic comedy-drama stars James Mason (co-producer with Powell), Helen Mirren in her first major film role, and Irish character actor Jack MacGowran and features actress Neva Carr Glyn. 
                    The screenplay by Peter Yeldham was adapted from the 1938 semi-autobiographical novel of the same name by Norman Lindsay, who died the year this film was released. 
                    Plot. Bradley Morahan (James Mason) is an Australian artist who feels he has become jaded by\n(2)Age of Consent (film)\n- Andonia Katsaros as Isabel Marley - Michael Boddy as Hendricks - Harold Hopkins as Ted Farrell - Slim De Grey as Cooley - Max Meldrum as TV Interviewer - Frank Thring as Godfrey, the Art Dealer - Clarissa Kaye as Meg - Judith McGrath as Grace Production.
                    Norman Lindsay's novel had been published in 1938 and was banned in Australia. A film version was announced in 1961 by producer Oscar Nichols who said he wanted Dan O'Herlihy and Glynis Johns to star. In 1962 Michael Pate had the rights and he eventually brought in Michael Powell. They hired Peter\n(3)Age of consent (disambiguation)\nAge of consent is the minimum age at which a person is considered to be legally competent to consent to sexual acts. Age of consent may also refer to: Books. - \"Age of Consent\", a semi-autobiographical 1938 novel by Norman Lindsay - \"Age of Consent\", a 1987 novel by Joanne Greenberg - \"The Age of Consent\", a 1995 novel by American writer Geoffrey Wolff - \"The Age of Consent: A Manifesto for a New World Order\", a book by political writer George Monbiot - \"The Age of Consent\", a 1998 non-fiction book by Robert H. Knight Films. - \"Age of\n(4)Age of Consent (film)\nThe Powell & Pressburger Pages\n(5)The Age of Consent (film)\nThe Age of Consent is a 1932 American pre-Code drama film directed by Gregory La Cava. 
                    The film stars Richard Cromwell as a young man who becomes involved with a waitress of a seedy restaurant, co-starring Dorothy Wilson and Arline Judge. Plot. Mike Harvey (Cromwell) and Betty Cameron (Wilson) are college sweethearts at a New York college. 
                    They are so anxious to consummate their relationship that Mike suggests that the two of them quit college, get married and move to California where a friend of his has offered him a job. Both Mike's mentor, Professor Matthews, and Betty talk him\n</documents>
                    \n\nFrom the search results, I can see that \"Age of Consent (Film)\" is a 1969 Australian film directed by British director Michael Powell.\n\nStep 2: Determine the nationality of the director.\nThe nationality of Michael Powell is British.
                    \n\nTherefore, the nationality of the director of the film \"Age of Consent (Film)\" is British.\n\n The final answer is: British.
                    """
                    prompt_template = """
                    You are a helpful English assistant that answers user questions clearly and accurately. 
                    You can think through the problem step by step and use your internal knowledge or retrieve external information if needed.
                    If useful, you may include a reasoning process before giving the final answer. You also have access to a retrieval tool to help find relevant information when necessary.
                    The search query and retrieved information are enclosed within <search> </search> and <documents> </documents> tags, respectively. Please present the final answer after the phrase: **The final answer is**: . 
                    Here is a example of the output format:
                    <think>
                    Break down the question, analyze the problem, and identify what information is needed for reasoning. 
                    <search> 
                    Generate the most precise and relevant search queries to retrieve necessary information. 
                    </search>
                    <documents> 
                    Include the retrieved relevant information here. 
                    </documents>
                    ...(multiple cycles of reasoning and retrieval as needed)...
                    Integrate the retrieved information with logical reasoning. Test multiple approaches, verify steps, and ensure correctness. 
                    The final answer is: .
                    Here is a specific example:
                    {example}
                    Please follow the output format of the example to answer the following question.
                    Question: {question}
                    <question_end>""".format(question=question, example=example)
        elif self.config.get("prompt_type", "default") == "deepseek":
                example = """
                    Question:\n What nationality is the director of film Age Of Consent (Film)? <question_end>
                    assistant\n <think>\n To solve this question, I need to find out the nationality of the director of the film \"Age of Consent (Film)\". 
                    \n\nnStep 1: Identify the director of \"Age of Consent (Film)\".\n\nStep 2: Determine the nationality of the director.
                    \n\nLet's start with Step 1:\n\nStep 1: Identify the director of \"Age of Consent (Film)\".\n\nI'll use a query to search for information about the director of \"Age of Consent (Film)\": 
                    \n\n<search> Age of Consent (Film) director </search>
                    <documents>\n(1)Age of Consent (film)\nAge of Consent (also known as Norman Lindsay's Age of Consent) is a 1969 Australian film which was the penultimate feature film directed by British director Michael Powell. 
                    The romantic comedy-drama stars James Mason (co-producer with Powell), Helen Mirren in her first major film role, and Irish character actor Jack MacGowran and features actress Neva Carr Glyn. 
                    The screenplay by Peter Yeldham was adapted from the 1938 semi-autobiographical novel of the same name by Norman Lindsay, who died the year this film was released. 
                    Plot. Bradley Morahan (James Mason) is an Australian artist who feels he has become jaded by\n(2)Age of Consent (film)\n- Andonia Katsaros as Isabel Marley - Michael Boddy as Hendricks - Harold Hopkins as Ted Farrell - Slim De Grey as Cooley - Max Meldrum as TV Interviewer - Frank Thring as Godfrey, the Art Dealer - Clarissa Kaye as Meg - Judith McGrath as Grace Production.
                    Norman Lindsay's novel had been published in 1938 and was banned in Australia. A film version was announced in 1961 by producer Oscar Nichols who said he wanted Dan O'Herlihy and Glynis Johns to star. In 1962 Michael Pate had the rights and he eventually brought in Michael Powell. They hired Peter\n(3)Age of consent (disambiguation)\nAge of consent is the minimum age at which a person is considered to be legally competent to consent to sexual acts. Age of consent may also refer to: Books. - \"Age of Consent\", a semi-autobiographical 1938 novel by Norman Lindsay - \"Age of Consent\", a 1987 novel by Joanne Greenberg - \"The Age of Consent\", a 1995 novel by American writer Geoffrey Wolff - \"The Age of Consent: A Manifesto for a New World Order\", a book by political writer George Monbiot - \"The Age of Consent\", a 1998 non-fiction book by Robert H. Knight Films. - \"Age of\n(4)Age of Consent (film)\nThe Powell & Pressburger Pages\n(5)The Age of Consent (film)\nThe Age of Consent is a 1932 American pre-Code drama film directed by Gregory La Cava. 
                    The film stars Richard Cromwell as a young man who becomes involved with a waitress of a seedy restaurant, co-starring Dorothy Wilson and Arline Judge. Plot. Mike Harvey (Cromwell) and Betty Cameron (Wilson) are college sweethearts at a New York college. 
                    They are so anxious to consummate their relationship that Mike suggests that the two of them quit college, get married and move to California where a friend of his has offered him a job. Both Mike's mentor, Professor Matthews, and Betty talk him\n</documents>
                    \n\nFrom the search results, I can see that \"Age of Consent (Film)\" is a 1969 Australian film directed by British director Michael Powell.\n\nStep 2: Determine the nationality of the director.\nThe nationality of Michael Powell is British.
                    \n\nTherefore, the nationality of the director of the film \"Age of Consent (Film)\" is British.\n\n <answer> British </answer>.
                    """     
                prompt_template = """
                <｜begin▁of▁sentence｜>
                You are a helpful English assistant that answers user questions clearly and accurately. 
                You can think through the problem step by step and use your internal knowledge or retrieve external information if needed.
                If useful, you may include a reasoning process before giving the final answer. You also have access to a retrieval tool to help find relevant information when necessary.
                The search query and retrieved information are enclosed within <search> </search> and <documents> </documents> tags, respectively. The output format of the final answer is enclosed within <answer> </answer> tags. 
                Here is a example of the output format:
                <think>
                Break down the question, analyze the problem, and identify what information is needed for reasoning. 
                <search> 
                Generate the most precise and relevant search queries to retrieve necessary information. 
                </search>
                <documents> 
                Include the retrieved relevant information here. 
                </documents>
                ...(multiple cycles of reasoning and retrieval as needed)...
                Integrate the retrieved information with logical reasoning. Test multiple approaches, verify steps, and ensure correctness. 
                <answer> 
                final answer here
                </answer>
                Here is a specific example:
                {example}
                Please follow the output format of the example to answer the following question.
                <｜User｜>
                {question}
                <question_end>""".format(question=question, example=example)
        elif self.config.get("prompt_type", "default") == "search-refine":
            prompt_template = """ <|im_start|>system
                You are a helpful assistant who is good at answering questions with multi-turn search engine calling. \
                To answer questions, you must first reason through the available information using <think> and </think>. \
                If you identify missing knowledge, you may issue a search request using <search> query </search> at any time. \
                The retrieval system will provide you with the three most relevant documents enclosed in <documents> and </documents>. \
                After each search, you need to summarize and refine the existing documents in <refine> and </refine>. \
                You may send multiple search requests if needed. Once you have sufficient information, \
                provide a concise final answer using <answer> and </answer>.
                <|im_end|>
                <|im_start|>user
                Question: {question}
                <|im_end|>
                <|im_start|>assistant
                <think>
                """.format(question=question)
        elif self.config.get("prompt_type", "default") == "simpledeep":
            prompt_template = """<|im_start|>systemYou are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. You have special tools:\n\n- To perform a search: write <search> your query here </search>.\nThen, the system will search and analyze relevant web pages, then provide you with helpful information in the format <information> ...search results... </information>.\n\nWhenever you encounter a topic, fact, or piece of information you are uncertain about or need further details on, please perform a search to gather more accurate, up-to-date, or specific information. You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to 10.\n\nOnce you have all the information you need, continue your reasoning.\n\nRemember:\n- Use <search> to request a web search and end with </search>.\n- When done searching, continue your reasoning.\n- Do not generate <information> and </information> tags yourself.\n\nPlease answer the following question. You should provide your final answer in the format <answer> </answer>.\n\n
            <|im_end|>
            <|im_start|>user
            Question: {question}
            <|im_end|>
            <|im_start|>assistant
            """.format(question=question)

        elif self.config.get("prompt_type", "default") == "high_reasoning_and_high_retrieval_effort":
                prompt_template = self.single_demons_prompts[self.config.get("prompt_type")].format(question=question, example=example)
        elif self.config.get("prompt_type", "default") == "low_reasoning_and_high_retrieval_effort":
            prompt_template =    self.single_demons_prompts[self.config.get("prompt_type")].format(question=question, example=example)

        elif self.config.get("prompt_type", "default") == "high_reasoning_and_low_retrieval_effort":
            prompt_template =    self.single_demons_prompts[self.config.get("prompt_type")].format(question=question, example=example)

        elif self.config.get("prompt_type", "default") == "low_reasoning_and_low_retrieval_effort":
            prompt_template = self.single_demons_prompts[self.config.get("prompt_type")].format(question = question, example = example)
        elif self.config.get("prompt_type", "default") == "normal":
            prompt_template = self.single_demons_prompts[self.config.get("prompt_type")].format(question=question, example=self.demons[self.config.get("prompt_type")])
        return prompt_template

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        final_datasets = self.run_batch(dataset)
        
        dataset = self.evaluate(final_datasets, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

class IterativePipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, iter_num=3, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.iter_num = iter_num
        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever
        

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question
        # run in batch
        past_generation_result = []  # list of N items
        for iter_idx in range(self.iter_num):
            if iter_idx == 0:
                input_query = questions
            else:
                assert len(questions) == len(past_generation_result)
                input_query = [f"{q} {r}" for q, r in zip(questions, past_generation_result)]

            # generation-augmented retrieval
            retrieval_results = self.retriever.batch_search(input_query)
            dataset.update_output(f"retrieval_result_iter_{iter_idx}", retrieval_results)

            # retrieval-augmented generation
            # input_prompts = self.build_prompt(questions, retrieval_results)
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(questions, retrieval_results)
            ]

            dataset.update_output(f"prompt_iter_{iter_idx}", input_prompts)
            past_generation_result = self.generator.generate(input_prompts)
            dataset.update_output(f"pred_iter_{iter_idx}", past_generation_result)

        # use last retrieval result for evaluation
        dataset.update_output("retrieval_result", retrieval_results)

        dataset.update_output("pred", past_generation_result)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class SelfRAGPipeline(BasicPipeline):
    # Source: https://github.com/AkariAsai/self-rag
    # The code is released under MIT license

    rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
    retrieval_tokens_names = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
    utility_tokens_names = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    ground_tokens_names = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
    other_special_tokens = ["<s>", "</s>", "[PAD]", "<unk>", "<paragraph>", "</paragraph>"]
    control_tokens = [
        "[Fully supported]",
        "[Partially supported]",
        "[No support / Contradictory]",
        "[No Retrieval]",
        "[Retrieval]",
        "[Irrelevant]",
        "[Relevant]",
        "<paragraph>",
        "</paragraph>",
        "[Utility:1]",
        "[Utility:2]",
        "[Utility:3]",
        "[Utility:4]",
        "[Utility:5]",
    ]

    task_inst = {
        "wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
        "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
        "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
        "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
        "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
        "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
        "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
        "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
        "normal_qa": "Answer the following question, give me a short answer.",
    }

    def __init__(
        self,
        config,
        threshold=0.2,
        max_depth=2,
        beam_width=2,
        w_rel=1.0,
        w_sup=1.0,
        w_use=1.0,
        use_grounding=True,
        use_utility=True,
        use_seqscore=True,
        ignore_cont=True,
        mode="adaptive_retrieval",
        prompt_template=None,
        retriever=None,
        generator=None
    ):

        super().__init__(config, prompt_template)
        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever

        assert mode in ["adaptive_retrieval", "always_retrieve", "no_retrieval"]

        self.task = config["dataset_name"]
        self.task_instruction = self.task_inst.get(self.task, self.task_inst["normal_qa"])
        if self.task_instruction is not None:
            question_inst = self.task_instruction + "\n\n## Input:\n\n{question}"
        else:
            question_inst = "{question}"
        if prompt_template is None:
            self.prompt_template = PromptTemplate(
                config, user_prompt="### Instruction:\n" + question_inst + "\n\n### Response:\n", enable_chat=False
            )

        self.threshold = threshold
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.w_rel, self.w_sup, self.w_use = w_rel, w_sup, w_use
        self.use_grounding = use_grounding
        self.use_utility = use_utility
        self.use_seqscore = use_seqscore
        self.ignore_cont = ignore_cont
        self.mode = mode
        self.closed = self.task in ["fever", "arc_c"]
        tokenizer = AutoTokenizer.from_pretrained(config["generator_model_path"], padding_side="left")
        self.ret_tokens, self.rel_tokens, self.grd_tokens, self.ut_tokens = self.load_special_tokens(
            tokenizer, use_grounding=use_grounding, use_utility=use_utility
        )
        self.vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)

    def load_special_tokens(self, tokenizer, use_grounding, use_utility):
        ret_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in self.retrieval_tokens_names}
        rel_tokens = {}
        for token in ["[Irrelevant]", "[Relevant]"]:
            rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        grd_tokens = None
        if use_grounding is True:
            grd_tokens = {}
            for token in self.ground_tokens_names:
                grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        ut_tokens = None
        if use_utility is True:
            ut_tokens = {}
            for token in self.utility_tokens_names:
                ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        return ret_tokens, rel_tokens, grd_tokens, ut_tokens

    def judge_retrieve(self, input_prompts):
        """Calculate whether a retrieve is required based on the output probability of
        the special token in the model"""

        if self.mode == "always_retrieve":
            retrieval_flags = [True] * len(input_prompts)

        elif self.mode == "no_retrieval":
            retrieval_flags = [False] * len(input_prompts)

        else:
            # result for total batch
            all_pred_text = []
            all_pred_log_probs = []
            # For vllm, requesting too many logprobes can seriously affect speed
            # 20 probs is enough for calculate
            preds = self.generator.generate(input_prompts, return_raw_output=True, logprobs=20, max_tokens=1, skip_special_tokens=False)
            for single_pred in preds:
                pred_text = single_pred.outputs[0].text
                pred_log_probs = single_pred.outputs[0].logprobs
                all_pred_text.append(pred_text)
                all_pred_log_probs.append(pred_log_probs)

            retrieval_flags = []
            for idx, single_pred in enumerate(preds):
                if self.threshold is not None:
                    score_dict = {}
                    for tok, tok_id in self.ret_tokens.items():
                        if tok_id not in all_pred_log_probs[idx][0]:
                            score_dict[tok] = np.exp(-100)
                        else:
                            prob = all_pred_log_probs[idx][0][tok_id].logprob
                            score_dict[tok] = np.exp(prob)
                    do_retrieve = (
                        score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])
                        > self.threshold
                    )
                else:
                    do_retrieve = "[Retrieval]" in all_pred_text[idx]

                retrieval_flags.append(do_retrieve)

        return retrieval_flags

    def critic_preds(self, preds):
        """Evaluate predictions using different retrieval docs"""

        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        results = {}
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / max(len(pred.outputs[0].token_ids), 1)
            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            for tok, id in self.rel_tokens.items():
                prob = pred_log_probs[0][id].logprob if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))

            if self.grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(self.grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in self.grd_tokens.items():
                        prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                        grd_score_dict[p_idx][token] = np.exp(float(prob))
            utility_token_appear_indices = []
            if self.ut_tokens is not None:
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(self.ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in self.ut_tokens.items():
                        prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(float(prob))

            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values()))
            )

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx]["[Partially supported]"] / gt_sum
                )
            else:
                ground_score = 0.0

            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum(
                    [
                        ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i + 1)] / ut_sum)
                        for i in range(len(ut_scores))
                    ]
                )
            else:
                utility_score = 0.0

            if self.use_seqscore is True:
                final_score = (
                    np.exp(seq_score)
                    + self.w_rel * relevance_score
                    + self.w_sup * ground_score
                    + self.w_use * utility_score
                )
            else:
                final_score = self.w_rel * relevance_score + self.w_sup * ground_score + self.w_use * utility_score

            overall_scores[p_idx] = {
                "final_score": final_score,
                "relevance_score": relevance_score,
                "ground_score": ground_score,
                "utility_score": utility_score,
                "relevance_score_dict": relevance_score_dict,
                "grd_score_dict": grd_score_dict,
                "ut_score_dict": utility_score,
            }
            results["retrieval_{}".format(p_idx)] = {"pred": pred_text, "score": final_score}

        # modify and add do retrieve tokens (only used in long-form generation)
        final_preds = []
        if "[No Retrieval]" in pred_text:
            ret_token_appear_indices = []
            substrings = pred_text.split("[No Retrieval]")

            for tok_idx, tok in enumerate(pred_token_ids):
                if tok == self.ret_tokens["[No Retrieval]"]:
                    ret_token_appear_indices.append(tok_idx)

            ret_token_score_dict = {}
            retrieval_remap = {}
            for order, idx in enumerate(ret_token_appear_indices):
                ret_token_score_dict.setdefault(order, {})
                for tok, tok_id in self.ret_tokens.items():
                    prob = pred_log_probs[idx][tok_id].logprob if tok_id in pred_log_probs[idx] else -100
                    ret_token_score_dict[order][tok] = np.exp(prob)
                if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0:
                    do_retrieve = (
                        ret_token_score_dict[order]["[Retrieval]"]
                        + ret_token_score_dict[order]["[Continue to Use Evidence]"]
                    ) / (
                        ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]
                    ) > self.threshold
                else:
                    do_retrieve = 0.0
                if do_retrieve > self.threshold:
                    retrieval_remap[order] = True
                else:
                    retrieval_remap[order] = False
            processed_pred = ""
            for substr_i, substring in enumerate(substrings):
                if substr_i in retrieval_remap and retrieval_remap[substr_i] is True:
                    processed_pred += substring + "[Retrieval]"
                else:
                    processed_pred += substring + "[No Retrieval]"
            pred_text = processed_pred
            final_preds.append(pred_text)
        else:
            final_preds.append(pred_text)

        scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores]

        return results, final_preds, scores, overall_scores

    def postprocess_prediction(self, pred):
        def fix_spacing(input_text):
            # Add a space after periods that lack whitespace
            output_text = re.sub(r"(?<=\w)([.!?])(?=\w)", r"\1 ", input_text)
            return output_text

        for token in self.control_tokens:
            pred = pred.replace(token, "")
        if "</s>" in pred:
            pred = pred.replace("</s>", "")
        if "\n" in pred:
            pred = pred.replace("\n", "")
        if "<|endoftext|>" in pred:
            pred = pred.replace("<|endoftext|>", "")

        pred = pred.strip()
        if type(pred) is str and pred[0] == "#" or pred[0] == ":":
            pred = pred[1:]
        if len(pred) == 0:

            return ""

        return fix_spacing(pred)

    def select_best_prediction(self, results):
        answer2score = {}
        if self.closed is True:
            for key, result in results.items():
                answer = self.postprocess_prediction(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)
            best_pred = sorted_answers[0][0]
        else:
            path2score = {key: item["score"] for key, item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][0]
            best_pred = results[best_path]["pred"]

        return best_pred

    def run_single_beam(self, prompt, item_retrieval_result=None):
        curr_depth = 1
        terminated = False
        node_id = 0
        prediction_tree = {}
        levels = {}
        prediction_tree[node_id] = {
            "prompt": prompt,
            "pred": "[Retrieval]",
            "processed_pred": "",
            "score": None,
            "ctx": None,
            "parent": None,
        }
        levels[0] = [0]
        while curr_depth < self.max_depth:
            levels[curr_depth] = []
            if curr_depth - 1 in levels and terminated is False:
                for node in levels[curr_depth - 1]:
                    pred = prediction_tree[node]["pred"]
                    if pred == "</s>":
                        terminated = True
                        continue
                    prompt = prediction_tree[node]["prompt"]
                    prev_generation = prediction_tree[node]["processed_pred"]
                    score = prediction_tree[node]["score"]
                    if "[Retrieval]" in pred:
                        retrieval_results = {}

                        if item_retrieval_result is not None:
                            aug_prompts = [
                                prompt
                                + prev_generation
                                + "[Retrieval]"
                                + "<paragraph>{}</paragraph>".format(para["contents"])
                                for para in item_retrieval_result
                            ]
                        else:
                            aug_prompts = [prompt + prev_generation]

                        item_pred = self.generator.generate(aug_prompts, return_raw_output=True, logprobs=5)
                        _, preds, scores, overall_score_dict = self.critic_preds(item_pred)

                        for i, (pred, p_score) in enumerate(zip(preds, scores)):
                            retrieval_results[i] = {"pred": pred, "score": p_score}

                        for i, result in retrieval_results.items():
                            node_id += 1
                            node_score = result["score"] * score if score is not None else result["score"]
                            pred = result["pred"]
                            prediction_tree[node_id] = {
                                "prompt": prompt + prev_generation,
                                "pred": pred,
                                "score": node_score,
                                "ctx": item_retrieval_result[i],
                                "parent": node,
                                "overall_score_dict": overall_score_dict,
                            }

                            if "[Retrieval]" in pred:
                                gen_result_index = pred.index("[Retrieval]")
                                prev_generation = pred[:gen_result_index]
                            else:
                                prev_generation = pred
                            prediction_tree[node_id]["processed_pred"] = prev_generation
                            levels[curr_depth].append(node_id)

                current_rank = levels[curr_depth]
                node2score = {node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[: self.beam_width]
                levels[curr_depth] = [node[0] for node in top_nodes]
                curr_depth += 1
            else:
                break

        final_prediction = ""
        parent = 0
        best_selections = {}

        # Traverse from the bottom
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0}
        for path_i, node in enumerate(levels[len(levels)]):
            if node == 0:
                break
            best_selections[path_i] = [node]
            current_node = node
            current_level = curr_depth
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]["parent"]
                best_selections[path_i] = [parent] + best_selections[path_i]
                current_node = parent
                current_level += 1

        final_prediction = {}
        splitted_sentences = {}
        original_splitted_sentences = {}
        ctxs = {}
        for path_i, nodes in best_selections.items():
            final_prediction[path_i] = " ".join(
                [
                    prediction_tree[node]["processed_pred"]
                    for node in nodes
                    if node is not None
                    and (
                        self.ignore_cont is False
                        or (
                            self.ignore_cont is True
                            and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                        )
                    )
                ]
            )
            splitted_sentences[path_i] = [
                prediction_tree[node]["processed_pred"]
                for node in nodes
                if node is not None
                and (
                    self.ignore_cont is False
                    or (
                        self.ignore_cont is True
                        and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                    )
                )
            ]
            original_splitted_sentences[path_i] = [
                prediction_tree[node]["pred"]
                for node in nodes
                if node is not None
                and (
                    self.ignore_cont is False
                    or (
                        self.ignore_cont is True
                        and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                    )
                )
            ]
            ctxs[path_i] = [
                prediction_tree[node]["ctx"]
                for node in nodes
                if node is not None
                and (
                    self.ignore_cont is False
                    or (
                        self.ignore_cont is True
                        and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                    )
                )
            ]

        result = {
            "final_prediction": final_prediction,
            "splitted_sentences": splitted_sentences,
            "original_splitted_sentences": original_splitted_sentences,
            "best_selections": best_selections,
            "ctxs": ctxs,
            "prediction_tree": prediction_tree,
        }

        return final_prediction[0], result

    def postprocess_long_form(self, pred, intermediate):
        final_output = ""
        docs = []
        prev_gen = []
        if "splitted_sentences" not in intermediate:
            final_output = self.postprocess_prediction(pred)
        else:
            if len(self.postprocess_prediction(pred)) == 0:
                intermediate["splitted_sentences"][0], intermediate["ctxs"][0] = (
                    intermediate["splitted_sentences"][1],
                    intermediate["ctxs"][1],
                )
            for idx, (sent, doc) in enumerate(zip(intermediate["splitted_sentences"][0], intermediate["ctxs"][0])):
                if len(sent) == 0:
                    continue
                postprocessed_result = self.postprocess_prediction(sent)
                if postprocessed_result in prev_gen:
                    continue
                else:
                    prev_gen.append(postprocessed_result)
                final_output += postprocessed_result[:-1] + " [{}]".format(idx) + ". "
                docs.append(doc)
            if len(final_output) == 0:
                final_output = final_output
            if len(final_output) > 0 and final_output[-1] == " ":
                final_output = final_output[:-1]
            final_output = final_output.strip()
            final_output = final_output.replace(".[Continue to Use Evidence]", " [1]. ")
            final_output = final_output.replace(". [1] ", " [1]. ")

        return final_output

    def run_batch_pred_long_form(self, dataset):
        questions = dataset.question
        retrieval_results = self.retriever.batch_search(questions)
        dataset.update_output("retrieval_result", retrieval_results)

        # input_prompts = self.build_prompt(questions)
        input_prompts = [self.prompt_template.get_string(question=q) for q in questions]

        # determine whether to retrieve
        retrieval_flags = self.judge_retrieve(input_prompts)
        dataset.update_output("retrieval_flag", retrieval_flags)

        # for long form task, only support single item run
        for item, prompt, retrieval_flag in zip(dataset, input_prompts, retrieval_flags):
            if retrieval_flag:
                pred, intermediate_result = self.run_single_beam(prompt, item_retrieval_result=item.retrieval_result)
                item.update_output("intermediate_result", intermediate_result)

                if self.task == "factscore":
                    pred = self.postprocess_prediction(pred)
                else:
                    assert self.task in ["asqa", "eli5"]
                    pred = self.postprocess_long_form(pred, intermediate_result)
            else:
                prompt += "[No Retrieval]"
                pred = self.generator.generate(prompt)[0]

            item.update_output("pred", pred)

        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None, long_form=False):
        run_func = self.run_batch_pred_long_form if long_form else self.run_batch_pred
        
        # # to avoid oom, split the total dataset into small batches
        # all_dataset_list = []
        # for batch_dataset in tqdm(get_batch_dataset(dataset, batch_size=batch_size), desc="Batch dataset: "):
        #     batch_dataset = run_func(batch_dataset)
        #     all_dataset_list.append(batch_dataset)
        # dataset = merge_batch_dataset(all_dataset_list)

        dataset = run_func(dataset)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run_batch_pred(self, dataset):
        questions = dataset.question
        retrieval_results = self.retriever.batch_search(questions)
        dataset.update_output("retrieval_result", retrieval_results)

        # input_prompts = self.build_prompt(questions)
        input_prompts = [self.prompt_template.get_string(question=q) for q in questions]

        # determine whether to retrieve
        retrieval_flags = self.judge_retrieve(input_prompts)
        dataset.update_output("retrieval_flag", retrieval_flags)

        # process input item based on whether to retrieve
        all_input_list = []
        for idx, (prompt, item) in enumerate(zip(input_prompts, dataset)):
            retrieval_flag = retrieval_flags[idx]

            if retrieval_flag:
                retrieval_result = retrieval_results[idx]
                # for each doc in retrieval result, there is a prompt as input
                prompt_list = [
                    prompt + "[Retrieval]<paragraph>{}</paragraph>".format(para["contents"])
                    for para in retrieval_result
                ]
            else:
                prompt += "[No Retrieval]"
                prompt_list = [prompt]

            item.update_output("prompt", prompt_list)
            all_input_list += prompt_list

        batch_pred = self.generator.generate(all_input_list, return_raw_output=True, logprobs=5)

        # parse output based on retrieval flag
        pred_idx = 0
        pred_answer_list = []
        for idx, (retrieval_flag, item) in enumerate(zip(retrieval_flags, dataset)):
            if retrieval_flag:
                # for item that need retrieval, there may have more than one prediction
                item_pred = batch_pred[pred_idx : pred_idx + len(retrieval_results[idx])]
                pred_idx += len(retrieval_results[idx])
                critic_result, _, _, _ = self.critic_preds(item_pred)
                item.update_output("critic_result", critic_result)

                # select best prediction
                pred = self.select_best_prediction(critic_result)

            else:
                item_pred = batch_pred[pred_idx : pred_idx + 1][0]
                pred_idx += 1
                pred = item_pred.outputs[0].text

            pred = self.postprocess_prediction(pred)
            pred_answer_list.append(pred)

        dataset.update_output("pred", pred_answer_list)

        return dataset


class FLAREPipeline(BasicPipeline):
    def __init__(
        self,
        config,
        threshold=0.2,
        look_ahead_steps=64,
        max_generation_length=256,
        max_iter_num=5,
        prompt_template=None,
        retriever=None,
        generator=None
    ):
        super().__init__(config, prompt_template)
        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever

        self.threshold = threshold
        self.max_generation_length = max_generation_length
        self.max_iter_num = max_iter_num
        self.look_ahead_steps = look_ahead_steps
        self.stop_sym = list("!@#$%^&*()\n\n)(*&^%$#@!")

    def get_next_sentence(self, output, scores):
        tokenizer = self.generator.tokenizer
        text_sentences = re.split(r"(?<=[^A-Z].[.?]) +", output)
        if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            token_id_sentences = [tokenizer.encode(s, add_special_tokens=False) for s in text_sentences]
        else:
            token_id_sentences = [tokenizer.encode(s, allowed_special="all") for s in text_sentences]

        output_ids = tokenizer.encode(output, add_special_tokens=False)

        # assert sum([len(s) for s in token_id_sentences]) == len(
        #    output_ids), "token id sentences length not equal to output ids length"

        first_sent_ids = token_id_sentences[0]
        first_sent_score = scores[: len(first_sent_ids)]

        return text_sentences[0], first_sent_score

    def judge_sent_confidence(self, sent, sent_score):
        judge_result = all([score > self.threshold for score in sent_score])
        new_query = None
        if not judge_result:
            tokenizer = self.generator.tokenizer
            if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                sent_ids = tokenizer.encode(sent, add_special_tokens=False)
            else:
                sent_ids = tokenizer.encode(sent, allowed_special="all")
            # assert len(sent_ids) == len(sent_score)
            new_query_ids = [i for i, score in zip(sent_ids, sent_score) if score > self.threshold]
            new_query = tokenizer.decode(new_query_ids)
            if len(new_query) == 0:
                judge_result = True
        return judge_result, new_query

    def run_item(self, item):
        question = item.question
        gen_length = 0
        iter_round = 0
        final_gen_result = ""
        while gen_length < self.max_generation_length and iter_round < self.max_iter_num:
            input_prompt = self.prompt_template.get_string(question=question, previous_gen=final_gen_result)

            # input_prompt = self.build_prompt(
            #     question_list=[question], use_reference=False, previous_gen=final_gen_result)[0]
            # scores: token logits of the whole generation seq
            round_gen_output, scores = self.generator.generate(
                input_prompt, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps
            )
            round_gen_output, scores = round_gen_output[0], scores[0]
            # next_sent_scores: token logits of the first sent in generation seq
            next_sent, next_sent_score = self.get_next_sentence(round_gen_output, scores)
            # judge next sentence
            judge_result, query = self.judge_sent_confidence(next_sent, next_sent_score)
            item.update_output(f"judge_result_iter{iter_round}", judge_result)

            if not judge_result:
                # do retrieval-augmented generation
                retrieval_result = self.retriever.search(query)
                item.update_output("retrieval_result", retrieval_result)
                input_prompt = self.prompt_template.get_string(
                    question=question, retrieval_result=retrieval_result, previous_gen=final_gen_result
                )

                # input_prompt = self.build_prompt(
                #     question_list = [question],
                #     retrieval_results = [retrieval_result],
                #     previous_gen = final_gen_result)[0]
                output, scores = self.generator.generate(
                    input_prompt, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps
                )
                output, scores = output[0], scores[0]
                next_sent, _ = self.get_next_sentence(output, scores)
                item.update_output(f"gen_iter_{iter_round}", next_sent)
                item.update_output("retrieval_result", retrieval_result)

            final_gen_result += next_sent
            gen_length += len(next_sent_score)
            iter_round += 1

        item.update_output("pred", final_gen_result)

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        for item in tqdm(dataset, desc="Inference: "):
            self.run_item(item)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset


class SelfAskPipeline(BasicPipeline):
    FOLLOW_UP_PATTERN = r"Follow up:.*\n"

    def __init__(self, config, prompt_template=None, max_iter=5, single_hop=True, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        from flashrag.prompt.selfask_examplars import SELF_ASK_PROMPT_SINGLE_HOP, SELF_ASK_PROMPT_MULTI_HOP

        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever

        self.single_hop = single_hop
        self.max_iter = max_iter
        self.P_INS = SELF_ASK_PROMPT_SINGLE_HOP if self.single_hop else SELF_ASK_PROMPT_MULTI_HOP

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Context{idx+1}: {text}\n"

        return format_reference

    def _remove_duplicate_doc(self, docs):
        assert all(["id" in doc for doc in docs])
        new_doc_list = []
        exist_ids = []
        for doc in docs:
            doc_id = doc["id"]
            if doc_id not in exist_ids:
                exist_ids.append(doc_id)
                new_doc_list.append(doc)
        return new_doc_list

    def run_item(self, item):
        question = item.question
        retrieval_result = self.retriever.search(question)

        stop_condition = "Intermediate answer:"
        follow_ups = "No." if self.single_hop else "Yes."
        res = ""
        early_exit = False
        for idx in range(self.max_iter):
            input_prompt = (
                self.P_INS
                + "\n"
                + self.format_reference(retrieval_result)
                + f"\nQuesiton: {question}"
                + "\nAre follow up questions needed here: "
                + follow_ups
                + "\n"
                + res
            )
            gen_out = self.generator.generate(input_prompt, stop=["Context:", "#", stop_condition])[0]
            item.update_output(f"intermediate_output_iter{idx}", gen_out)

            if stop_condition == "Intermediate answer:":
                res += gen_out.split("Intermediate answer:")[0]
                stop_condition = "Follow up:"

            elif stop_condition == "Follow up:":
                followup_split = re.split(self.FOLLOW_UP_PATTERN, gen_out)
                res += followup_split[0]

                if len(followup_split) > 1:
                    res += re.findall(self.FOLLOW_UP_PATTERN, gen_out)[0]
                stop_condition = "Intermediate answer:"

            # make sure the result does not end in a new line
            if len(res) == 0:
                early_exit = True
                break
            if res[-1] == "\n":
                res = res[:-1]

            if "Follow up: " in gen_out:
                # get the first follow up
                new_query = [l for l in gen_out.split("\n") if "Follow up: " in l][0].split("Follow up: ")[-1]
                retrieval_result = self.retriever.search(new_query)

            if "So the final answer is: " in gen_out:
                res = (
                    self.format_reference(retrieval_result)
                    + f"\nQuesiton: {question}"
                    + "\nAre follow up questions needed here: "
                    + follow_ups
                    + "\n"
                    + res
                )
                early_exit = True
                # print("Success: early exit!")
                break

        if not early_exit:
            res = (
                self.format_reference(retrieval_result)
                + f"\nQuesiton: {question}"
                + "\nAre follow up questions needed here: "
                + follow_ups
                + "\n"
                + res
            )

        item.update_output("retrieval_result", retrieval_result)
        item.update_output("pred", res)

    def run(self, dataset, do_eval=True, pred_process_fun=selfask_pred_parse):
        for item in tqdm(dataset, desc="Inference: "):
            self.run_item(item)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    
class IRCOTPipeline(BasicPipeline):
    IRCOT_INSTRUCTION = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
    IRCOT_EXAMPLE = "Wikipedia Title: Kurram Garhi\nKurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\nWikipedia Title: 2001–02 UEFA Champions League second group stage\nEight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.\n\nWikipedia Title: Satellite tournament\nA satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.\n\nWikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\nWikipedia Title: Telephone numbers in Ascension Island\nCountry Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.\n\nQuestion: Are both Kurram Garhi and Trojkrsti located in the same country?\nThought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.\n\n"

    def __init__(
        self, config, prompt_template=None, max_iter=2, retriever=None, generator=None
    ):
        # if not provide prompt template, use default template provided by IRCOT
        if prompt_template is None:
            prompt_template = PromptTemplate(
                config=config,
                system_prompt=f"{self.IRCOT_INSTRUCTION}\n\n{self.IRCOT_EXAMPLE}",
                user_prompt="{reference}Question: {question}\nThought:",
                reference_template="Wikipedia Title: {title}\n{text}\n\n",
                enable_chat=False,
            )

        super().__init__(config, prompt_template)
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        self.max_iter = max_iter

    def run_batch(self, items):
        # Initialize the necessary data structures
        batch_thoughts = {item_id: [] for item_id in range(len(items))}
        iter_num = 0
        batch_retrieval_results = []
        doc2score_batch = []
        id2doc_batch = []

        # Initial retrieval for all items in the batch
        questions = [item.question for item in items]
        retrieval_results, scoress = self.retriever.batch_search(questions, return_score=True)
        for retrieval_result, scores in zip(retrieval_results,scoress):   
            
            doc2score = {doc_item['id']: score for doc_item, score in zip(retrieval_result, scores)}
            id2doc = {doc_item['id']: doc_item for doc_item in retrieval_result}
            batch_retrieval_results.append(retrieval_result)
            doc2score_batch.append(doc2score)
            id2doc_batch.append(id2doc)

        # Start the iterative process
        active_item_ids = list(range(len(items)))  # Track items that need more iterations
        while iter_num < self.max_iter:
            # Generate prompts and new thoughts for the active items
            input_prompts = [
                self.prompt_template.get_string(
                    question=items[item_id].question,
                    retrieval_result=batch_retrieval_results[item_id],
                    previous_gen=' '.join(batch_thoughts[item_id])
                )
                for item_id in active_item_ids
            ]

            # Batch generation for active items
            new_thoughts_batch = self.generator.generate(input_prompts, stop=['.', '\n'])
            
            # Update thoughts and determine next active items
            new_active_item_ids = []
            for idx, item_id in enumerate(active_item_ids):
                new_thought = new_thoughts_batch[idx]
                batch_thoughts[item_id].append(new_thought)
                
                # Check for termination condition
                # Store intermediate outputs
                if "So the answer is:" in new_thought:
                    items[item_id].update_output(
                        f'intermediate_output_iter{iter_num}', 
                        {
                            'input_prompt': input_prompts[idx],
                            'new_thought': new_thought,
                        },
                    )
                else:
                    new_active_item_ids.append(item_id)

            # Update active item IDs for the next iteration
            active_item_ids = new_active_item_ids

            # Perform batch retrieval for new thoughts of active items
            if active_item_ids:
                new_thoughts_for_retrieval = [batch_thoughts[item_id][-1] for item_id in active_item_ids]
                new_retrieval_results, new_scoress = self.retriever.batch_search(new_thoughts_for_retrieval, return_score=True)

                for i, item_id in enumerate(active_item_ids):
                    new_retrieval_result, new_scores = new_retrieval_results[i],new_scoress[i]
                    
                    # Update doc2score and id2doc for the current item
                    for doc_item, score in zip(new_retrieval_result, new_scores):
                        doc_id = doc_item['id']
                        id2doc_batch[item_id][doc_id] = doc_item
                        if doc_id in doc2score_batch[item_id]:
                            doc2score_batch[item_id][doc_id] = max(doc2score_batch[item_id][doc_id], score)
                        else:
                            doc2score_batch[item_id][doc_id] = score

                    # Sort and update retrieval results
                    sorted_doc_score = sorted(doc2score_batch[item_id].items(), key=lambda x: x[1], reverse=False)
                    sorted_doc_id = [t[0] for t in sorted_doc_score]
                    batch_retrieval_results[item_id] = [id2doc_batch[item_id][id] for id in sorted_doc_id]

            iter_num += 1

        # Final update for each item in the batch
        for item_id, item in enumerate(items):
            item.update_output('retrieval_result', batch_retrieval_results[item_id])
            item.update_output('pred', ' '.join(batch_thoughts[item_id]))

    def run(self, dataset, do_eval=True, pred_process_fun=ircot_pred_parse):

        self.run_batch(dataset)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

class RQRAGPipeline(BasicPipeline):
    expand_on_tokens = [
        "[S_Rewritten_Query]",
        "[S_Decomposed_Query]",
        "[S_Disambiguated_Query]",
        "[A_Response]"
    ]
    
    system_prompt = {
        "qa": "Given a question that requires multi-hop reasoning, you need to decompose the question and answer based on the given context. Please provide a short and concise response."
    }
    
    response_generation_params = {
        "temperature": 0,
        "top_p": 0.9,
        "stop": ["[EOS]", "</s>"],
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "logprobs": 1,
        "spaces_between_special_tokens": False,
        "max_tokens": 4096
    }
    
    other_generation_params = {
        "temperature": 1,
        "top_p": 0.9,
        "stop": ["[EOS]", "</s>"],
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "logprobs": 1,
        "spaces_between_special_tokens": False,
        "max_tokens": 4096
    }

    def __init__(
        self,
        config: dict,
        prompt_template = None,
        retriever = None,
        generator = None,
        max_depth = 3,
        batch_size = 32
    ):
        super().__init__(config, prompt_template)

        self.generator = generator if generator is not None else get_generator(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config["generator_model_path"], padding_side = "left")
        self.retriever = retriever if retriever is not None else get_retriever(config)
        
        self.max_depth = max_depth
        self.batch_size = batch_size
        
        # Due to the low effiency of original method, it only supports vllm now.
    
    def preprocess_eval_data(self, items: List) -> List[str]:
        eval_examples = []

        for item in items:
            eval_example = f"<s><|system|>\n{self.system_prompt['qa']}" + self.tokenizer.eos_token + "\n<|user|>\n" + item.question + self.tokenizer.eos_token + "\n"
            eval_example += "<|assistant|>\n"
            eval_examples.append(eval_example)

        return eval_examples

    def format_evidences(self, evidences: List[str]):
        format_evidence = ""
        for evidence in evidences:
            title = evidence['contents'].split('\n')[0]
            text = "\n".join(evidence['contents'].split('\n')[1:])
            format_evidence += f"Title: {title}\n"
            format_evidence += f"Text: {text}\n"
        return format_evidence

    def generate_tree_of_thoughts_batch(self, initial_prompts_batch: List[str]):
        paths_batch_dict = {
            idx: [{
                "prompt": initial_prompt,
                "depth": 0,
                "done": False
            }]
            for idx, initial_prompt in enumerate(initial_prompts_batch)
        }
        
        final_outputs_batch = {idx: [] for idx in range(len(initial_prompts_batch))}
        
        while any(paths for paths in paths_batch_dict.values()):
            current_batch = []
            for i, _ in paths_batch_dict.items():
                if paths_batch_dict[i]:
                    current_path = paths_batch_dict[i].pop(0)
                    current_batch.append(current_path)
                else:
                    continue
            
            if not current_batch:
                break
            
            for special_token in self.expand_on_tokens:
                
                if current_batch[0]["depth"] >= self.max_depth and special_token != "[A_Response]":
                    continue
                
                # Prepare for inputs
                input_texts = [path["prompt"] + special_token for path in current_batch]
            
                # Generate outputs
                if special_token != "[A_Response]":
                    init_outputs = self.generator.generate(
                        input_list = input_texts,
                        return_raw_output = True,
                        **self.response_generation_params
                    )
                else:
                    init_outputs = self.generator.generate(
                        input_list = input_texts,
                        return_raw_output = True,
                        **self.other_generation_params
                    )

                # Decode outputs
                decoded_outputs = [output.outputs[0].text for output in init_outputs]
                # Initialize lists to collect queries for batch retrieval
                queries_for_search = []
                
                # Process outputs and prepare for retrieval
                for i, decoded_output in enumerate(decoded_outputs):
                    current_path = current_batch[i]
                    decoded_output = decoded_output.replace("<s> ", "<s>")
                    
                    if special_token == "[A_Response]":
                        pattern = r"(.*?)\[EOS\]"
                        matches = re.findall(pattern, decoded_output, re.DOTALL)
                        result = matches[-1].strip() if matches else "Unable to detect valid answer"
                        token_ids = init_outputs[i].outputs[0].token_ids[1:-1]
                        logprobs = init_outputs[i].outputs[0].logprobs[1:-1]
                        confidence = 0
                        for token_id, logprobs in zip(token_ids, logprobs):
                            logprob = logprobs[token_id].logprob
                            prob = math.exp(logprob)
                            confidence += prob
                        
                        if len(token_ids) > 0:
                            confidence /= len(token_ids)
                        
                        new_path = {
                            "prompt": input_texts[i] + decoded_output,
                            "depth": current_path["depth"],
                            "done": True,
                            "final_answer": result,
                            "confidence": confidence
                        }
                        final_outputs_batch[i].append(new_path)
                    else:
                        # Extract the query
                        pattern = r"(.*?)\[EOS\]"
                        matches = re.findall(pattern, decoded_output, re.DOTALL)
                        query_for_search = matches[-1].strip() if matches else "dummy"
                        queries_for_search.append(query_for_search)
                
                # Perform batch retrieval
                if queries_for_search:
                    batch_search_results = self.retriever.batch_search(queries_for_search)
                    
                    for i, decoded_output in enumerate(decoded_outputs):
                        search_results = batch_search_results[i]
                        format_evidence = self.format_evidences(search_results)
                        new_prompt = decoded_output + "[R_Evidences]" + format_evidence + "[/R_Evidences]"
                        new_path = {
                            "prompt": input_texts[i] + new_prompt,
                            "depth": current_path["depth"] + 1,
                            "done": False,
                        }
                        paths_batch_dict[i].append(new_path)

        final_outputs_batch_list = [final_outputs_batch[i] for i in range(len(initial_prompts_batch))]
        
        return final_outputs_batch_list

    def select_best_path_single_turn(self, final_outputs):
        # After generating all paths, we can select the best answer
        # Compute perplexity and confidence for each path
        
        scores = []
        for path in final_outputs:
            confidence = path["confidence"]
            path["confidence"] = confidence
            scores.append((path, confidence))

        # Select the path with the highest confidence
        best_path = max(scores, key = lambda x: x[1])[0]  # x[2] is confidence
        pred = best_path["final_answer"]

        return pred, best_path

    def run(self, dataset, do_eval = True):
        preds = []
        meta_results = []

        for i in tqdm(range(0, len(dataset), self.batch_size), position=0, desc='RQRAG Process'):
            batch_items = dataset[i : i + self.batch_size]
            eval_datas = self.preprocess_eval_data(batch_items)
            paths_batch = self.generate_tree_of_thoughts_batch(initial_prompts_batch = eval_datas)
            for paths in paths_batch:
                pred, best_path = self.select_best_path_single_turn(paths)
                preds.append(pred)
                meta_results.append(best_path)


        dataset.update_output("paths", meta_results)
        dataset.update_output("pred", preds)

        dataset = self.evaluate(dataset, do_eval = do_eval)
        return dataset
    
