import os
from typing import List
from copy import deepcopy
import warnings
from tqdm import tqdm
import numpy as np
import threading
import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI
import tiktoken
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

_background_loop = None

def get_background_loop():
    global _background_loop
    if _background_loop is None:
        _background_loop = asyncio.new_event_loop()
        t = threading.Thread(target=lambda: _background_loop.run_forever(), daemon=True)
        t.start()
    return _background_loop

class OpenaiGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self._config = config
        self.update_config()
        
        # load openai client
        if "api_type" in self.openai_setting and self.openai_setting["api_type"] == "azure":
            del self.openai_setting["api_type"]
            self.client = AsyncAzureOpenAI(**self.openai_setting)
        else:
            self.client = AsyncOpenAI(**self.openai_setting)
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except Exception as e:
            print("Error: ", e)
            warnings.warn("This model is not supported by tiktoken. Use gpt-3.5-turbo instead.")
            self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()
    
    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()

    def update_base_setting(self):
        self.model_name = self._config["generator_model"]
        self.batch_size = self._config["generator_batch_size"]
        self.generation_params = self._config["generation_params"]

        self.openai_setting = self._config["openai_setting"]
        if self.openai_setting["api_key"] is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")

    def update_additional_setting(self):
        pass
    
    async def _get_response(self, messages: List, **params):
        m = messages.split('<question_end>')
        if(m[2] != ''):
            if(self.model_name == 'deepseek-reasoner'):
                messages = [{"role": "user", "content":  m[1]+'<question_end>'+m[1]+'<question_end>'},{'role': 'assistant', 'content': " ".join(m[2:]), "prefix": True}]
            else:
                messages = [{"role": "user", "content":  m[1]+'<question_end>'+m[1]+'<question_end>'},{'role': 'assistant', 'content': " ".join(m[2:])}]
            print('打印信息长度',messages)
        else:
            messages = [{"role": "user", "content": messages}]
            print('打印信息长度10',len(m), messages)

        # print('打印信息',params)
        if('max_tokens' in params):
            params.pop('max_tokens')
        try:    
            response = await self.client.chat.completions.create(
                model=self.model_name, messages=messages, **params
            )
        except Exception as e:
            print('打印错误',e)
            return  Choice( finish_reason='stop',index=0, logprobs=None, message=ChatCompletionMessage(content="None", role="assistant", function_call=None, tool_calls=None))
        if not response.choices:
            raise ValueError("No choices returned from API.")
        return response.choices[0]

    async def _get_batch_response(self, input_list: List[List], batch_size, **params):
        tasks = [self._get_response(messages, **params) for messages in input_list]
        all_results = []
        for idx in tqdm(range(0, len(tasks), batch_size), desc="Generation process: "):
            batch_tasks = tasks[idx: idx + batch_size]
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
        return all_results

    async def _generate_async(self, input_list: List, batch_size=None, return_scores=False, **params) -> List[str]:
        if isinstance(input_list, dict):
            input_list = [[input_list]]
        elif isinstance(input_list[0], dict):
            input_list = [input_list]

        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        generation_params.pop("do_sample", None)

        max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)  or params.pop("max_completion_tokens", None)
        if max_tokens is not None:
            generation_params["max_completion_tokens"] = max_tokens
        else:
            generation_params["max_completion_tokens"] = generation_params.get(
                "max_completion_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        if return_scores:
            generation_params["logprobs"] = True
            warnings.warn("Set logprobs to True to get generation scores.")


        results = await self._get_batch_response(input_list, batch_size, **generation_params)

        response_texts = []
        scores = []
        for res in results:
            response_texts.append(res.message.content)
            if return_scores:
                score = np.exp([item.logprob for item in res.logprobs.content])
                scores.append(score)
        return (response_texts, scores) if return_scores else response_texts

    # ----------------- 同步包装接口 -----------------
    def generate(self, input_list: List, batch_size=None, return_scores=False, **params) -> List[str]:
        loop = get_background_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._generate_async(input_list, batch_size=batch_size, return_scores=return_scores, **params),
            loop
        )
        return future.result()
