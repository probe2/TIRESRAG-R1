from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        question = data["question"]
        idx = data["idx"]
        answer = data["answer"]
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


        prompt = apply_chat_template(messages_chat, tokenize=False, add_generation_prompt=True) + "<think>"

    else:

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
        question = data["question"]
        idx = data["idx"]
        answer = data["answer"]
        prompt = base_prompt.format(question=question)

    return str(idx) + "<|idx_prompt_split|>" + prompt + "<pro_answer>" + answer


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)
        # print("len(self.prompts):",len(self.prompts))
        # print("self.prompts[0:5]:",self.prompts[0:5])
        # kill

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
