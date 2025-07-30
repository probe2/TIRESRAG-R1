import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=100):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=360)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, score_key="rewards", group_size = None, exit_flags = None, global_step = None, total_training_steps = None):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    # print("sht-debug-"*10)
    # print(api_url)
    # print({"query": queries})
    # print("score_key",score_key)
    # print("sht-debug-"*10)
    if(group_size != None):
        scores = request_api_wrapper(api_url, {"query": queries,"group_size": group_size, "exit_flags": exit_flags, "global_step": global_step, "total_training_steps": total_training_steps}, score_key)
    else:
        scores = request_api_wrapper(api_url, {"query": queries}, score_key)
    return scores
    # return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, score_key="rewards", group_size = None, exit_flags = None, global_step = None, total_training_steps = None):
    return remote_rm_fn(api_url, queries, score_key, group_size, exit_flags, global_step, total_training_steps)


if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_rm_score"
    score = remote_rm_fn(url, ["example query"], ["example response"])
    print(score)
