import requests
import random
import os
import time


def parse_if_rm_reward(response):
    score = response["rewards"][0]
    return score

def get_messages(prompt, response):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

def parse_solution(text):
    return text.split("</think>")[-1].strip()

def compute_score(prompt_str, solution_str, ground_truth):
    solution_str = parse_solution(solution_str)
    url = os.environ["REMOTE_HF_RM_URL"]
    query = [
        get_messages(prompt_str, solution_str)
    ]
    data = {
        "query": query,
        "prompts": [prompt_str]
    }
    headers = {
        'Content-Type': 'application/json'
    }

    wait_seconds = 0.1
    max_retry = 5
    for attempt in range(1, max_retry + 1):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            return parse_if_rm_reward(response.json())

        except requests.exceptions.RequestException as e:
            if attempt < max_retry:
                print(
                    f"Attempt {attempt} failed: {e}. Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                print(f"Attempt {attempt} failed: {e}. No more retries.")
                # raise  # 在达到最大重试次数后，抛出最后一个异常
                return -1.0