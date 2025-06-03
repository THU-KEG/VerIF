import requests
import random
import os
import time


def parse_if_verifier_rm_reward(response):
    score = response[0]
    return score


def compute_score(prompt_str, solution_str, ground_truth):
    url = os.environ["REMOTE_IF_URL"]
    payload = {"prompt": prompt_str, "answers": [solution_str], "kwargs": ground_truth}

    headers = {
        'Content-Type': 'application/json'
    }

    wait_seconds = 0.1
    max_retry = 5
    for attempt in range(1, max_retry + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return parse_if_verifier_rm_reward(response.json())

        except requests.exceptions.RequestException as e:
            if attempt < max_retry:
                print(
                    f"Attempt {attempt} failed: {e}. Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                print(f"Attempt {attempt} failed: {e}. No more retries.")
                # raise  # 在达到最大重试次数后，抛出最后一个异常
                return -1.0