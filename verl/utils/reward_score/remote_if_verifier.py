import requests
import random
import os
import time
import json
from .local_server import local_serve
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_if_verifier_rm_reward(response):
    score = response["result"][0]
    return score

def parse_solution(text):
    return text.split("</think>")[-1].strip()


def local_request(data_list):
    max_threads = 128  # 并发线程数
    results = [None] * len(data_list)  # 预分配结果列表

    with ThreadPoolExecutor(max_threads) as executor:
        future_to_index = {
            executor.submit(local_serve, data): idx
            for idx, data in enumerate(data_list)
        }

        for future in tqdm(as_completed(future_to_index), total=len(data_list), desc="Processing"):
            idx = future_to_index[future]
            results[idx] = parse_if_verifier_rm_reward(future.result())
    return results



def compute_score(prompt_str, solution_str, ground_truth):
    solution_str = [parse_solution(_solution_str) for _solution_str in solution_str]
    data = []
    for i in range(len(solution_str)):
        data.append({
            "instruction": prompt_str[i],
            "answers": [solution_str[i]],
            "labels": ground_truth[i]
        })
    try:
        results = local_request(data)
    except:
        print("Error!", "!"*100)
        return [-1.0] * len(prompt_str)
    print(results)
    return results


    # headers = {
    #     'Content-Type': 'application/json'
    # }

    # wait_seconds = 0.1
    # max_retry = 5
    # for attempt in range(1, max_retry + 1):
    #     try:
    #         response = requests.post(url, json=payload, headers=headers, timeout=900)
    #         response.raise_for_status()
    #         return parse_if_verifier_rm_reward(response.json())

    #     except requests.exceptions.RequestException as e:
    #         if attempt < max_retry:
    #             print(
    #                 f"Attempt {attempt} failed: {e}. Retrying in {wait_seconds} seconds...")
    #             time.sleep(wait_seconds)
    #         else:
    #             print(f"Attempt {attempt} failed: {e}. No more retries.")
    #             # raise  # 在达到最大重试次数后，抛出最后一个异常
    #             return [-1.0] * len(prompt_str)



if __name__ == "__main__":
    data = {
        "prompt_str": ["I'm brainstorming for a story and would like your assistance in fleshing out the beginning. I'm not too interested in how the story ends, only the beginning. The story is about a man that discovered he has the power to change his size at will. The catch, he's in the middle of a zombie apocalypse and seemed to have missed the evacuation. Making him the sole, non-zombie character. What are things that he might start doing?\n\nThe response should consist of 5-7 paragraphs, with a blank line separating each paragraph to enhance readability.\nAvoid using overly graphic or violent descriptions of zombie encounters, and refrain from using a first-person narrative voice.\nThe response should be written in a speculative and analytical tone, with a focus on exploring the possibilities and implications of the protagonist's powers in a zombie apocalypse setting, while maintaining a neutral sentiment and a moderate level of formality.\nThe first sentence should begin with 'Since he was not able to receive news of a zombie apocalypse...' to establish the protagonist's isolation.\nEmploy hypothetical scenarios and thought experiments to explore the protagonist's abilities and their consequences, and use rhetorical questions to encourage further speculation and exploration.\nPrioritize the development of the protagonist's initial actions and reactions to his newfound power, followed by his exploration of its capabilities and limitations, and finally, his interactions with the military and zombies.\nThe response should primarily use complex sentences with multiple clauses to convey the protagonist's thoughts and actions, and occasionally employ short, simple sentences for emphasis or to convey a sense of urgency."],
        "solution_str": ["I'm brainstorming for a story and would like your assistance in fleshing out the beginning."],
        "ground_truth": [json.dumps({
            "checkers": ["Hierarchical_Instructions"],
            "functions": ["from llm_call import llm_judge, llm_extract, llm_score\ndef check_following(instruction, response):\n    return llm_score(instruction, response, \"1. Hierarchical_Instructions: Prioritize the development of the protagonist's initial actions and reactions to his newfound power, followed by his exploration of its capabilities and limitations, and finally, his interactions with the military and zombies. 2.Specific_Literary_Devices: Employ hypothetical scenarios and thought experiments to explore the protagonist's abilities and their consequences, and use rhetorical questions to encourage further speculation and exploration. 3.Specific_Sentence: The first sentence should begin with 'Since he was not able to receive news of a zombie apocalypse...' to establish the protagonist's isolation. 4.Paragraphs_Constraints: The response should consist of 5-7 paragraphs, with a blank line separating each paragraph to enhance readability. 5.Desired_Writing_Style: The response should be written in a speculative and analytical tone, with a focus on exploring the possibilities and implications of the protagonist's powers in a zombie apocalypse setting, while maintaining a neutral sentiment and a moderate level of formality. 6.Specific_Grammatical_Structure: The response should primarily use complex sentences with multiple clauses to convey the protagonist's thoughts and actions, and occasionally employ short, simple sentences for emphasis or to convey a sense of urgency. 7.Morphological_Constraints: Avoid using overly graphic or violent descriptions of zombie encounters, and refrain from using a first-person narrative voice.\")"]
        })]
    }
    print(compute_score(data["prompt_str"], data["solution_str"], data["ground_truth"]))




# import requests
# import random
# import os
# import time


# def parse_if_verifier_rm_reward(response):
#     score = response["result"]
#     return score


# def parse_solution(text):
#     return text.split("</think>")[-1].strip()


# def compute_score(prompt_str, solution_str, ground_truth):
#     solution_str = [parse_solution(_solution_str) for _solution_str in solution_str]
#     url = os.environ["REMOTE_IF_VERIFIER_URL"]
#     payload = {"prompt": prompt_str, "answers": solution_str, "labels": ground_truth}

#     headers = {
#         'Content-Type': 'application/json'
#     }

#     wait_seconds = 0.1
#     max_retry = 5
#     for attempt in range(1, max_retry + 1):
#         try:
#             response = requests.post(url, json=payload, headers=headers, timeout=900)
#             response.raise_for_status()
#             return parse_if_verifier_rm_reward(response.json())

#         except requests.exceptions.RequestException as e:
#             if attempt < max_retry:
#                 print(
#                     f"Attempt {attempt} failed: {e}. Retrying in {wait_seconds} seconds...")
#                 time.sleep(wait_seconds)
#             else:
#                 print(f"Attempt {attempt} failed: {e}. No more retries.")
#                 # raise  # 在达到最大重试次数后，抛出最后一个异常
#                 return [-1.0] * len(prompt_str)



# if __name__ == "__main__":
#     data = {
#         "prompt_str": ["I'm brainstorming for a story and would like your assistance in fleshing out the beginning. I'm not too interested in how the story ends, only the beginning. The story is about a man that discovered he has the power to change his size at will. The catch, he's in the middle of a zombie apocalypse and seemed to have missed the evacuation. Making him the sole, non-zombie character. What are things that he might start doing?\n\nThe response should consist of 5-7 paragraphs, with a blank line separating each paragraph to enhance readability.\nAvoid using overly graphic or violent descriptions of zombie encounters, and refrain from using a first-person narrative voice.\nThe response should be written in a speculative and analytical tone, with a focus on exploring the possibilities and implications of the protagonist's powers in a zombie apocalypse setting, while maintaining a neutral sentiment and a moderate level of formality.\nThe first sentence should begin with 'Since he was not able to receive news of a zombie apocalypse...' to establish the protagonist's isolation.\nEmploy hypothetical scenarios and thought experiments to explore the protagonist's abilities and their consequences, and use rhetorical questions to encourage further speculation and exploration.\nPrioritize the development of the protagonist's initial actions and reactions to his newfound power, followed by his exploration of its capabilities and limitations, and finally, his interactions with the military and zombies.\nThe response should primarily use complex sentences with multiple clauses to convey the protagonist's thoughts and actions, and occasionally employ short, simple sentences for emphasis or to convey a sense of urgency."],
#         "solution_str": ["I'm brainstorming for a story and would like your assistance in fleshing out the beginning."],
#         "ground_truth": [json.dumps({
#             "checkers": ["Hierarchical_Instructions"],
#             "functions": ["from llm_call import llm_judge, llm_extract, llm_score\ndef check_following(instruction, response):\n    return llm_score(instruction, response, \"1. Hierarchical_Instructions: Prioritize the development of the protagonist's initial actions and reactions to his newfound power, followed by his exploration of its capabilities and limitations, and finally, his interactions with the military and zombies. 2.Specific_Literary_Devices: Employ hypothetical scenarios and thought experiments to explore the protagonist's abilities and their consequences, and use rhetorical questions to encourage further speculation and exploration. 3.Specific_Sentence: The first sentence should begin with 'Since he was not able to receive news of a zombie apocalypse...' to establish the protagonist's isolation. 4.Paragraphs_Constraints: The response should consist of 5-7 paragraphs, with a blank line separating each paragraph to enhance readability. 5.Desired_Writing_Style: The response should be written in a speculative and analytical tone, with a focus on exploring the possibilities and implications of the protagonist's powers in a zombie apocalypse setting, while maintaining a neutral sentiment and a moderate level of formality. 6.Specific_Grammatical_Structure: The response should primarily use complex sentences with multiple clauses to convey the protagonist's thoughts and actions, and occasionally employ short, simple sentences for emphasis or to convey a sense of urgency. 7.Morphological_Constraints: Avoid using overly graphic or violent descriptions of zombie encounters, and refrain from using a first-person narrative voice.\")"]
#         })]
#     }
#     print(compute_score(data["prompt_str"], data["solution_str"], data["ground_truth"]))