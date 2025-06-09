import concurrent.futures

from code_executer import execute_code
from llm_call import llm_score


def mean_func(numbers):
    mean_num = sum(numbers) / len(numbers)
    return mean_num

# def all_func(numbers):
#     if 0 in numbers:
#         return 0
#     return 1


def _evaluate_reward(instruction, response, functions, reduction="mean"):
    def process_function(function):
        result, _ = execute_code(instruction, response, function)
        return 1 if result else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_function, functions))
    if reduction == "mean":
        return mean_func(results)
    else:
        return results



def evaluate_if_reward_multi(instruction, answers, checker_names=None, functions=None):
    llm_checkers, llm_functions, rule_checkers, rule_functions = [], [], [], []
    for checker, function in zip(checker_names, functions):
        if "[llm]" in checker:
            llm_checkers.append(checker)
            llm_functions.append(function)
        elif "[rule]" in checker:
            rule_checkers.append(checker)
            rule_functions.append(function)
        else:
            pass

    # single processing
    scores = []
    if len(rule_functions) != 0:
        for answer in answers:
            scores.append(_evaluate_reward(instruction, answer, rule_functions, reduction="mean"))
    else:
        scores = [1] * len(answers)
        
    # quality scores
    q_scores = []
    for answer in answers:
        q_scores.append(llm_score(instruction, answer, llm_checkers))

    # return scores
    return {
        "overall": [_scores * q_scores[i] for i, _scores in enumerate(scores)]
    }


