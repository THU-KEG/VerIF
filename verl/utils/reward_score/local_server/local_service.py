from constraint_analyzer import evaluate_if_reward_multi
import json


def local_serve(request):
    item = json.loads(request["labels"])
    checkers, functions = item["checkers"], item["functions"]
    result = evaluate_if_reward_multi(request["instruction"], request["answers"], checkers, functions)
    return {"result": result["overall"]}
