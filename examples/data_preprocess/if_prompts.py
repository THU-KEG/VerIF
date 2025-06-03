"""
Preprocess the IFEval dataset to parquet format
"""

import re
import os
import json
import datasets
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def load_dataset(data_paths):
    def _load_dataset(data_path):
        data = []
        with open(data_path) as f:
            for line in tqdm(f.readlines()):
                item = json.loads(line.strip())
                functions = ["import sys\nsys.path.append('/mnt/ph/ScaleIF/verl/verl/utils/reward_score/local_server')\n"+function.replace("local_server", "llm_call") for function in item["functions"]]
                data.append({
                    "id": "NA",
                    "prompt": item["prompt"],
                    "checkers": item["checkers"],
                    "functions": functions
                })
        return data
    
    data = []
    for data_path in data_paths:
        data.extend(_load_dataset(data_path))
    
    return data

# def generate_llm_functions(llm_checker):
#     code = f'from llm_call import llm_judge, llm_extract, llm_score\ndef check_following(instruction, response):\n    return llm_score(instruction, response, {repr(llm_checker)})'
#     return code


# def load_dataset(data_paths):
#     def _load_dataset(data_path):
#         data = []
#         with open(data_path) as f:
#             for line in tqdm(f.readlines()):
#                 item = json.loads(line.strip())
#                 # functions = [function for function in item["functions"] if "llm_score" not in function]
#                 # if len(functions) == 0:
#                 #     continue

#                 functions = []
#                 for checker, function in zip(item["checkers"], item["functions"]):
#                     if checker.startswith("[rule]"):
#                         functions.append(function.replace("local_server", "llm_call"))
#                     elif checker.startswith("[llm]"):
#                         # functions.append(generate_llm_functions(checker))
#                         continue
#                     else:
#                         raise ValueError()
#                 if len(functions) == 0:
#                     continue

#                 data.append({
#                     # "id": item["id"],
#                     "id": "NA",
#                     "prompt": item["prompt"],
#                     "checkers": item["checkers"],
#                     "functions": functions
#                 })
#         return data
    
#     data = []
#     for data_path in data_paths:
#         data.extend(_load_dataset(data_path))
    
#     return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/if_prompts_3_4')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    
    # data_paths = []
    # for i in range(5, 6):
        # data_paths.append(f"/mnt/ph/ScaleIF/data/WildChat/train-0000{i}-of-00006-labels.jsonl")
    
    # data_paths = ["/mnt/ph/ScaleIF/data/WildChat/post_train_0402/rl_data.jsonl"]
    # data_paths = ["/mnt/ph/ScaleIF/data/crab/crab_labels.json", "/mnt/ph/ScaleIF/data/crab/crab_labels_2.jsonl"]
    # data_paths = ["/mnt/ph/ScaleIF/data/crab/crab_labels.json"]
    data_paths = ["/mnt/ph/ScaleIF/data/crab/crab_labels_0411.jsonl"]
 
    data_source = data_paths[0].split("/")[-2].lower()
    # dataset = datasets.load_dataset("json", data_files=data_paths)["train"]
    data_list = load_dataset(data_paths)
    dataset = datasets.Dataset.from_list(data_list)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('prompt')
            checkers = example.pop('checkers')
            functions = example.pop('functions')
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "if",
                "reward_model": {
                    "style": "rm",
                    "ground_truth": json.dumps({
                        "checkers": checkers,
                        "functions": functions
                    })
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)
    # columns_to_keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    # train_dataset = train_dataset.remove_columns(
    #     [col for col in train_dataset.column_names if col not in columns_to_keep]
    # )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
