# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict
import numpy as np



def z_score_normalize(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-5)


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # data_source = data_item.non_tensor_batch['data_source']
            data_source = "hf_rm"

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                prompt_str=prompt_str,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # reward_sources = ["hf_rm", "if_rm", "if_verifier"]
        reward_sources = ["if_verifier"]
        data_infos = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)


            data_infos["data_source"].append(data_source)
            data_infos["prompt_str"].append(prompt_str)
            data_infos["response_str"].append(response_str)
            data_infos["ground_truth"].append(ground_truth)
            data_infos["extra_info"].append(extra_info)
            data_infos["valid_response_length"].append(valid_response_length)

            # for data_source in reward_sources:
            #     score = self.compute_score(
            #         data_source=data_source,
            #         prompt_str=prompt_str,
            #         solution_str=response_str,
            #         ground_truth=ground_truth,
            #         extra_info=extra_info,
            #     )
            #     # reward_tensor[i, valid_response_length - 1] = score
            #     all_rewards[data_source].append(score)

            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print("[prompt]", prompt_str)
            #         print("[response]", response_str)
            #         print("[ground_truth]", ground_truth)
            #         print("[score]", score)
        
        all_rewards = defaultdict(list)
        for reward_source in reward_sources:
            scores = self.compute_score(
                data_source=reward_source,
                prompt_str=data_infos["prompt_str"],
                solution_str=data_infos["response_str"],
                ground_truth=data_infos["ground_truth"],
                extra_info=data_infos["extra_info"],
            )
            all_rewards[reward_source] = scores

        for key in all_rewards:
            # scores = z_score_normalize(all_rewards[key])
            scores = all_rewards[key]
            for i in range(len(scores)):
                reward_tensor[i, data_infos["valid_response_length"][i] - 1] += scores[i]

        return reward_tensor
