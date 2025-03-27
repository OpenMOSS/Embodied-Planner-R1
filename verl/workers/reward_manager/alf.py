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


def alfworld_done_reward(completion):
    completion_str = completion
    reward = 0
    if completion_str.endswith("user\nSUCCESS\n"):
        reward = 1.0
    else:
        reward = 0.0
    return reward

class AlfRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = alfworld_done_reward

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
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        # breakpoint()
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['input_ids'], dtype=torch.float32)

        # already_print_data_sources = {}
        already_print = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem


            prompt_completion_ids = data_item.batch['input_ids']
            attention_mask = data_item.batch['attention_mask']

            prompt_completion_length = attention_mask.sum()

            valid_prompt_completion_ids = prompt_completion_ids[:prompt_completion_length]

            # decode
            prompt_completion_str = self.tokenizer.decode(valid_prompt_completion_ids, skip_special_tokens=True)

            # data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                completion=prompt_completion_str,
            )
            reward_tensor[i, prompt_completion_length - 1] = score

            # if data_source not in already_print_data_sources:
            #     already_print_data_sources[data_source] = 0

            if already_print < self.num_examine:
                already_print += 1
                print("[dialogue]", prompt_completion_str)
                print("[score]", score)

        return reward_tensor
