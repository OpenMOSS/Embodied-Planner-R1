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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List, Any, Dict, Tuple, Sequence, Union
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
import copy
import re
import requests
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from vllm.outputs import RequestOutput
# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def extract_action(text):
    # 使用正则表达式匹配Action:后面的内容
    pattern = r'Action:\s*(.*?)(?:\n|$)'
    match = re.search(pattern, text)
    
    if match:
        return match.group(1).strip()
    else:
        return ""

class AlfRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, server_url, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = int(self.config.get('max_num_batched_tokens', 8192))

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')
        

        if vllm_version == '0.7.0+': 
            self.inference_engine = LLM(
                model=model_path,
                # actor_module_path,
                # tokenizer=model_path,
                # model_hf_config=model_hf_config,
                enable_sleep_mode=True,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="external_launcher",
                dtype=config.dtype,
                enforce_eager=config.enforce_eager,
                gpu_memory_utilization=config.gpu_memory_utilization,
                disable_custom_all_reduce=True,
                skip_tokenizer_init=False,
                max_model_len=max_model_len,
                # load_format=config.load_format,
                disable_log_stats=config.disable_log_stats,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=config.enable_chunked_prefill,
                enable_prefix_caching=True,
            )
        else:
            raise NotImplementedError

        # Offload vllm model to reduce peak memory usage
        # self.inference_engine.offload_model_weights()
        self.inference_engine.sleep(1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        # self.sampling_params = kwargs

        self.pad_token_id = tokenizer.pad_token_id

        # self.chat_template = tokenizer.get_chat_template()
        self.debug = 0
        self.server_url = server_url
        self.ping()

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def ping(self):
        repeat_time = 0
        max_waiting_time = 100000
        while True:
            try:
                res = requests.get(self.server_url + "/health")
                if res.status_code == 200:
                    return True
            except:
                assert repeat_time < max_waiting_time, f"server is not ready in {str(max_waiting_time)} s, please check the server status."
                repeat_time += 1
                logging.info(f"server is not ready, wait for {str(repeat_time * 5)} s")
                time.sleep(repeat_time * 5)

    def fake_init(self, game_file, **kwargs):
        if self.debug:
            res = {}
            observation_str = ["You arrive at shelf 1. On the shelf 1, you see a candle 2, and a soapbar 1."] * len(game_file)
            task_str = ["put soapbar into shelf"] * len(game_file)
        else:
            # /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/open-embodied-r1/alfworld_server/data/json_2.1.1/valid_seen/pick_cool_then_place_in_recep-Pan-None-DiningTable-7/trial_T20190908_232648_241836/game.tw-pddl
            res = requests.post(self.server_url + "/reset", json={'game_file': game_file.tolist(), 'batch_size': len(game_file)}) 
            res = res.json()['observations']

            outputs = [r.split("\n\n") for r in res]
            _, observation_str, task_str = zip(*outputs)

            observation_str = list(observation_str)
            task_str = list(task_str)
        return observation_str, task_str

    def fake_step(self, batch_steps, **kwargs):
        if self.debug:
            res = {}
            res['observations'] = ['Nothing happens.'] * len(batch_steps)
            res['scores'] = [0] * len(batch_steps)
            res['dones'] = [0] * len(batch_steps)
        else:
            res = requests.post(self.server_url + "/step", json={'actions': batch_steps})
            res = res.json()
 
        return res['observations'], res['scores'], res['dones']
    
    def _generate(
        self,
        states: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: SamplingParams) -> Tuple[List[Dict[str, Any]], List[RequestOutput]]:

        outputs = self.inference_engine.chat([s["messages"] for s in states], sampling_params=self.sampling_params, use_tqdm=False)
        # print(outputs)
        # breakpoint()
        batch_action = []
        for i, state in enumerate(states):
            if state["skip_flag"] == True:
                batch_action.append("")
                continue
            
            text = self.inference_engine.get_tokenizer().decode(outputs[i].outputs[0].token_ids, skip_special_tokens=True)
            state["messages"].append({
                "role": "assistant", 
                "content": text
            })
            batch_action.append(extract_action(text))
            # Track prompt_tokens to later slice out the completion part
            if state["prompt_tokens"] == -1:
                state["prompt_tokens"] = len(outputs[i].prompt_token_ids)
        
        return states, batch_action, outputs

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # breakpoint()
        # TODO: only for debug
        # self.debug = 1
        self.max_steps = prompts.meta_info['max_steps']
        self.max_length = prompts.meta_info['max_length']
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        game_file = prompts.non_tensor_batch['game_file']

        all_completion_ids = [None] * len(game_file)
        all_completion_mask = [None] * len(game_file)
        all_prompt_completion_mask = [None] * len(game_file)
        all_position_ids = [None] * len(game_file)
        
        # idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # # left-padded attention_mask
        # attention_mask = prompts.batch['attention_mask']
        # position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        system_prompt = prompts.meta_info['system_prompt']

        batch_size = len(game_file)

        # idx_list = []
        # # parse idx from torch.Tensor to List[List[str]]
        # for i in range(batch_size):
        #     idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        system_prompt = prompts.meta_info.get('system_prompt',"Interact with a household to solve a task.")
        
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):

            system_info, task = self.fake_init(game_file)
            # system_prompt = self.get_system_prompt(system_info)
            system_prompt = system_prompt

            states = [{
                "messages": [
                        {
                            "content": system_prompt,
                            "role": "system"
                        },
                        {
                            "content": f"{task[i]}\n\nObservation:{system_info[i]}",
                            "role": "user"
                        }
                ], 
                "completed": False, 
                "skip_flag": False,
                "prompt_tokens": -1
            } for i in range(len(game_file))]

            # breakpoint()
            completion_mask = [[] for _ in states]
            prompt_outputs_ids = [[] for _ in states]

            pre_prompt_length = [0 for _ in states]
            origin_prompt_length = [0 for _ in states]
            
            max_steps = copy.deepcopy(self.max_steps) 
            while any([s['skip_flag'] != True for s in states]) and max_steps > 0:
                max_steps -= 1
                states, batch_action, outputs = self._generate(states, self.inference_engine, self.sampling_params)
                # breakpoint()
                batch_obs, batch_scores, batch_reward = self.fake_step(batch_action)

                for i, state in enumerate(states):
                    
                    if states[i]["skip_flag"] == True:
                        continue

                    if batch_action[i] == "done":
                        states[i]["skip_flag"] = True
                    if states[i]["completed"] == False and batch_scores[i] == 1:
                        states[i]["completed"] = True
                    
                    states[i]["messages"].append({
                        "role": "tool",
                        "content": "Observation:" + batch_obs[i]
                    })
                
                    prompt_token_ids = outputs[i].prompt_token_ids
                    token_ids = outputs[i].outputs[0].token_ids
                
                    prompt_outputs_ids[i] = prompt_token_ids + list(token_ids)

                    pre_prompt_length[i] = len(prompt_token_ids)
                    # if origin_prompt_length[i] == 0:
                    #     # origin_prompt_idx is the first prompt
                    #     origin_prompt_length[i] = pre_prompt_length[i]
                    completion_mask[i].extend([0 for _ in range(len(completion_mask[i]), pre_prompt_length[i])])

                    completion_mask[i].extend([1 for _ in range(len(token_ids))])

                    if len(prompt_token_ids) + len(token_ids) > self.max_length:
                        states[i]["skip_flag"] = True
            
            for i, state in enumerate(states):

                if state["completed"] == True:
                    state["messages"].append({
                        "role": "user",
                        "content": "SUCCESS"
                    })
                else:
                    state['messages'].append({
                        "role": "user",
                        "content": "FAIL"
                    })
            
            final_output = self.inference_engine.get_tokenizer().apply_chat_template([s['messages'] for s in states])
            # breakpoint()
            for i, state in enumerate(states):
                # 提取完整上下文 all_completion_ids，包括系统消息、用户输入、助手回复等
                prompt_outputs_ids[i] = final_output[i]
                completion_mask[i].extend([0 for _ in range(len(completion_mask[i]), len(prompt_outputs_ids[i]))])

                all_completion_ids[i] = prompt_outputs_ids[i][origin_prompt_length[i]:]
                all_completion_mask[i] = completion_mask[i][origin_prompt_length[i]:]

                if len(all_completion_ids[i]) > self.max_length:
                    all_completion_ids[i] = all_completion_ids[i][:self.max_length]
                    all_completion_mask[i] = all_completion_mask[i][:self.max_length]

                # all_prompt_completion_mask: 用于计算 logits 的有效区域，所有有效部分置为 1
                all_prompt_completion_mask[i] = [1 for _ in all_completion_ids[i]]

                # 填充完整上下文和 Mask 到统一长度 self.max_length（右填充）
                padded_completion_ids = all_completion_ids[i] + [self.pad_token_id] * (self.max_length - len(all_completion_ids[i]))
                padded_completion_mask = all_completion_mask[i] + [0] * (self.max_length - len(all_completion_mask[i]))
                padded_prompt_completion_mask = all_prompt_completion_mask[i] + [0] * (self.max_length - len(all_prompt_completion_mask[i]))

                # 添加到列表
                all_completion_ids[i] = padded_completion_ids
                all_completion_mask[i] = padded_completion_mask
                all_prompt_completion_mask[i] = padded_prompt_completion_mask

                # 生成 position_ids
                # 右填充的 position_ids，填充部分为 0，有效部分从 0 开始递增
                position_ids = list(range(self.max_length))  # 直接从 0 开始递增
                all_position_ids[i] = position_ids

        # 转换为 Tensor 格式
        input_ids = torch.tensor(all_completion_ids, dtype=torch.long)
        input_mask = torch.tensor(all_completion_mask, dtype=torch.long)
        attention_mask = torch.tensor(all_prompt_completion_mask, dtype=torch.long)
        position_ids = torch.tensor(all_position_ids, dtype=torch.long)

        # breakpoint()

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'input_ids': input_ids,  # the whole sentences
                'input_mask': input_mask, # the assistant is 1
                'attention_mask': attention_mask, # whole sentence is 1
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
