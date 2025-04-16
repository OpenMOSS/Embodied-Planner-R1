import hydra
import numpy as np
import re
import torch
import torch.distributed
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig
from collections import defaultdict
import os
from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn, JSONDataset
from verl.workers.reward_manager import AlfRewardManager
# from verl.workers.rollout.hf_rollout import HFRollout
from verl.workers.rollout.vllm_rollout.sci_rollout_for_test import SciRollout
import logging


@hydra.main(config_path='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/verl/trainer/config', config_name='ppo_trainer')
def main(config):
    # Loading huggingface-style checkpoint
    logger = logging.getLogger(__name__)
    # model_path = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/verl/eval/scienceworld/test"

    model_path = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/verl/eval/scienceworld/easy_150"

    # model_path = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/verl/outputs_v2/20250404/alf_v2/20250404_1749/rank_0/ckpt/global_step_10/actor/huggingface"

    val_dataset = JSONDataset("/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/data/ScienceWolrd/valid_dataset.json")

    val_dataloader = DataLoader(
        dataset=val_dataset,
        # Validation datasets are sent to inference engines as a whole batch,
        # which will schedule the memory themselves.
        batch_size=8,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn)

    assert len(val_dataloader) >= 1

    tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
    actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    val_reward_fn = AlfRewardManager(
        tokenizer=tokenizer, num_examine=0, compute_score=None
    )
    
    rollout = SciRollout(model_path=model_path,
                            config=config.actor_rollout_ref.rollout,
                            tokenizer=tokenizer,
                            model_hf_config=actor_model_config,
                            server_url=8001)
    system_prompt='You are a helpful assistant to do some scientific experiment in an environment.\nYou should explore the environment and find the items you need to complete the experiment.\n\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway.\nYou can teleport to any room in one step.\nThe available actions are:\nactivate OBJ\nclose OBJ\nconnect OBJ to OBJ\ndeactivate OBJ\ndisconnect OBJ\ndunk OBJ in OBJ\neat OBJ\nflush OBJ\nfocus on OBJ\ngo LOC\ninventory\nlook around\nlook at OBJ\nlook in OBJ\nmix OBJ\nmove OBJ to OBJ\nopen OBJ\npick up OBJ\npour OBJ in OBJ\nput down OBJ\nread OBJ\nuse OBKJ on OBJ\nteleport to LOC\nwait: wait 10 steps\nwait1: wait 1 step\ntask: check your task\ndone: indicate that you believe the task is complete\nWhen arrive a new location, you should use look around to check the OBj you can interact with.\nUse focus on OBJ only neccessary as incorrect use will cause environment ends.\nDo not proceed with any further exploration or actions until you receive the feedback from the environment after your action.\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>'
    reward_tensor_lst = []
    for test_data in val_dataloader:
        # breakpoint()
        test_batch = DataProto.from_single_dict(test_data)

        test_batch = test_batch.pop(
            batch_keys=['_dummy'],
            non_tensor_batch_keys=['task', 'var'],
        )

        test_batch = test_batch.repeat(repeat_times=1,
                                        interleave=True)
        test_gen_batch = test_batch
        test_gen_batch.meta_info = {
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': False,
            'validate': True,
            'system_prompt': system_prompt,
            'max_length': 4096,
            'max_steps': 30,
            'easy': config.data.easy
        }
        logger.info(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

        # pad to be divisible by dp_size
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, 1)
        test_output_gen_batch_padded = rollout.generate_sequences(test_gen_batch_padded)

        # breakpoint()
        # unpad
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

        # Store generated outputs
        output_ids = test_output_gen_batch.batch['input_ids']
        output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        folder_path = f'./{config.data.dir}' 
        if not os.path.exists(folder_path):
            # 创建文件夹
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, 'outputs.txt')
        with open(file_path, 'a') as f:
            separator="\n\n=====\n\n"
            content = separator.join(output_texts)
            f.write(content)

        test_batch = test_batch.union(test_output_gen_batch)

        # evaluate using reward_function
        reward_tensor = val_reward_fn(test_batch)

        # Store scores
        scores = reward_tensor.sum(-1).cpu().tolist()
        logger.info(f'scores: {scores}')
        # sample_scores.extend(scores)

        reward_tensor_lst.append(reward_tensor)

    reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
    metric_dict = {}
    metric_dict[f'val/test_score'] = reward_tensor.mean(-1)

    logger.info(f'metric_dict: {metric_dict}')

if __name__ == "__main__":
    main()
