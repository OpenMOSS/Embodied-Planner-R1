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
from verl.workers.rollout.vllm_rollout.alf_rollout_for_test import AlfRollout
import logging
import json
from tqdm import tqdm

def check_success(completion_str):
    if completion_str.endswith("user\nSUCCESS\n"):
        return True
    else:
        return False

def get_task_type(game_file):

    game_name = game_file.split("/")[13]
    task_type = game_name.split("-")[0]
    return task_type

@hydra.main(config_path='/embodied-r1/verl/trainer/config', config_name='ppo_trainer')
def main(config):
    # Loading huggingface-style checkpoint
    logger = logging.getLogger(__name__)

    model_path = "/path/to/ckpt"

    val_dataset = JSONDataset("get_data/rl/alf_valid_unseen.json")

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
    
    rollout = AlfRollout(model_path=model_path,
                            config=config.actor_rollout_ref.rollout,
                            tokenizer=tokenizer,
                            model_hf_config=actor_model_config,
                            server_url=8000)
    system_prompt='You are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:Thought: your thoughts.\nAction: your next action.\n\nThe available actions are:\n1. `go to (receptacle)`\n2. `open (receptacle)`\n3. `close (receptacle)`\n4. `take (object) from (receptacle)`\n5. `move (object) to (receptacle)`\n6. `examine (something) with (object)`\n7. `use (object)`\n8. `heat (object) with (receptacle)`\n9. `clean (object) with (receptacle)`\n10. `cool (object) with (receptacle)`\n11. `slice (object) with (object)` - slice an object using a sharp object\n12. `look` - look around your current location\n13. `inventory` - check your current inventory\n14. `done` - Indicate that you believe the task is complete\nWhere `(object)` refers to manipulable objects and `(receptacle)` refers to receptacles or locations in the environment.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output: Nothing happens, that means the previous action is invalid and you should try more options.\nYou can only hold one object at a time. Before taking a new object, make sure you have placed down any object you are currently holding.\nYou should not assume or anticipate the feedback.\nEven if you have planned multiple steps ahead, you should only execute one action at a time\nDo not proceed with any further exploration or actions until you receive the feedback from the environment after your action.\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>'
    reward_tensor_lst = []
    total = 0
    success = 0
    type_results = {}
    for test_data in tqdm(val_dataloader):
        # breakpoint()
        test_batch = DataProto.from_single_dict(test_data)

        test_batch = test_batch.pop(
            batch_keys=['_dummy'],
            non_tensor_batch_keys=['game_file'],
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
            'max_steps': 30
        }
        # logger.info(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

        # pad to be divisible by dp_size
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, 1)
        test_output_gen_batch_padded = rollout.generate_sequences(test_gen_batch_padded)

        # breakpoint()
        # unpad
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

        # Store generated outputs
        output_ids = test_output_gen_batch.batch['input_ids']
        output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        folder_path = './demo_seen'
        if not os.path.exists(folder_path):
            # 创建文件夹
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, 'outputs.txt')
        

        test_batch = test_batch.union(test_output_gen_batch)

        # evaluate using reward_function
        # reward_tensor = val_reward_fn(test_batch)

        saved_contexts = []
        for gf, context in zip(test_gen_batch_padded.non_tensor_batch['game_file'], output_texts):
            game_type = get_task_type(gf)
            output_content = {"game_file": gf, "traj": context, "success":check_success(context), "game_type": game_type}
            if game_type not in type_results.keys():
                type_results[game_type] = {"success": 0, "total": 0}

            if output_content['success']:
                success += 1
                type_results[game_type]['success'] += 1
            type_results[game_type]['total'] += 1
            total += 1
            with open(file_path, 'a') as f:
                f.write(json.dumps(output_content) + "\n")


    # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
    metric_dict = {}
    metric_dict[f'val/test_score'] = success / total

    for key, value in type_results.items():
        type_results[key] = type_results[key]['success'] / type_results[key]['total']
    
    metric_dict['scores'] = type_results


    # metric_dict[f'val/test_score'] = reward_tensor.mean(-1)

    logger.info(f'metric_dict: {metric_dict}')

if __name__ == "__main__":
    main()
