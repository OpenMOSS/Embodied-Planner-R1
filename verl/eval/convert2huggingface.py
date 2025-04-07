import hydra
import numpy as np
import os
import re
import torch
import torch.distributed
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig
from collections import defaultdict

from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn, JSONDataset
from verl.workers.reward_manager import AlfRewardManager
# from verl.workers.rollout.hf_rollout import HFRollout
from verl.workers.rollout.vllm_rollout.alf_rollout import AlfRollout

def load_sharded_model(fsdp_checkpoint_path):
    state_dict = defaultdict(list)
    checkpoint_dir = Path(fsdp_checkpoint_path)

    shard_files = list(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise ValueError(f"No checkpoint files found in {fsdp_checkpoint_path}")

    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")
    world_sizes = set()
    for file in shard_files:
        match = pattern.match(file.name)
        if match:
            world_sizes.add(int(match.group(1)))

    if len(world_sizes) != 1:
        raise ValueError(
            f"Inconsistent world_size found in checkpoint files: {world_sizes}"
        )

    world_size = world_sizes.pop()
    print(f"Found checkpoints with world_size = {world_size}")

    for rank in range(world_size):
        filepath = checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not filepath.exists():
            raise ValueError(f"Missing shard file: {filepath}")

        print(f"Loading shard: {filepath}")
        shard_dict = torch.load(filepath)

        for key, value in shard_dict.items():
            if hasattr(value, "to_local"):
                value = value.to_local()
            state_dict[key].append(value)

    consolidated_state_dict = {}
    for key in state_dict:
        try:
            consolidated_state_dict[key] = torch.cat(state_dict[key], dim=0)
        except (RuntimeError, TypeError):
            consolidated_state_dict[key] = state_dict[key][0]
            print(
                f"Parameter '{key}' does not need concatenation, using first shard value"
            )

    return consolidated_state_dict


def initialize_model_and_tokenizer(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
):
    local_path = copy_local_path_from_hdfs(model_path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    actor_model_config = AutoConfig.from_pretrained(
        local_path, trust_remote_code=trust_remote_code
    )
    actor_module = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch_dtype,
        config=actor_model_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )

    return tokenizer, actor_module


def main():
    # Loading huggingface-style checkpoint
    model_path = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/data/Qwen/Qwen2.5-7B-Instruct"
    # model_path = "Qwen/Qwen2.5-3B"
    # model_path = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer, actor_module = initialize_model_and_tokenizer(model_path)

    # Loading FSDP checkpoint (optional: these three lines can be skipped. Prerequisite: actor_module must be preloaded)
    fsdp_checkpoint_path = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/verl/outputs_v2/20250404/alf_v2/20250404_1749/rank_0/ckpt/global_step_50/actor"
    state_dict = load_sharded_model(fsdp_checkpoint_path)
    actor_module.load_state_dict(state_dict)

    actor_module.save_pretrained('/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/verl/eval/test')
    tokenizer.save_pretrained('/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/verl/eval/test')

if __name__ == "__main__":
    main()
