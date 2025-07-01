"""
To convert model checkpoint trained from verl into transformers type checkpoint.
"""

import argparse
import re
import os
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoConfig

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Load model and optionally FSDP checkpoint, then save it.")
    parser.add_argument("--model", required=True, help="Path to the HuggingFace-style pretrained model.") # "/path/to/Qwen2.5-7B-Instruct"
    parser.add_argument("--fsdp", help="Path to the FSDP checkpoint directory.") # "/path/to/ckpt/actor"
    parser.add_argument("--output", required=True, help="Path to save the processed model and tokenizer.")
    return parser.parse_args()


def load_sharded_model(fsdp_checkpoint_path):
    """加载分片模型（FSDP 检查点）"""
    state_dict = defaultdict(list)
    checkpoint_dir = Path(fsdp_checkpoint_path)

    # 找到所有分片文件
    shard_files = list(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise ValueError(f"No checkpoint files found in {fsdp_checkpoint_path}")

    # 检查 world_size 是否一致
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

    # 加载每个分片文件
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

    # 合并分片到完整的 state_dict
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
    """初始化模型和分词器"""
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


def ensure_output_directory(output_path):
    """确保输出文件夹存在，如果不存在则创建"""
    output_dir = Path(output_path)
    if not output_dir.exists():
        print(f"Output directory {output_path} does not exist. Creating it.")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Output directory {output_path} already exists.")


def main():
    # 解析命令行参数
    args = parse_args()

    # 确保输出文件夹存在
    ensure_output_directory(args.output)

    # 加载 HuggingFace 模型
    print(f"Loading HuggingFace model from: {args.model}")
    tokenizer, actor_module = initialize_model_and_tokenizer(args.model)

    # 如果提供了 FSDP 检查点路径，则加载检查点
    if args.fsdp:
        print(f"Loading FSDP checkpoint from: {args.fsdp}")
        state_dict = load_sharded_model(args.fsdp)
        actor_module.load_state_dict(state_dict)

    # 保存模型和分词器
    print(f"Saving model and tokenizer to: {args.output}")
    actor_module.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()