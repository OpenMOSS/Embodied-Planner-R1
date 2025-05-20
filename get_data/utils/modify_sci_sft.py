#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------
# 新 system prompt ------------------------------------------------------
system_prompt_new = 'You are a helpful assistant to do some scientific experiment in an environment.\nYou should explore the environment and find the items you need to complete the experiment.\n\nIn the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway.\nYou can teleport to any room in one step.\nThe available actions are:\nactivate OBJ\nclose OBJ\nconnect OBJ to OBJ\ndeactivate OBJ\ndisconnect OBJ\ndunk OBJ in OBJ\neat OBJ\nflush OBJ\nfocus on OBJ\ngo LOC\ninventory\nlook around\nlook at OBJ\nlook in OBJ\nmix OBJ\nmove OBJ to OBJ\nopen OBJ\npick up OBJ\npour OBJ in OBJ\nput down OBJ\nread OBJ\nuse OBKJ on OBJ\nteleport to LOC\nwait: wait 10 steps\nwait1: wait 1 step\ntask: check your task\ndone: indicate that you believe the task is complete\nWhen arrive a new location, you should use look around to check the OBj you can interact with.\nUse focus on OBJ only neccessary as incorrect use will cause environment ends.\nDo not proceed with any further exploration or actions until you receive the feedback from the environment after your action.\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>'

# ---------------------------------------------------------------------
# 角色映射 --------------------------------------------------------------
role_mapping = {"human": "user", "gpt": "assistant"}

# ---------------------------------------------------------------------
def transform(input_path: Path, output_path: Path) -> None:
    """执行格式转换并写入 output_path"""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for d in tqdm(data, desc="Processing data", unit="item"):
        # conversations → messages，并跳过前两个
        d["messages"] = d.pop("conversations")[2:]

        for msg in d["messages"]:
            role = msg.pop("from")
            msg["role"] = role_mapping.get(role, role)
            msg["content"] = msg.pop("value")

        # 插入新版 system prompt
        d["messages"].insert(0, {"role": "system", "content": system_prompt_new})

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SciWorld train data to the new message format."
    )
    parser.add_argument(
        "-i", "--input", metavar="FILE", type=Path,
        help="Path to source SciWorld JSON"
    )
    parser.add_argument(
        "-o", "--output", metavar="FILE", type=Path,
        help="Path to destination JSON"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_json = args.input
    output_json = args.output

    transform(input_json, output_json)
    print(f"Success!  {input_json} ➜ {output_json}")