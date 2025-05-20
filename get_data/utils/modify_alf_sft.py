#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm

# ----------------------------------------------------------------------
# 新旧 prompt ------------------------------------------------------------------
system_prompt_new = 'You are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:Thought: your thoughts.\nAction: your next action.\n\nThe available actions are:\n1. `go to (receptacle)`\n2. `open (receptacle)`\n3. `close (receptacle)`\n4. `take (object) from (receptacle)`\n5. `move (object) to (receptacle)`\n6. `examine (something) with (object)`\n7. `use (object)`\n8. `heat (object) with (receptacle)`\n9. `clean (object) with (receptacle)`\n10. `cool (object) with (receptacle)`\n11. `slice (object) with (object)` - slice an object using a sharp object\n12. `look` - look around your current location\n13. `inventory` - check your current inventory\n14. `done` - Indicate that you believe the task is complete\nWhere `(object)` refers to manipulable objects and `(receptacle)` refers to receptacles or locations in the environment.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output: Nothing happens, that means the previous action is invalid and you should try more options.\nYou can only hold one object at a time. Before taking a new object, make sure you have placed down any object you are currently holding.\nYou should not assume or anticipate the feedback.\nEven if you have planned multiple steps ahead, you should only execute one action at a time\nDo not proceed with any further exploration or actions until you receive the feedback from the environment after your action.\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>'

# ----------------------------------------------------------------------
# 预编译正则 -------------------------------------------------------------------
pattern_move = re.compile(r"put (.*?) in/on (.*)", re.IGNORECASE)         # put → move

role_mapping = {"human": "user", "gpt": "assistant"}                      # 角色映射

# ----------------------------------------------------------------------
# 主转换函数 -------------------------------------------------------------------
def transform(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for d in tqdm(data, desc="Processing data", unit="item"):
        # conversations → messages，并跳过前两条
        d["messages"] = d.pop("conversations")[2:]

        for msg in d["messages"]:
            # put → move
            msg["value"] = pattern_move.sub(r"move \1 to \2", msg["value"])

            # 字段改名 + 角色转换
            role = msg.pop("from")
            msg["role"] = role_mapping.get(role, role)
            msg["content"] = msg.pop("value")

        # 在最前面插入 system prompt
        d["messages"].insert(0, {"role": "system", "content": system_prompt_new})

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ----------------------------------------------------------------------
# 命令行解析 -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Alfworld train data to the new message format."
    )
    parser.add_argument(
        "-i", "--input", metavar="FILE", type=Path,
        help="Path to source Alfworld JSON"
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
    print(f"✅  Success!  {input_json}  ➜  {output_json}")