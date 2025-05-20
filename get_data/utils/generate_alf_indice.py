import os
import json
import re
from pathlib import Path
import sys
import argparse
def extract_dataset(valid_unseen_path):
    """
    从valid_unseen目录中提取数据集，其中：
    - 'type'由父文件夹前缀确定（例如'look_at_obj_in_light'）
    - 'game_file'是每个子文件夹中'game.tw-pddl'的路径
    
    参数:
        valid_unseen_path (str): 'valid_unseen'目录的路径
        
    返回:
        list: 包含'type'和'game_file'键的字典列表
    """
    dataset = []
    
    # 映射前缀到6种类型之一
    type_mapping = {
        "pick_and_place_simple": 1,
        "look_at_obj_in_light": 2,
        "pick_clean_then_place_in_recep": 3,
        "pick_heat_then_place_in_recep": 4,
        "pick_cool_then_place_in_recep": 5,
        "pick_two_obj_and_place": 6
    }
    
    # 遍历valid_unseen目录
    for root, dirs, files in os.walk(valid_unseen_path):
        # 跳过根目录本身
        if root == valid_unseen_path:
            continue
        
        # 如果当前目录包含'game.tw-pddl'
        if 'game.tw-pddl' in files:
            # 存储game.tw-pddl文件的路径作为game_file
            game_file_path = os.path.join(root, 'game.tw-pddl')
            
            # 获取game.tw-pddl文件的父级的父级文件夹
            # game.tw-pddl所在路径: valid_unseen/look_at_obj_in_light-XXX/trial_XXXXX/game.tw-pddl
            current_path = Path(root)
            parent_path = current_path.parent
            
            # 如果父级就是valid_unseen，那么使用当前目录的父级作为参考
            if parent_path.name == Path(valid_unseen_path).name:
                parent_dir_name = current_path.name
            else:
                # 否则使用父级目录的名称
                parent_dir_name = parent_path.name
            
            # 提取前缀（连字符前的部分）
            prefix = ""
            if '-' in parent_dir_name:
                prefix = parent_dir_name.split('-')[0]
            else:
                prefix = parent_dir_name
            
            # 确定类型
            type_name = None
            for pattern, type_value in type_mapping.items():
                if prefix.startswith(pattern):
                    type_name = type_value
                    break
            
            # 如果没有找到匹配的类型，则跳过此文件（不添加到数据集）
            if type_name is None:
                print(f"跳过: 未能识别前缀 '{prefix}' 的类型，对应文件: {game_file_path}")
                continue
            
            # 添加到数据集
            dataset.append({
                'type': type_name,
                'game_file': game_file_path
            })
    
    return dataset

def save_dataset(dataset, output_path):
    """
    将数据集保存到JSON文件
    
    参数:
        dataset (list): 包含'type'和'game_file'键的字典列表
        output_path (str): 保存输出JSON文件的路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"数据集已保存到 {output_path}，共 {len(dataset)} 个条目")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert valid_unseen data")
    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="PATH",
        help="Path to valid_unseen directory or file",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Path to output JSON file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    # 把字符串转成 Path，方便后续操作
    VALID_UNSEEN_PATH = Path(args.input).expanduser().resolve()
    OUTPUT_PATH       = Path(args.output).expanduser().resolve()
    
    print(f"使用数据目录: {VALID_UNSEEN_PATH}")
    
    # 提取并保存数据集
    dataset = extract_dataset(VALID_UNSEEN_PATH)
    save_dataset(dataset, OUTPUT_PATH)
    
    # 打印统计信息
    types = {}
    for item in dataset:
        type_name = item['type']
        if type_name in types:
            types[type_name] += 1
        else:
            types[type_name] = 1
    
    print("\n数据集统计信息:")
    print(f"总条目数: {len(dataset)}")
    print("按类型划分的条目:")
    for type_name, count in sorted(types.items(), key=lambda x: (0 if isinstance(x[0], int) else 1, x[0])):
        type_desc = {
            1: "pick_and_place_simple",
            2: "look_at_obj_in_light",
            3: "pick_clean_then_place_in_recep",
            4: "pick_heat_then_place_in_recep",
            5: "pick_cool_then_place_in_recep",
            6: "pick_two_obj_and_place"
        }.get(type_name, "未知类型")
        
        print(f"  类型 {type_name} ({type_desc}): {count}个条目")