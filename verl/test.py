import json
from torch.utils.data import Dataset
# from torchdata.stateful_dataloader import StatefulDataLoader
# class JSONDataset(Dataset):
#     """
#     一个能够适应 JSON 文件（数组格式或 JSON Lines 格式）的通用 PyTorch Dataset。
#     """
#     def __init__(self, json_file, transform=None):
#         """
#         Args:
#             json_file (str): JSON 文件路径。
#             transform (callable, optional): 数据转换函数，应用于每一条样本。
#         """
#         self.data = []
#         self.transform = transform

#         # 检测文件格式并加载数据
#         with open(json_file, "r", encoding="utf-8") as f:
#             first_line = f.readline().strip()
#             if first_line.startswith("["):
#                 # JSON 数组格式
#                 f.seek(0)  # 重置文件指针到开头
#                 self.data = json.load(f)
#             else:
#                 # JSON Lines 格式
#                 f.seek(0)  # 重置文件指针到开头
#                 for line in f:
#                     if line.strip():  # 跳过空行
#                         self.data.append(json.loads(line.strip()))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         if self.transform:
#             item = self.transform(item)
#         return item


# dataset = JSONDataset('/inspire/ssd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zyfei/open-embodied-r1/data/alfworld-data/train_data.json')

# print(type(dataset))

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/data/Qwen/Qwen2.5-7B-Instruct')

# print(tokenizer)


import requests
import sys
sys.path.append("/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/verl_mod/verl")
from alfworld_server.alfworld_server_lite.tw_env import get_tw_env
url="http://localhost:8000"

# res = requests.get(url + "/health")
game_file = ['/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/qiuxipeng-24028/xpqiu/lji/open-embodied-r1/alfworld_server/data/json_2.1.1/valid_seen/pick_cool_then_place_in_recep-Pan-None-DiningTable-7/trial_T20190908_232648_241836/game.tw-pddl']
# res = requests.post(url + "/reset", json={'game_file': game_file, 'batch_size': 2}) 
# print(res)

env = get_tw_env(game_file=game_file, batch_size=2)
print(env.reset())