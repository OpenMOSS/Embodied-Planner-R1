import json

# 读取源文件 'train.json'
origin_path = '/inspire/ssd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zyfei/open-embodied-r1/data/alfworld-data/train_data.json'
with open(origin_path, 'r') as infile:
    data = [json.loads(line) for line in infile]

# 筛选出包含 'pick_and_place_simple' 的项
simple_tasks = [entry for entry in data if 'pick_and_place_simple' in entry['game_file']]

# 筛选出不包含 'pick_and_place_simple' 的项（即 harder tasks）
harder_tasks = [entry for entry in data if 'pick_and_place_simple' not in entry['game_file']]

# 将 harder tasks 写入新文件 'harder_task.json'
with open('./harder_task.json', 'w') as outfile:
    for task in harder_tasks:
        json.dump(task, outfile)
        outfile.write('\n')

print(f"已将 {len(harder_tasks)} 项任务保存到 'harder_task.json' 文件中。")
