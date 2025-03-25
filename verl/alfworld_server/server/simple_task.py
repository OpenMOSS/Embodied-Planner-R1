import json

# 读取源文件 'train.json'
origin_path = '/inspire/ssd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zyfei/open-embodied-r1/data/alfworld-data/train_data.json'
with open(origin_path, 'r') as infile:
    data = [json.loads(line) for line in infile]

# 筛选出包含 'pick_and_place_simple' 的项
simple_tasks = [entry for entry in data if 'pick_and_place_simple' in entry['game_file']]

# 将筛选结果写入新文件 'simple_task.json'
with open('./simple_task.json', 'w') as outfile:
    for task in simple_tasks:
        json.dump(task, outfile)
        outfile.write('\n')

print(f"已将 {len(simple_tasks)} 项任务保存到 'simple_task.json' 文件中。")

