import json


json_path = r'src/sfcnn/targets/target3.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


for idx, pdb_id in enumerate(data.keys(), start=1):
    data[pdb_id]['index'] = idx

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)