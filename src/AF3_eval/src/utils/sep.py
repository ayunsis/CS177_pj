import json

with open('src/sfcnn/targets/coreset_smiles_chains.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

keys = list(data.keys())
chunks = [keys[i:i+95] for i in range(0, len(keys), 95)]

for idx, chunk in enumerate(chunks, 1):
    split_data = {k: data[k] for k in chunk}
    with open(f'target{idx}.json', 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2)

print(f"Split into {len(chunks)} files.")