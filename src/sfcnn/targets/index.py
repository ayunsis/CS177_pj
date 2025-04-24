import json

# Load the JSON file
json_path = r'd:\Bertram Rowen\texts\code\CS177\CS177_pj\src\sfcnn\targets\target3.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Add index to each protein entry (starting from 1)
for idx, pdb_id in enumerate(data.keys(), start=1):
    data[pdb_id]['index'] = idx

# Save back to the original JSON file
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)