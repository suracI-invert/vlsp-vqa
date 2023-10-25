from src.utils.evaluation import evaluate
import json

with open('./data/results/public_results.json', 'r', encoding= 'utf8') as f:
    file = json.load(f)
print(len(file))
with open('./data/public_results.json', 'w') as f:
    json.dump(file, f)