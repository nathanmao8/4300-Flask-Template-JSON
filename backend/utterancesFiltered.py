import json
with open("backend/workoutUtterances.jsonl", 'r') as file:
    reader = [json.loads(line) for line in file]
    new_data = []
    for row in reader:
        text  = row["text"] if row["text"] else None
        if text and len(text) > 30: #filtered out comments that were too small
            new_data.append({"text": text})

with open("backend/utterancesFiltered.json", 'w', encoding='utf-8') as jsonfile:
    json.dump(new_data, jsonfile, indent=1, ensure_ascii=False)


