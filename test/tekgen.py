import csv
import json

data = []
json_data = []

with open('../data/TekGen/quadruples-test.tsv', 'r', encoding='utf-8', newline='') as f:
    tr = csv.reader(f, delimiter='\t')
    for idx, row in enumerate(tr):
        print("\r {}".format(idx), end='')
        data.append(row)
        json_data.append(json.loads(data[-1][0]))
    print()

print(len(json_data))
json_data = json_data[:int(len(json_data) * 0.8)]
print(json.loads(data[2][0]))

json_data = {
    "version": "0.0.1",
    "data": json_data,
}
with open('../data/TekGen/quadruples-validation.json', 'w') as json_file:
    json.dump(json_data, json_file)

with open('../data/TekGen/quadruples-validation.json', 'r') as json_file:
    json_save_data = json.load(json_file)

print(type(json_save_data))