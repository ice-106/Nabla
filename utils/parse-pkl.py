import pickle

file2 = '/Extraction/Nabla/utils/extraction/OSX/demo/results/trimmed_2025_03_31_batch_2_C0003/pkl/000001_0_body_data.pkl'

with open(file2, 'rb') as f:
    d2 = pickle.load(f)

for k, v in d2.items():
    print(f"Key: {k}, Type: {type(v)}")
    if isinstance(v, dict):
        print(f"  Nested keys: {list(v.keys())}")