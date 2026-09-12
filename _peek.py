
from datasets import load_dataset
ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
print(ds)
for s in ds:
    print(s, len(ds[s]), ds[s].features)
    print(ds[s][:2])
    break
