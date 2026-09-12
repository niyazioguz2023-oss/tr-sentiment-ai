import json, pathlib
LABELS = ["negatif", "notr", "pozitif"]
def load_samples(path=None):
    p = pathlib.Path(path) if path else pathlib.Path(__file__).resolve().parent.parent / "data" / "samples.jsonl"
    texts, labels = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        texts.append(row["text"])
        labels.append(row["label"])
    return texts, labels
