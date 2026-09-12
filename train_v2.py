"""v2: winvoker veri kumesi (60k ornek) TF-IDF + LogisticRegression."""
import json, pickle, pathlib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
MAP = {"positive": "pozitif", "negative": "negatif", "neutral": "notr", "Positive": "pozitif", "Negative": "negatif", "Neutral": "notr"}
def main(n_train=60000, n_test=8000, seed=42):
    ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
    print("ham etiketler:", ds["train"].unique("label"))
    tr = ds["train"].shuffle(seed=seed).select(range(min(n_train, len(ds["train"]))))
    te = ds["test"].shuffle(seed=seed).select(range(min(n_test, len(ds["test"]))))
    Xtr = [MAP.get(l, l) for l in tr["label"]]
    Xte = [MAP.get(l, l) for l in te["label"]]
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100000)), ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))])
    pipe.fit(tr["text"], Xtr)
    pred = pipe.predict(te["text"])
    acc = accuracy_score(Xte, pred)
    mf1 = f1_score(Xte, pred, average="macro")
    print("accuracy: %.4f  macro-F1: %.4f" % (acc, mf1))
    print(classification_report(Xte, pred, zero_division=0))
    out = pathlib.Path(__file__).parent / "model"
    out.mkdir(exist_ok=True)
    (out / "pipeline_v2.pkl").write_bytes(pickle.dumps(pipe))
    (out / "metrics_v2.json").write_text(json.dumps({"accuracy": acc, "macro_f1": mf1, "n_train": len(tr), "n_test": len(te), "dataset": "winvoker/turkish-sentiment-analysis-dataset"}, indent=2), encoding="utf-8")
    print("kaydedildi: model/pipeline_v2.pkl")
if __name__ == "__main__":
    main()
