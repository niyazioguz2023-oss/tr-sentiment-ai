"""v3: tam veri (440k) TF-IDF + LinearSVC."""
import json, pickle, pathlib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
MAP = {"Positive": "pozitif", "Negative": "negatif", "Notr": "Notr"}
def main():
    ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset")
    tr, te = ds["train"], ds["test"]
    Xtr = [MAP.get(l, l) for l in tr["label"]]
    Xte = [MAP.get(l, l) for l in te["label"]]
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=150000, sublinear_tf=True)), ("clf", LinearSVC(C=1.0))])
    pipe.fit(tr["text"], Xtr)
    pred = pipe.predict(te["text"])
    acc = accuracy_score(Xte, pred)
    mf1 = f1_score(Xte, pred, average="macro")
    print("accuracy: %.4f  macro-F1: %.4f" % (acc, mf1))
    print(classification_report(Xte, pred, zero_division=0))
    out = pathlib.Path(__file__).parent / "model"
    (out / "pipeline_v3.pkl").write_bytes(pickle.dumps(pipe))
    (out / "metrics_v3.json").write_text(json.dumps({"accuracy": acc, "macro_f1": mf1, "n_train": len(tr), "n_test": len(te)}, indent=2), encoding="utf-8")
    print("kaydedildi: model/pipeline_v3.pkl")
if __name__ == "__main__":
    main()
