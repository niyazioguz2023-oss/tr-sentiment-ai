from src.data import load_samples
from src.model import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pathlib

def main():
    texts, labels = load_samples()
    Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.3, random_state=42)
    pipe = build_pipeline()
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"accuracy: {acc:.3f}")
    print(classification_report(yte, pred, zero_division=0))
    out = pathlib.Path(__file__).parent / "model"
    out.mkdir(exist_ok=True)
    import pickle
    (out / "pipeline.pkl").write_bytes(pickle.dumps(pipe))
    print(f"model yazildi: {out / 'pipeline.pkl'}")

if __name__ == "__main__":
    main()
