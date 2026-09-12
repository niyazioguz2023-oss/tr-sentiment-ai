from src.data import load_samples
from src.model import build_pipeline

def test_train_predict_smoke():
    texts, labels = load_samples()
    assert len(texts) >= 10
    pipe = build_pipeline()
    pipe.fit(texts, labels)
    out = pipe.predict(["harika bir deneyimdi", "berbat ve kotuydu"])
    assert len(out) == 2
    assert set(out) <= {"pozitif", "negatif", "notr"}
