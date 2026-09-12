import sys, pickle, pathlib
def main():
    text = " ".join(sys.argv[1:]) or "Bu urun harika, cok memnun kaldim"
    p = pathlib.Path(__file__).parent / "model" / "pipeline.pkl"
    if not p.exists():
        print("Once train.py calistir: model/pipeline.pkl yok")
        sys.exit(1)
    pipe = pickle.loads(p.read_bytes())
    print(f"{text} => {pipe.predict([text])[0]}")

if __name__ == "__main__":
    main()
