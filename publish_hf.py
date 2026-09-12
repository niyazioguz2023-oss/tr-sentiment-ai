import os, pathlib
from huggingface_hub import HfApi
tok = os.environ.get("HF_TOKEN")
if not tok:
    envp = os.path.join(os.environ.get("USERPROFILE", r"C:\Users\Os"), "AppData", "Local", "hermes", ".env")
    for line in open(envp, encoding="utf-8"):
        if line.startswith("HF_TOKEN="):
            tok = line.strip().split("=", 1)[1]
api = HfApi(token=tok)
me = api.whoami()
print("HF kullanici:", me.get("name"))
repo = f"{me.get('name')}/tr-sentiment-mini"
api.create_repo(repo, repo_type="model", exist_ok=True)
root = pathlib.Path(__file__).parent
api.upload_file(path_or_fileobj=str(root/"model"/"pipeline.pkl"), path_in_repo="pipeline.pkl", repo_id=repo)
card = """---
language: tr
tags: [sentiment-analysis, turkish, scikit-learn]
---
# tr-sentiment-mini
Kucuk Turkce duygu analizi modeli (TF-IDF + LogisticRegression). `predict.py` ile kullanilir.
Etiketler: pozitif / negatif / notr.
"""
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
print("YAYIN TAMAM: https://huggingface.co/" + repo)
