@echo off
cd /d %~dp0
uv pip install --python .venv-gpu\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-gpu\Scripts\python.exe transformers datasets scikit-learn
.venv-gpu\Scripts\python.exe -c "import torch; print('torch',torch.__version__,'| cuda:',torch.cuda.is_available())"
pause
