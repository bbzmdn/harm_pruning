#!/bin/bash
set -e
cd /workspace/harm_pruning

python3 "$(dirname "$0")/fix_harm_pruning.py" /workspace/harm_pruning/src

uv pip install --python .venv/bin/python --reinstall \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python hf_transfer

sed -i 's/+cu126/+cu128/; /^nvidia-/d' requirements.txt
grep -q "download.pytorch.org/whl/cu128" requirements.txt || sed -i '1i --extra-index-url https://download.pytorch.org/whl/cu128' requirements.txt

.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list()); print(torch.ones(1).cuda() + 1)"
