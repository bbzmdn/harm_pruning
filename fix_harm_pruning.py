import re
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/harm_pruning/src")

prune = src / "prune.py"
s = prune.read_text()
s = s.replace(", prune_attribution_score_set_difference_with_refusal", "")
prune.write_text(s)
print("prune.py import:", "ok" if "with_refusal" not in s else "STILL BROKEN")

data_utils = src / "data_utils.py"
s = data_utils.read_text()
if "_Qwen2.5-7B-Instruct_selfgen.csv" not in s:
    old = '    elif "Qwen2.5-14B-Instruct" in model_name:\n        if "selfgen" in data_name:'
    new = ('    elif "Qwen2.5-7B-Instruct" in model_name:\n'
           '        if "selfgen" in data_name:\n'
           '            data_files = {\n'
           '                "train": f"../data/em_prune_data/{data_name.replace(\'_selfgen\', \'\')}_Qwen2.5-7B-Instruct_selfgen.csv"}\n'
           '        else:\n'
           '            data_files = {"train": f"../data/em_prune_data/{data_name}_Qwen2.5.csv"}\n') + old
    if s.count(old) != 1:
        sys.exit("data_utils.py: could not find the Qwen2.5-14B EM branch to insert before")
    s = s.replace(old, new)
    data_utils.write_text(s)
print("data_utils.py 7B branch:", "ok" if "_Qwen2.5-7B-Instruct_selfgen.csv" in data_utils.read_text() else "MISSING")

prune_utils = src / "prune_utils.py"
s = prune_utils.read_text()
if '"Qwen/Qwen2.5-7B-Instruct": 28' not in s:
    s = s.replace('    "Qwen/Qwen2.5-14B-Instruct": 48,\n', '    "Qwen/Qwen2.5-14B-Instruct": 48,\n    "Qwen/Qwen2.5-7B-Instruct": 28,\n', 1)
modules = re.search(r'    "Qwen/Qwen2.5-14B-Instruct": (\[[^\]]*\])', s)
if modules and '"Qwen/Qwen2.5-7B-Instruct": [' not in s:
    line = modules.group(0)
    s = s.replace(line + "\n}", line + ",\n    \"Qwen/Qwen2.5-7B-Instruct\": " + modules.group(1) + "\n}", 1)
prune_utils.write_text(s)
s = prune_utils.read_text()
print("prune_utils.py 7B entries:", "ok" if '"Qwen/Qwen2.5-7B-Instruct": 28' in s and '"Qwen/Qwen2.5-7B-Instruct": [' in s else "MISSING")
