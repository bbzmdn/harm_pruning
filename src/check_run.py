"""Summarize a prune.py run from its tee'd log (+ the offline wandb run it points to).

Usage (from harm_pruning/src):
    ../.venv/bin/python check_run.py base14b_rerun.log prune14b_p1e-4_q2e-5_rerun.log

Reports per run: total pruned weights, sparsity, wikitext perplexity, TriviaQA accuracy,
and how many TriviaQA answers contain a corrupted year (5-digit 19xxx/20xxx, e.g. "19942").
"""
import glob
import json
import re
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore


def wandb_metrics(run_id):
    files = glob.glob(f"wandb/{run_id}/*.wandb")
    if not files:
        return {}
    ds = DataStore()
    ds.open_for_scan(files[0])
    out = {}
    while (data := ds.scan_data()) is not None:
        rec = pb.Record()
        rec.ParseFromString(data)
        kind = rec.WhichOneof("record_type")
        items = rec.summary.update if kind == "summary" else rec.history.item if kind == "history" else []
        for it in items:
            if it.key in ("triviaqa_acc", "wikitest_ppl", "sparsity_ratio"):
                out[it.key] = float(json.loads(it.value_json))
    return out


def check(log_path):
    text = open(log_path, errors="replace").read()
    run = re.search(r"offline-run-[0-9_]+-[a-z0-9]+", text)
    m = wandb_metrics(run.group(0)) if run else {}
    ppl = re.findall(r"wikitext perplexity ([\d.]+)", text)
    answers = re.findall(r"Q: .*? A: (.*)", text)
    corrupted = sum(bool(re.search(r"\b(19|20)\d{3}\b", a)) for a in answers)
    return dict(
        log=log_path,
        pruned_weights=sum(int(x) for x in re.findall(r"n_pruned_weights: (\d+)", text)),
        sparsity=m.get("sparsity_ratio"),
        wikitext_ppl=float(ppl[0]) if ppl else None,
        triviaqa_acc=m.get("triviaqa_acc"),
        triviaqa_answers=len(answers),
        corrupted_year_answers=corrupted,
        wandb_run=run.group(0) if run else None,
    )


if __name__ == "__main__":
    results = [check(p) for p in sys.argv[1:]]
    for r in results:
        print("\n".join(f"{k:>24}: {v}" for k, v in r.items()), end="\n\n")
    if len(results) >= 2 and results[0]["triviaqa_acc"] and results[1]["triviaqa_acc"]:
        base, pruned = results[0], results[1]
        drop = 100 * (base["triviaqa_acc"] - pruned["triviaqa_acc"]) / base["triviaqa_acc"]
        print(f"TriviaQA relative drop: {drop:.1f}%  (paper's rule: <= 10%)  -> {'OK' if drop <= 10 else 'TOO DAMAGED'}")
        print(f"Perplexity: {base['wikitext_ppl']:.2f} -> {pruned['wikitext_ppl']:.2f}")
