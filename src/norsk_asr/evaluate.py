"""Greedy CTC decoding and WER on the packed eval sets.

    python -m norsk_asr.evaluate --ckpt runs/full_top1 --eval-dir data/eval    # or a .pt checkpoint

Every set reports WER on all utterances and on the digit-free subset ("wer_nodigit"). The
digit-free number is the fair one: our model speaks numbers as words ("tjue tjuefem") while
references are expanded mechanically ("to tusen og tjuefem").
"""

import argparse
import json
import time
from pathlib import Path

import jiwer
import numpy as np
import torch

from .data import SR, PackedSet, bucket_batches, collate
from .units import BLANK, Units


def greedy(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    best = log_probs.argmax(-1).cpu()
    out = []
    for seq, n in zip(best, lengths.cpu()):
        seq = torch.unique_consecutive(seq[:n])
        out.append([int(u) for u in seq if u != BLANK])
    return out


def score(refs: list[str], hyps: list[str], digits: list[bool]) -> dict:
    res = {"utts": len(refs), "wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}
    nd = [(r, h) for r, h, d in zip(refs, hyps, digits) if not d]
    if nd and len(nd) < len(refs):
        res["wer_nodigit"] = jiwer.wer([r for r, _ in nd], [h for _, h in nd])
    return res


@torch.no_grad()
def evaluate_set(
    model, units: Units, ds: PackedSet, device, max_batch_seconds: float = 600, dump: dict | None = None
) -> tuple[dict, list[dict]]:
    """Greedy decode one set. If `dump` is a dict, per-utterance log-probs [T, V] (float16) go into it."""
    model.eval()
    hyps = {}
    t0 = time.time()
    for idx in bucket_batches(ds, max_batch_seconds, seed=0, shuffle=False):
        b = collate(ds, idx)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lp, olens = model(b["wav"].to(device, non_blocking=True), b["lengths"].to(device))
        for i, ids in zip(idx, greedy(lp, olens)):
            hyps[i] = units.decode(ids)
        if dump is not None:
            for j, i in enumerate(idx):
                dump[ds.rows[i]["id"]] = lp[j, : olens[j]].half().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    order = range(len(ds))
    refs = [ds.rows[i]["text"] for i in order]
    hyp = [hyps[i] for i in order]
    res = score(refs, hyp, [ds.rows[i].get("has_digits", False) for i in order])
    res["speed_x"] = ds.hours() * 3600 / wall
    model.train()
    return res, [
        {"id": ds.rows[i]["id"], "ref": refs[i], "hyp": hyp[i], "has_digits": ds.rows[i].get("has_digits", False)}
        for i in order
    ]


def load_eval_sets(eval_dir: Path, names: list[str] | None, limit: int | None) -> dict[str, PackedSet]:
    sets = {}
    for p in sorted(Path(eval_dir).glob("*.jsonl")):
        if names and p.stem not in names:
            continue
        sets[p.stem] = PackedSet([p], limit=limit)
    return sets


def main():
    from .train import load_checkpoint

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval-dir", default="data/eval")
    p.add_argument("--sets", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--out")
    p.add_argument("--dump-logprobs", action="store_true", help="also save <out>/<set>.logprobs.npz for LM decoding")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, units = load_checkpoint(a.ckpt, device)
    results = {}
    for name, ds in load_eval_sets(a.eval_dir, a.sets, a.limit).items():
        dump = {} if a.dump_logprobs else None
        res, rows = evaluate_set(model, units, ds, device, dump=dump)
        results[name] = res
        print(name, json.dumps(res), flush=True)
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            with open(Path(a.out) / f"{name}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if dump is not None:
                np.savez(Path(a.out) / f"{name}.logprobs.npz", **dump)
    if a.out:
        (Path(a.out) / "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
