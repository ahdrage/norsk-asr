"""NB-Whisper baseline on the same packed eval sets, normalization and scoring as our model.

    python -m norsk_asr.baseline --model NbAiLab/nb-whisper-large --eval-dir data/eval --out results/nb-whisper-large

Whisper output (cased, punctuated, digits) goes through `normalize`, the same function that
produced the references; "wer_nodigit" is the comparable number (see evaluate.py).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import pipeline

from .data import SR
from .evaluate import load_eval_sets, score
from .text import normalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NbAiLab/nb-whisper-large")
    p.add_argument("--eval-dir", default="data/eval")
    p.add_argument("--sets", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--beams", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cuda = torch.cuda.is_available()
    asr = pipeline(
        "automatic-speech-recognition", model=a.model, device=0 if cuda else -1,
        dtype=torch.float16 if cuda else torch.float32,
    )
    gen = {"task": "transcribe", "language": "no", "num_beams": a.beams}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, ds in load_eval_sets(Path(a.eval_dir), a.sets, a.limit).items():
        audio = [{"raw": ds.audio(i).astype(np.float32) / 32768.0, "sampling_rate": SR} for i in range(len(ds))]
        t0 = time.time()
        preds = asr(audio, batch_size=a.batch_size, chunk_length_s=28, generate_kwargs=gen)
        wall = time.time() - t0
        hyps = [normalize(x["text"]) for x in preds]
        refs = [r["text"] for r in ds.rows]
        res = score(refs, hyps, [r.get("has_digits", False) for r in ds.rows])
        res["speed_x"] = ds.hours() * 3600 / wall
        results[name] = res
        print(name, json.dumps(res), flush=True)
        with open(out / f"{name}.jsonl", "w") as f:
            for r, h, x in zip(ds.rows, hyps, preds):
                f.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": h, "raw": x["text"]}, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"model": a.model, "beams": a.beams, "results": results}, indent=1))


if __name__ == "__main__":
    main()
