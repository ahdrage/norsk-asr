"""Word n-gram language model for CTC beam search (KenLM + pyctcdecode).

    python -m norsk_asr.lm corpus --out data/lm/ncc.txt --shards 4         # open text from NCC
    python -m norsk_asr.lm build --text data/lm/ncc.txt data/lm_text --eval-dirs data/eval data/dev \
        --out data/lm/no4                                                 # decontaminate + lmplz
    python -m norsk_asr.lm tune --lm data/lm/no4 --logprobs results/x/dev  # alpha/beta grid on dev sets
    python -m norsk_asr.lm decode --lm data/lm/no4 --logprobs results/x --alpha .5 --beta 1

Text is normalized exactly like the ASR targets (lowercase, numbers as words, no punctuation).
Any LM sentence that shares an 8-gram with an eval or dev reference is dropped, so the LM never
sees test text. Decoding parameters are tuned on dev sets only.

Needs the `lm` extra (`uv sync --extra lm`) and KenLM's `lmplz` and `build_binary` on PATH.
"""

import argparse
import collections
import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .text import is_clean, normalize

NCC_REPO = "NbAiLab/NCC"
# Bokmål text close to spoken, modern Norwegian. OCR'd newspapers and books are left out.
NCC_TYPES = re.compile(r"^(parliament|government_nb|publicreports|wikipedia_download_nbo|maalfrid_.*)$")
NGRAM = 8
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÆØÅ0-9«\"])|\n+")


def _ncc_chunk(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        r = json.loads(line)
        if r.get("lang_fasttext") != "no" or not NCC_TYPES.match(r.get("doc_type") or ""):
            continue
        if (r.get("publish_year") or 0) < 1990:
            continue
        for sent in SENT_SPLIT.split(r.get("text") or ""):
            t = normalize(sent)
            if len(t.split()) >= 3 and is_clean(t):
                out.append(t)
    return out


def cmd_corpus(a):
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(NCC_REPO, repo_type="dataset") if f.startswith("data/train-"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w") as out, ProcessPoolExecutor(a.workers) as ex:
        for f in files[: a.shards]:
            path = hf_hub_download(NCC_REPO, f, repo_type="dataset")
            chunks, chunk = [], []
            with open(path) as fh:
                for line in fh:
                    chunk.append(line)
                    if len(chunk) == 2000:
                        chunks.append(chunk)
                        chunk = []
            chunks.append(chunk)
            for sents in ex.map(_ncc_chunk, chunks):
                out.writelines(s + "\n" for s in sents)
                n += len(sents)
            print(f"{f}: {n:,} sentences so far", flush=True)


def eval_ngrams(eval_dirs: list[str]) -> set[tuple[str, ...]]:
    grams = set()
    for d in eval_dirs:
        for p in Path(d).glob("*.jsonl"):
            for line in open(p):
                w = json.loads(line)["text"].split()
                grams.update(tuple(w[i : i + NGRAM]) for i in range(len(w) - NGRAM + 1))
                if len(w) < NGRAM and len(w) >= 4:
                    grams.add(tuple(w))  # short references: drop exact matches
    return grams


def text_files(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out += sorted(p.glob("*.txt")) if p.is_dir() else [p]
    return out


def cmd_build(a):
    grams = eval_ngrams(a.eval_dirs)
    print(f"{len(grams):,} eval n-grams to exclude", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = out.with_suffix(".txt")
    counts = collections.Counter()
    kept = dropped = dup = 0
    seen = set()
    with open(clean, "w") as fo:
        for p in text_files(a.text):
            for line in open(p):
                w = line.split()
                if not w:
                    continue
                key = hash(line)
                if key in seen:  # NST read prompts and boilerplate repeat a lot
                    dup += 1
                    continue
                seen.add(key)
                if (len(w) < NGRAM and tuple(w) in grams) or any(
                    tuple(w[i : i + NGRAM]) in grams for i in range(len(w) - NGRAM + 1)
                ):
                    dropped += 1
                    continue
                fo.write(line)
                counts.update(w)
                kept += 1
    print(f"kept {kept:,} sentences, dropped {dropped:,} overlapping eval text, {dup:,} duplicates", flush=True)
    vocab = [w for w, c in counts.most_common() if c >= a.min_word_count]
    Path(f"{out}.unigrams.txt").write_text("\n".join(vocab) + "\n")
    print(f"{len(vocab):,} unigrams (count >= {a.min_word_count})", flush=True)
    arpa = f"{out}.arpa"
    prune = ["--prune", "0", "0", "1", "1"][: 1 + a.order]
    subprocess.run(
        ["lmplz", "-o", str(a.order), "-S", "40%", "-T", str(out.parent), *prune, "--text", str(clean), "--arpa", arpa],
        check=True,
    )
    subprocess.run(["build_binary", "-q", "8", "-b", "7", "trie", arpa, f"{out}.bin"], check=True)
    os.remove(arpa)
    print("wrote", f"{out}.bin", flush=True)


def make_decoder(lm: str, labels: list[str], alpha: float, beta: float):
    from pyctcdecode import build_ctcdecoder

    unigrams = Path(f"{lm}.unigrams.txt").read_text().split()
    return build_ctcdecoder(labels, kenlm_model_path=f"{lm}.bin", unigrams=unigrams, alpha=alpha, beta=beta)


def ctc_labels(units) -> list[str]:
    # pyctcdecode: "" is the blank, a leading "▁" starts a word (BPE-style).
    return ["", *units.pieces[1:]]


def load_set(logprobs_dir: Path, name: str) -> tuple[list[np.ndarray], list[str], list[bool]]:
    lps = np.load(logprobs_dir / f"{name}.logprobs.npz")
    rows = [json.loads(line) for line in open(logprobs_dir / f"{name}.jsonl")]
    return [lps[r["id"]].astype(np.float32) for r in rows], [r["ref"] for r in rows], [r.get("has_digits", False) for r in rows]


def decode_sets(decoder, sets: dict, beam: int, workers: int) -> dict[str, list[str]]:
    from multiprocessing import get_context

    out = {}
    with get_context("fork").Pool(workers) as pool:
        for name, (lps, _, _) in sets.items():
            out[name] = decoder.decode_batch(pool, lps, beam_width=beam)
    return out


def load_sets(logprobs_dir: str, names: list[str] | None) -> dict:
    d = Path(logprobs_dir)
    found = sorted(p.name.removesuffix(".logprobs.npz") for p in d.glob("*.logprobs.npz"))
    return {n: load_set(d, n) for n in found if not names or n in names}


def scores(sets: dict, hyps: dict) -> dict:
    from .evaluate import score

    return {n: score(sets[n][1], hyps[n], sets[n][2]) for n in sets}


def avg_wer(res: dict) -> float:
    return float(np.mean([r.get("wer_nodigit", r["wer"]) for r in res.values()]))


def cmd_tune(a):
    from .units import Units

    units = Units.load(a.units)
    sets = load_sets(a.logprobs, a.sets)
    print("sets:", {n: len(s[0]) for n, s in sets.items()}, flush=True)
    best = None
    for alpha in a.alphas:
        for beta in a.betas:
            dec = make_decoder(a.lm, ctc_labels(units), alpha, beta)
            res = scores(sets, decode_sets(dec, sets, a.beam, a.workers))
            line = {"alpha": alpha, "beta": beta, "avg": round(avg_wer(res), 4), **{n: round(r.get("wer_nodigit", r["wer"]), 4) for n, r in res.items()}}
            print(json.dumps(line), flush=True)
            if best is None or line["avg"] < best["avg"]:
                best = line
    print("BEST", json.dumps(best), flush=True)


def cmd_decode(a):
    from .units import Units

    units = Units.load(a.units)
    sets = load_sets(a.logprobs, a.sets)
    dec = make_decoder(a.lm, ctc_labels(units), a.alpha, a.beta)
    import time

    t0 = time.time()
    hyps = decode_sets(dec, sets, a.beam, a.workers)
    res = scores(sets, hyps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, r in res.items():
        print(n, json.dumps(r), flush=True)
        with open(out / f"{n}.jsonl", "w") as f:
            for (ref, hyp) in zip(sets[n][1], hyps[n]):
                f.write(json.dumps({"ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")
    meta = {"lm": a.lm, "alpha": a.alpha, "beta": a.beta, "beam": a.beam, "decode_seconds": time.time() - t0}
    (out / "summary.json").write_text(json.dumps({"decoding": meta, **res}, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--out", default="data/lm/ncc.txt")
    c.add_argument("--shards", type=int, default=4)
    c.add_argument("--workers", type=int, default=os.cpu_count())
    b = sub.add_parser("build")
    b.add_argument("--text", nargs="+", required=True, help="text files or directories of .txt files")
    b.add_argument("--eval-dirs", nargs="+", default=["data/eval", "data/dev"])
    b.add_argument("--out", default="data/lm/no4")
    b.add_argument("--order", type=int, default=4)
    b.add_argument("--min-word-count", type=int, default=2)
    for name in ("tune", "decode"):
        s = sub.add_parser(name)
        s.add_argument("--lm", default="data/lm/no4")
        s.add_argument("--units", required=True, help="units.json of the acoustic model")
        s.add_argument("--logprobs", required=True, help="dir with <set>.logprobs.npz + <set>.jsonl from evaluate")
        s.add_argument("--sets", nargs="*")
        s.add_argument("--beam", type=int, default=64)
        s.add_argument("--workers", type=int, default=os.cpu_count())
    t = sub.choices["tune"]
    t.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    t.add_argument("--betas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    d = sub.choices["decode"]
    d.add_argument("--alpha", type=float, required=True)
    d.add_argument("--beta", type=float, required=True)
    d.add_argument("--out", required=True)
    a = p.parse_args()
    {"corpus": cmd_corpus, "build": cmd_build, "tune": cmd_tune, "decode": cmd_decode}[a.cmd](a)


if __name__ == "__main__":
    main()
