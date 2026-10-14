"""Download open Norwegian speech data and pack it for training.

Each output shard is `<name>.i16` (concatenated 16 kHz mono int16 samples) plus `<name>.jsonl`
with one line per utterance: id, offset, n (samples), text (normalized), orig, has_digits.
Packing once means training only memory-maps PCM and never decodes audio.

    python -m norsk_asr.prep train --out data/train --stortinget-shards 100 [--npsc] [--lm-text data/lm_text]
    python -m norsk_asr.prep eval --out data/eval
    python -m norsk_asr.prep dev --out data/dev    # for tuning decoding, never for reporting
"""

import argparse
import base64
import csv
import io
import json
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

from .text import has_digits, is_clean, normalize

SR = 16000
ST_REPO = "NbAiLab/nb_distil_speech_noconcat_stortinget"
NST_REPO = "NbAiLab/nb_distil_speech_noconcat_nst"


def to_i16(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


class ShardWriter:
    def __init__(self, out_dir: Path, name: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.pcm = open(out_dir / f"{name}.i16", "wb")
        self.index = open(out_dir / f"{name}.jsonl", "w")
        self.offset = 0
        self.kept = self.seconds = 0

    def add(self, uid: str, pcm: np.ndarray, text: str, orig: str, **extra):
        self.pcm.write(pcm.tobytes())
        row = {"id": uid, "offset": self.offset, "n": len(pcm), "text": text, "orig": orig, **extra}
        self.index.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.offset += len(pcm)
        self.kept += 1
        self.seconds += len(pcm) / SR

    def close(self):
        self.pcm.close()
        self.index.close()


def pack_distil_shard(args) -> dict:
    """Pack one NbAiLab distil jsonl shard, keeping clean, digit-free, well-aligned utterances."""
    repo, filename, out_dir, max_whisper_wer, keep_digits, lm_text = args
    path = hf_hub_download(repo, filename, repo_type="dataset")
    name = Path(filename).name.removesuffix(".jsonl")
    w = ShardWriter(Path(out_dir), name)
    lm = open_lm_text(lm_text, name)
    seen = 0
    drops = {"digits": 0, "wer": 0, "chars": 0, "decode": 0}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            seen += 1
            orig = r["text"] or ""
            if lm:
                lm.write(normalize(orig) + "\n")
            if has_digits(orig) and not keep_digits:
                drops["digits"] += 1
                continue
            wer = r.get("whisper_wer")
            if max_whisper_wer is not None and wer is not None and wer > max_whisper_wer:
                drops["wer"] += 1
                continue
            text = normalize(orig)
            if not is_clean(text):
                drops["chars"] += 1
                continue
            try:
                audio, sr = sf.read(io.BytesIO(base64.b64decode(r["audio"])), dtype="float32")
            except Exception:
                drops["decode"] += 1
                continue
            w.add(r["id"], to_i16(audio, sr), text, orig, has_digits=has_digits(orig), source=r["source"])
    w.close()
    if lm:
        lm.close()
    return {"shard": name, "seen": seen, "kept": w.kept, "hours": w.seconds / 3600, **drops}


def open_lm_text(lm_text, name: str):
    """All normalized transcripts of a training shard, as language-model text (one per line)."""
    if not lm_text:
        return None
    Path(lm_text).mkdir(parents=True, exist_ok=True)
    return open(Path(lm_text) / f"{name}.txt", "w")


def npsc_rows(split: str, day: str) -> tuple[dict, str]:
    """Bokmål sentences of one NPSC meeting day, keyed by audio file stem, plus the audio tarball."""
    meta = hf_hub_download("NbAiLab/NPSC", f"data/{split}/{day}.json", repo_type="dataset")
    tgz = hf_hub_download("NbAiLab/NPSC", f"data/{split}/{day}_16K_mp3.tar.gz", repo_type="dataset")
    rows = {}
    for line in open(meta):
        r = json.loads(line)
        if r["sentence_language_code"] == "nb-NO":
            rows[Path(r["path"]).name] = r
    return rows, tgz


def npsc_text(orig: str) -> str:
    # Verbatim markers such as <inaudible>, <ee>, <mm> are not words.
    return normalize(re.sub(r"<[^>]*>", " ", orig))


def npsc_days(split: str) -> list[str]:
    files = HfApi().list_repo_files("NbAiLab/NPSC", repo_type="dataset")
    return sorted({Path(f).name.split(".")[0] for f in files if re.match(rf"data/{split}/\d+\.json$", f)})


def pack_npsc_day(args) -> dict:
    """Pack one NPSC training day with the same filters as the distil shards (no whisper_wer here)."""
    day, out_dir, lm_text = args
    rows, tgz = npsc_rows("train", day)
    name = f"npsc-train-{day}"
    w = ShardWriter(Path(out_dir), name)
    lm = open_lm_text(lm_text, name)
    drops = {"digits": 0, "wer": 0, "chars": 0, "decode": 0}
    with tarfile.open(tgz) as tar:
        for m in tar:
            stem = Path(m.name).stem
            if not m.isfile() or stem not in rows:
                continue
            orig = rows[stem]["text"]
            text = npsc_text(orig)
            if lm and text:
                lm.write(text + "\n")
            if has_digits(orig):
                drops["digits"] += 1
                continue
            if not text or not is_clean(text):
                drops["chars"] += 1
                continue
            try:
                audio, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
            except Exception:
                drops["decode"] += 1
                continue
            w.add(stem, to_i16(audio, sr), text, orig, has_digits=False, source="npsc")
    w.close()
    if lm:
        lm.close()
    return {"shard": name, "seen": len(rows), "kept": w.kept, "hours": w.seconds / 3600, **drops}


def run_pool(jobs, workers: int):
    """jobs: (function, args) pairs."""
    totals = {}
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_call, jobs):
            print(json.dumps(res), flush=True)
            for k, v in res.items():
                if k != "shard":
                    totals[k] = totals.get(k, 0) + v
    print("TOTAL", json.dumps(totals))


def _call(job):
    fn, args = job
    return fn(args)


def shard_files(repo: str, split: str) -> list[str]:
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if re.match(rf"data/{split}-\d+-of-\d+\.", f))


def cmd_train(a):
    st = shard_files(ST_REPO, "train")[: a.stortinget_shards]
    nst = shard_files(NST_REPO, "train")[: a.nst_shards]
    days = npsc_days("train") if a.npsc else []
    # NPSC days and big Stortinget shards first, so the pool doesn't end on one long job.
    jobs = [(pack_npsc_day, (d, a.out, a.lm_text)) for d in days]
    jobs += [(pack_distil_shard, (ST_REPO, f, a.out, a.max_whisper_wer, False, a.lm_text)) for f in st]
    jobs += [(pack_distil_shard, (NST_REPO, f, a.out, a.max_whisper_wer, False, a.lm_text)) for f in nst]
    print(f"packing {len(days)} npsc days + {len(st)} stortinget + {len(nst)} nst shards with {a.workers} workers", flush=True)
    run_pool(jobs, a.workers)


def eval_fleurs(out: Path, split: str = "test", name: str = "fleurs_nb"):
    tsv = hf_hub_download("google/fleurs", f"data/nb_no/{split}.tsv", repo_type="dataset")
    tgz = hf_hub_download("google/fleurs", f"data/nb_no/audio/{split}.tar.gz", repo_type="dataset")
    rows = {}
    with open(tsv) as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.setdefault(r[1], r[2])  # file name -> raw transcription
    w = ShardWriter(out, name)
    with tarfile.open(tgz) as tar:
        for m in tar:
            name = Path(m.name).name
            if not m.isfile() or name not in rows:
                continue
            audio, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
            orig = rows[name]
            w.add(name, to_i16(audio, sr), normalize(orig), orig, has_digits=has_digits(orig))
    w.close()
    return w


def eval_npsc(out: Path, split: str = "test", name: str = "npsc_nb"):
    w = ShardWriter(out, name)
    for day in npsc_days(split):
        rows, tgz = npsc_rows(split, day)
        with tarfile.open(tgz) as tar:
            for m in tar:
                stem = Path(m.name).stem
                if not m.isfile() or stem not in rows:
                    continue
                orig = rows[stem]["text"]
                text = npsc_text(orig)
                if not text:
                    continue
                audio, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
                w.add(stem, to_i16(audio, sr), text, orig, has_digits=has_digits(orig))
    w.close()
    return w


def eval_distil(out: Path, repo: str, filename: str, name: str):
    path = hf_hub_download(repo, filename, repo_type="dataset")
    w = ShardWriter(out, name)
    for line in open(path):
        r = json.loads(line)
        orig = r["text"] or ""
        audio, sr = sf.read(io.BytesIO(base64.b64decode(r["audio"])), dtype="float32")
        w.add(r["id"], to_i16(audio, sr), normalize(orig), orig, has_digits=has_digits(orig))
    w.close()
    return w


def cmd_eval(a):
    out = Path(a.out)
    for name, fn in [
        ("fleurs_nb", lambda: eval_fleurs(out)),
        ("npsc_nb", lambda: eval_npsc(out)),
        ("stortinget_val", lambda: eval_distil(out, ST_REPO, "data/validation_clean_stortinget_no-00000-of-00001.stortinget.jsonl", "stortinget_val")),
        ("nst_val", lambda: eval_distil(out, NST_REPO, "data/validation-00000-of-00001.nst.jsonl", "nst_val")),
    ]:
        w = fn()
        print(f"{name}: {w.kept} utts, {w.seconds / 3600:.2f} h", flush=True)


def cmd_dev(a):
    """Dev sets for tuning decoding (language-model weight etc.), disjoint from the reported test sets."""
    out = Path(a.out)
    for name, fn in [
        ("fleurs_dev", lambda: eval_fleurs(out, "dev", "fleurs_dev")),
        ("npsc_dev", lambda: eval_npsc(out, "eval", "npsc_dev")),
    ]:
        w = fn()
        print(f"{name}: {w.kept} utts, {w.seconds / 3600:.2f} h", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", default="data/train")
    t.add_argument("--stortinget-shards", type=int, default=100)
    t.add_argument("--nst-shards", type=int, default=10_000)
    t.add_argument("--max-whisper-wer", type=float, default=0.15)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--npsc", action="store_true", help="also pack the NPSC training split (Bokmål)")
    t.add_argument("--lm-text", help="also write every normalized transcript here, for the language model")
    e = sub.add_parser("eval")
    e.add_argument("--out", default="data/eval")
    d = sub.add_parser("dev")
    d.add_argument("--out", default="data/dev")
    a = p.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "dev": cmd_dev}[a.cmd](a)


if __name__ == "__main__":
    main()
