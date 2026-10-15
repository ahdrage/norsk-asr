"""Fetch a trained model and the language model from the private Hub repo, into the default paths
that `transcribe` uses (runs/<run>/ and data/lm/). Needs a Hub login with read access (`hf auth login`).

    uv run python -m norsk_asr.download                 # runs/full_top1 + data/lm/no4
    uv run python -m norsk_asr.download --run pilot_top1 --no-lm
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "ahdrage/norsk-asr"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="full_top1")
    p.add_argument("--lm", default="no4")
    p.add_argument("--no-lm", action="store_true")
    p.add_argument("--root", default=".", help="repo checkout to download into")
    a = p.parse_args()
    root = Path(a.root)
    files = [(f"runs/{a.run}/{f}", root / "runs" / a.run) for f in ("best.pt", "units.json", "args.json")]
    if not a.no_lm:
        files += [(f"lm/{a.lm}{ext}", root / "data" / "lm") for ext in (".bin", ".unigrams.txt")]
    for remote, local_dir in files:
        local_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(REPO, remote, local_dir=root)
        print("got", path, flush=True)
    # lm/ on the Hub lands in <root>/lm; transcribe expects data/lm.
    for ext in (".bin", ".unigrams.txt"):
        src = root / "lm" / f"{a.lm}{ext}"
        if src.exists() and not a.no_lm:
            src.replace(root / "data" / "lm" / src.name)


if __name__ == "__main__":
    main()
