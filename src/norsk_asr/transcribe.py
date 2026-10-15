"""Transcribe audio files of any length.

    uv run --extra lm python -m norsk_asr.transcribe opptak.mp3
    uv run --extra lm python -m norsk_asr.transcribe opptak.m4a --out opptak.txt --timestamps
    uv run python -m norsk_asr.transcribe opptak.wav --no-lm          # greedy, no KenLM needed

Long recordings are cut into pieces of at most --max-seconds (the model was trained on clips of
30 s or less), each cut placed at the quietest 100 ms in the last few seconds before the limit, so
words are rarely split. Output is raw text: lowercase, no punctuation, numbers as words.

wav/flac/ogg/mp3 are read directly; anything else (m4a, mp4, webm, ...) goes through ffmpeg,
either the system one or the binary bundled with imageio-ffmpeg.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .data import SR
from .evaluate import greedy

WIN = SR // 10  # 100 ms energy windows for choosing cut points


def load_audio(path: str) -> np.ndarray:
    """Mono float32 at 16 kHz."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        return audio
    except sf.LibsndfileError:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    raw = subprocess.run(
        [ffmpeg, "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def split(audio: np.ndarray, max_seconds: float, search_seconds: float = 5.0) -> list[tuple[int, int]]:
    """(start, end) sample ranges of at most max_seconds, cut at low-energy points."""
    n, max_len, search = len(audio), int(max_seconds * SR), int(search_seconds * SR)
    out, start = [], 0
    while n - start > max_len:
        lo, hi = start + max_len - search, start + max_len
        frames = audio[lo:hi][: (hi - lo) // WIN * WIN].reshape(-1, WIN)
        cut = lo + int(np.argmin((frames**2).mean(axis=1))) * WIN + WIN // 2
        out.append((start, cut))
        start = cut
    if n - start > SR // 10:
        out.append((start, n))
    return out


@torch.no_grad()
def log_probs(model, audio: np.ndarray, spans: list[tuple[int, int]], device, batch_seconds: float) -> list[np.ndarray]:
    """CTC log-probs per span, batching spans of similar length."""
    order = sorted(range(len(spans)), key=lambda i: spans[i][1] - spans[i][0])
    out: list[np.ndarray | None] = [None] * len(spans)
    batch: list[int] = []

    def run(idx):
        lens = torch.tensor([spans[i][1] - spans[i][0] for i in idx])
        wav = torch.zeros(len(idx), int(lens.max()))
        for j, i in enumerate(idx):
            wav[j, : lens[j]] = torch.from_numpy(audio[spans[i][0] : spans[i][1]])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lp, olens = model(wav.to(device), lens.to(device))
        for j, i in enumerate(idx):
            out[i] = lp[j, : olens[j]].float().cpu().numpy()

    longest = 0
    for i in order:
        n = spans[i][1] - spans[i][0]
        if batch and max(longest, n) * (len(batch) + 1) > batch_seconds * SR:
            run(batch)
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, n)
    if batch:
        run(batch)
    return out


def stamp(samples: int) -> str:
    s = samples / SR
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:04.1f}"


def main():
    from .train import load_checkpoint

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--ckpt", default="runs/full_top1/best.pt")
    p.add_argument("--lm", default="data/lm/no4", help="KenLM prefix (<lm>.bin + <lm>.unigrams.txt)")
    p.add_argument("--no-lm", action="store_true", help="greedy decoding only")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beam", type=int, default=64)
    p.add_argument("--max-seconds", type=float, default=25.0)
    p.add_argument("--batch-seconds", type=float, default=300.0)
    p.add_argument("--timestamps", action="store_true", help="one line per piece, prefixed with its time range")
    p.add_argument("--out", help="write the transcript here (one file: this path; several: a directory)")
    p.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cpu, cuda or mps (Apple GPU)")
    a = p.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    t0 = time.time()
    model, units = load_checkpoint(a.ckpt, device)
    decoder = None
    if not a.no_lm:
        from .lm import ctc_labels, make_decoder

        decoder = make_decoder(a.lm, ctc_labels(units), a.alpha, a.beta)
    print(f"loaded model on {device} in {time.time() - t0:.1f}s", file=sys.stderr)

    for path in a.files:
        t0 = time.time()
        audio = load_audio(path)
        spans = split(audio, a.max_seconds)
        lps = log_probs(model, audio, spans, device, a.batch_seconds)
        t_model = time.time() - t0
        if decoder is not None:
            texts = [decoder.decode(lp, beam_width=a.beam) for lp in lps]
        else:
            texts = [units.decode(greedy(torch.from_numpy(lp)[None], torch.tensor([len(lp)]))[0]) for lp in lps]
        wall = time.time() - t0
        if a.timestamps:
            body = "\n".join(f"[{stamp(s)} - {stamp(e)}] {t}" for (s, e), t in zip(spans, texts) if t)
        else:
            body = " ".join(t for t in texts if t)
        dur = len(audio) / SR
        print(
            f"{path}: {dur / 60:.1f} min of audio, {len(spans)} pieces, model {t_model:.1f}s, total {wall:.1f}s "
            f"({dur / wall:.0f}x real time)",
            file=sys.stderr,
        )
        if a.out:
            out = Path(a.out) / (Path(path).stem + ".txt") if len(a.files) > 1 else Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(body + "\n")
        else:
            print(body)


if __name__ == "__main__":
    main()
