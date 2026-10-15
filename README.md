# norsk-asr

Fast Norwegian (Bokmål) speech-to-text built on the audio encoder of
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), following the recipe of
[QUAN-Transcribe-1.0](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0), trained only on open
data. Model and language model: [huggingface.co/ahdrage/norsk-asr](https://huggingface.co/ahdrage/norsk-asr).

- the 12-layer Gemma 4 audio conformer (25 frames/s), with only the top layer trained;
- a learned softmax mix over all 13 hidden states;
- a 6-layer conformer reader (d=768) trained with CTC, in one pass with no autoregressive decoding;
- output units are Gemma's single-character pieces (83 units), with the head initialised from
  EmbeddingGemma's text embeddings;
- optional beam search with a word 4-gram KenLM language model.

It runs at ~1,000–1,600× real time on one H100 (greedy) and ~10× real time on a laptop-class CPU.
The bar to beat is [NB-Whisper](https://huggingface.co/NbAiLab/nb-whisper-large).

## Results

WER (%) on digit-free utterances; the same 1,000-utterance subsets for every model (FLEURS has 357).

| Model | Training data | FLEURS | NPSC | NST | Stortinget |
|---|---|---|---|---|---|
| NB-Whisper large | ~66k h | **4.2** | 16.5 | **2.0** | **9.0** |
| `pilot_top1`, greedy | 465 h | 17.8 | 28.1 | 11.8 | 23.2 |
| `full_top1`, greedy | 1,680 h | 15.2 | 12.9 | 8.9 | 15.3 |
| `full_top1` + 4-gram LM | 1,680 h | 10.0 | **9.7** | 5.6 | 12.2 |

The NPSC lead is mostly transcription style, not better recognition: NB-Whisper writes clean text
and drops verbatim fillers, which NPSC's references keep. Counting only misheard words,
NB-Whisper is 1.5–2.5× better on the other sets. [`PROGRESS.md`](PROGRESS.md) has the full log,
error breakdown and costs.

Scope and caveats:

- Output is raw: lowercase, no punctuation, numbers as words.
- Recordings only (no streaming).
- Training utterances whose transcripts contain digits are dropped (27% of Stortinget), since we
  can't know how the numbers were spoken; for the same reason scores use digit-free utterances.
- Test sets: FLEURS nb_no test, NPSC Bokmål test, Stortinget validation (clean), NST validation.
  About 70% of the test audio is parliament speech; none is broadcast, conversational or noisy.

## Transcribing a file

```bash
uv sync --extra lm                                   # KenLM bindings for beam search (needs a C++ compiler)
uv run --extra lm python -m norsk_asr.download       # model + LM from the Hub (~850 MB)
uv run --extra lm python -m norsk_asr.transcribe opptak.mp3                    # print the transcript
uv run --extra lm python -m norsk_asr.transcribe opptak.m4a --timestamps --out opptak.txt
uv run --extra lm python -m norsk_asr.transcribe *.wav --out transkripsjoner/  # many files
uv run python -m norsk_asr.transcribe opptak.wav --no-lm                       # greedy, no KenLM
```

Any length and any format ffmpeg reads (a bundled ffmpeg is used if none is installed). Long audio
is cut at quiet points into pieces of at most 25 s. The first run also downloads the
EmbeddingGemma 2 encoder (1.5 GB). Always pass `--extra lm` to `uv run`, or uv removes KenLM.

### On a Mac

Apple Silicon and macOS 14+ (torch has no Intel-Mac builds).

```bash
xcode-select --install                    # C++ compiler for KenLM's Python bindings (once)
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/ahdrage/norsk-asr.git && cd norsk-asr
uv sync --extra lm
uv run --extra lm python -m norsk_asr.download
uv run --extra lm python -m norsk_asr.transcribe ~/Downloads/opptak.m4a --timestamps
```

`--device mps` tries the Apple GPU (untested); the default is the CPU. If KenLM fails to build,
use `uv sync` and `--no-lm`.

## Training

```bash
uv sync
uv run python -m norsk_asr.prep eval --out data/eval
uv run python -m norsk_asr.prep dev --out data/dev       # for tuning the LM only
uv run python -m norsk_asr.prep train --out data/train --stortinget-shards 10000 --npsc \
    --lm-text data/lm_text --workers 16
uv run python -m norsk_asr.train --train-dir data/train --eval-dir data/eval --out runs/full \
    --epochs 5 --unfreeze-top-k 1 --unfreeze-after 2000 --encoder-lr 1e-5
uv run python -m norsk_asr.evaluate --ckpt runs/full/best.pt --limit 1000 --out results/full --dump-logprobs
uv run python -m norsk_asr.baseline --model NbAiLab/nb-whisper-large --limit 1000 --out results/nb-whisper-large
```

Language model (needs KenLM's `lmplz` and `build_binary` on PATH):

```bash
uv run --extra lm python -m norsk_asr.lm corpus --out data/lm/ncc.txt --shards 4
uv run --extra lm python -m norsk_asr.lm build --text data/lm/ncc.txt data/lm_text --out data/lm/no4
uv run --extra lm python -m norsk_asr.lm tune --units runs/full/units.json --logprobs results/full_dev
```

`scripts/pod.sh` manages a RunPod GPU pod and `scripts/run_full.sh` / `finish_full.sh` run the
full recipe on it (about 4 h on one H100).

| Path | What |
|---|---|
| `src/norsk_asr/text.py` | normalization used for targets and scoring |
| `src/norsk_asr/units.py` | character CTC vocabulary (Gemma pieces) |
| `src/norsk_asr/features.py` | GPU log-mel matching `Gemma4AudioFeatureExtractor` (max diff 2e-5) |
| `src/norsk_asr/model.py` | encoder + layer mix + reader + head |
| `src/norsk_asr/prep.py` | download and pack data into int16 PCM shards |
| `src/norsk_asr/train.py` | training loop, checkpoints (.pt or safetensors) |
| `src/norsk_asr/evaluate.py` | greedy CTC decoding and WER |
| `src/norsk_asr/lm.py` | LM corpus, decontamination, KenLM build, beam search tuning |
| `src/norsk_asr/transcribe.py` | transcribe audio files |
| `src/norsk_asr/baseline.py` | NB-Whisper on the same sets |

## Notes

- Feed the encoder **no padding mask**. Padding is filled with silence instead, because masked
  attention in the unfrozen layers produced NaN gradients during development. The encoder only
  attends left, so trailing silence barely changes real frames.
- Training the top encoder layers against a randomly initialised reader makes their gradients
  overflow within ~10 steps. `--unfreeze-after` keeps them frozen until the reader has warmed up.
  Unfreezing the top 2 layers was unstable even then.

## License

Code: Apache-2.0 ([LICENSE](LICENSE)). Model weights: Apache-2.0 (derived from EmbeddingGemma 2,
Apache-2.0). The language model files (`lm/` on the Hub) are CC BY-SA 4.0 because their text
includes Norwegian Wikipedia (CC BY-SA 3.0); see the model card for data sources and attribution.
