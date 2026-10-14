# norsk-asr

Fast Norwegian speech-to-text built on the audio encoder of
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (Apache 2.0), following
the recipe of [QUAN-Transcribe-1.0](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0):

- the 12-layer Gemma 4 audio conformer (25 frames/s), frozen or with only the top layer trainable;
- a learned softmax mix over all 13 hidden states;
- a 6-layer conformer reader (d=768) trained with CTC, in one pass with no autoregressive decoding;
- output units are Gemma's single-character pieces (83 units), with the head initialised from
  EmbeddingGemma's text embeddings.

The bar to beat is [NB-Whisper](https://huggingface.co/NbAiLab/nb-whisper-large).

## Pilot scope

- Target text is raw: lowercase, no punctuation, numbers as words. Readable output comes later.
- Recordings only, so the reader uses full attention. There is no streaming mode.
- Open data only: the NbAiLab Stortinget and NST sets (CC0 / Apache 2.0). Test sets are FLEURS
  nb_no, NPSC Bokmål test, Stortinget validation (clean) and NST validation.
- Training utterances whose transcripts contain digits are dropped (27% of Stortinget), since we
  can't know how the numbers were spoken. Eval reports `wer` on everything and `wer_nodigit` on
  digit-free utterances; `wer_nodigit` is the fair comparison.

## Pilot results

465 h of training data (100 Stortinget shards plus all of NST), 6 epochs, one H100, about 3 h of
GPU time and about $13. Scores are WER on digit-free utterances, using greedy CTC on the same
1,000-utterance subsets (FLEURS has 357). The checkpoints are in the private Hub repo
`ahdrage/norsk-asr`.

| Model | FLEURS | NPSC | NST | Stortinget | Speed (× real time) |
|---|---|---|---|---|---|
| NB-Whisper large | 4.2 | 16.5 | 2.0 | 9.0 | 40–90 |
| `pilot_frozen` (encoder frozen) | 18.8 | 28.9 | 12.4 | 24.4 | ~2,400 |
| `pilot_top1` (top encoder layer trained) | **17.8** | **28.1** | **11.8** | **23.2** | ~2,500 |

Both runs were still improving at the end. `pilot_top1` beats the frozen run on every set.

## Layout

| Path | What |
|---|---|
| `src/norsk_asr/text.py` | normalization used for targets and scoring |
| `src/norsk_asr/units.py` | Gemma-piece CTC vocabulary |
| `src/norsk_asr/features.py` | GPU log-mel matching `Gemma4AudioFeatureExtractor` (max diff 2e-5) |
| `src/norsk_asr/model.py` | encoder + layer mix + reader + head |
| `src/norsk_asr/prep.py` | download and pack data into int16 PCM shards |
| `src/norsk_asr/train.py` | training loop |
| `src/norsk_asr/evaluate.py` | greedy CTC decoding and WER |
| `src/norsk_asr/baseline.py` | NB-Whisper on the same sets |
| `scripts/pod.sh` | RunPod pod lifecycle |

## Running

```bash
uv sync
uv run python -m norsk_asr.prep eval --out data/eval
uv run python -m norsk_asr.prep train --out data/train --stortinget-shards 100 --workers 16
uv run python -m norsk_asr.train --train-dir data/train --eval-dir data/eval --out runs/pilot
uv run python -m norsk_asr.evaluate --ckpt runs/pilot/best.pt --out results/pilot
uv run python -m norsk_asr.baseline --model NbAiLab/nb-whisper-large --out results/nb-whisper-large
```

## Notes

- Feed the encoder **no padding mask**. Padding is filled with silence instead, because masked
  attention in the unfrozen layers produced NaN gradients during development. The encoder only
  attends left, so trailing silence barely changes real frames.
- Training the top encoder layers against a randomly initialised reader makes their gradients
  overflow within ~10 steps. `--unfreeze-after` keeps them frozen until the reader has warmed up.
