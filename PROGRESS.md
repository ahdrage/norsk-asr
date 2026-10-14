# Progress log

Newest entries first. Scores are WER (%) on digit-free utterances, greedy CTC unless noted, on
the same subsets as the NB-Whisper baseline (1,000 utterances per set; FLEURS has 357).

## Scoreboard

| Model | Data | FLEURS | NPSC | NST | Stortinget | Notes |
|---|---|---|---|---|---|---|
| NB-Whisper large | ~66k h | 4.2 | 16.5 | 2.0 | 9.0 | 40–90× real time |
| pilot_frozen | 465 h | 18.8 | 28.9 | 12.4 | 24.4 | encoder frozen |
| pilot_top1 | 465 h | 17.8 | 28.1 | 11.8 | 23.2 | top encoder layer trained, ~2,500× real time |

## Spend

RunPod H100 SXM at $3.99/h. Balance before each step from `scripts/pod.sh balance`.

| Date | What | GPU time | Cost | Balance after |
|---|---|---|---|---|
| 2026-10-14 | small run + pilot (two runs) + NB-Whisper baseline | ~3.2 h | ~$13 | – |
| 2026-10-14 | (credits added) | | | $46.20 |

## 2026-10-14: next round (in progress)

Plan, approved by the user:

1. **More data.** All 636 Stortinget shards + all NST + NPSC train (Bokmål, 32 days, ~100 h):
   about 1,700 h after filtering, 3.7× the pilot. One run with the pilot_top1 setup, 5 epochs
   (`scripts/run_full.sh`). Open data is smaller than first thought: the digit filter drops ~27%
   of Stortinget, and the rest of NbAiLab's speech sets are either test-only or duplicates.
2. **Language model.** Word 4-gram KenLM on open Norwegian text (NCC: parliament, government,
   Wikipedia; plus training transcripts), decoded with pyctcdecode. Text sharing any 8-gram with
   a test reference is removed. LM weight is tuned on dev sets (FLEURS dev, NPSC eval split),
   never on the reported test sets. Built and tuned on the NUC, so it costs no GPU time.
3. **NPSC training data**: part of step 1.

## 2026-10-14: pilot

465 h (100 Stortinget shards + all NST), 6 epochs, two runs side by side on one H100.

- Characters work as CTC units; Gemma's multi-character pieces stayed on the blank plateau.
- Training the top 2 encoder layers overflowed the gradients; the top 1 layer, unfrozen after
  2,000 steps, is stable and beats the frozen encoder on every set by about 1 point.
- Both runs were still improving at the end.
