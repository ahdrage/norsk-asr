# Progress log

Newest entries first. Scores are WER (%) on digit-free utterances, greedy CTC unless noted, on
the same subsets as the NB-Whisper baseline (1,000 utterances per set; FLEURS has 357).

## Scoreboard

| Model | Data | FLEURS | NPSC | NST | Stortinget | Notes |
|---|---|---|---|---|---|---|
| NB-Whisper large | ~66k h | 4.2 | 16.5 | 2.0 | 9.0 | 40–90× real time |
| pilot_frozen | 465 h | 18.8 | 28.9 | 12.4 | 24.4 | encoder frozen |
| pilot_top1 | 465 h | 17.8 | 28.1 | 11.8 | 23.2 | top encoder layer trained, ~2,500× real time |
| pilot_top1 + 4-gram LM | 465 h | 12.8 | 20.3 | 7.0 | 16.5 | beam 32, α=0.5 β=2.0 (tuned on dev); 26 s on 8 CPU cores for all four sets |
| full_top1 | 1,680 h | 15.2 | 12.9 | 8.9 | 15.3 | greedy; 1,000–1,650× real time on one H100 |
| **full_top1 + 4-gram LM** | 1,680 h | **10.0** | **9.7** | **5.6** | **12.2** | beam 64, α=0.6 β=1.0 (tuned on dev); ~1,500× real time on 8 CPU cores |

## Spend

RunPod H100 SXM at $3.99/h. Balance before each step from `scripts/pod.sh balance`.

| Date | What | GPU time | Cost | Balance after |
|---|---|---|---|---|
| 2026-10-14 | small run + pilot (two runs) + NB-Whisper baseline | ~3.2 h | ~$13 | – |
| 2026-10-14 | (credits added) | | | $46.20 |
| 2026-10-14 | full round: pack 1,680 h + train full_top1 + evals (pod 17:17–21:07 UTC) | 3.8 h | $15.38 | $30.82 |

## 2026-10-14: full round, results

- `full_top1` trained 52,600 steps (5 epochs) with no skipped steps; best checkpoint at the end
  (avg greedy WER 13.1% on the 500-utt checks). Uploaded to `ahdrage/norsk-asr` (`runs/full_top1`,
  `results/full_top1`); the LM is in `lm/`.
- LM retuned on the new model's dev outputs (`logs/lm_tune_full.log`): α=0.6, β=1.0, beam 64.
  The LM still cuts WER by 20–37% relative.
- **NPSC is a style win, not an accuracy win.** NB-Whisper's 16.5% is mostly deletions (1,801 vs
  our 613): it drops fillers ("og", "så", "jo", "president") and standardizes forms ("regjeringa" →
  "regjeringen", "meiner" → "mener"), because it writes clean subtitle-style text and NPSC is
  verbatim. Substitutions are about equal (1,011 vs 1,060).
- Per-error-type (% of reference words, digit-free; NB-Whisper vs ours + LM):
  Stortinget sub 3.4 vs 5.3, del 2.9 vs 4.1, ins 2.7 vs 2.8; FLEURS sub 3.0 vs 7.3;
  NST sub 1.6 vs 4.1. On genuinely misheard words NB-Whisper is 1.5–2.5× better outside NPSC.

## 2026-10-14: next round (setup notes)

Status (17:55 UTC):
- Pod `0m8gb6jriql1zx` (H100) up since 17:17 UTC. It had a CUDA 12.8 driver, so torch was swapped
  for the cu126 build of the same version (now automatic in `pod.sh setup`).
- Data packed in 26 min: **1,679.9 h, 373,627 utterances** (Stortinget 636 shards, NST 636 shards,
  NPSC train 32 days ≈ 100 h). Dropped: 75,706 with digits, 15,895 with whisper_wer > 0.15.
- Training `runs/full_top1`: 10,520 steps/epoch, 52,600 steps, 0.83 audio-h/s → about 3 h.
- Test/train overlap check (sentences of 6+ words found verbatim in training text): NPSC 40/4,771,
  Stortinget 1/697, NST 56/1,494, FLEURS 0/357. The NPSC hits are stock Speaker's phrases
  ("neste taler er …") and the NST hits are NST's shared reading prompts, so no leaked sessions.
- LM text: NCC shards 1–4 (parliament, government, Wikipedia, public reports; 1990+; number-heavy
  sentences dropped) = 8.8 M sentences, 162 M words; plus 20 M words of training transcripts.

LM results (pilot model, 20:15 UTC):
- `data/lm/no4.bin`: 4-gram, 3- and 4-gram singletons pruned, 406 MB; 8.57 M sentences after
  removing 16,312 that share an 8-gram with eval/dev references and 643,850 duplicates; 918k
  unigrams. Built on the NUC in ~6 min.
- Grid on dev (FLEURS dev + 1,000 NPSC eval-split utts), beam 32: best α=0.5, β=2.0
  (avg 16.5% vs 23.3% greedy). Log: `logs/lm_tune_pilot.log`.
- On the test sets the LM cuts WER by 28–40% relative (scoreboard).

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
