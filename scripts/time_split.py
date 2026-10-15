"""Where does inference time go? Our CTC model vs NB-Whisper large, same clips, CPU fp32.

    uv run --extra lm python scripts/time_split.py

import json, time, sys
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
from norsk_asr.data import PackedSet
from norsk_asr.train import load_checkpoint

def clips(name, n):
    ds = PackedSet([f"data/eval/{name}.jsonl"], limit=n)
    return [ds.audio(i).astype(np.float32) / 32768 for i in range(len(ds))]

sets = {"long (Stortinget, ~26 s)": clips("stortinget_val", 4), "short (NPSC, ~7 s)": clips("npsc_nb", 4)}

def t(fn, reps=2):
    fn(); best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(); best = min(best, time.perf_counter() - t0)
    return best, r

m, units = load_checkpoint("runs/full_top1", torch.device("cpu"))
from transformers import WhisperForConditionalGeneration, WhisperProcessor
proc = WhisperProcessor.from_pretrained("NbAiLab/nb-whisper-large")
wm = WhisperForConditionalGeneration.from_pretrained("NbAiLab/nb-whisper-large", dtype=torch.float32).eval()
print("params: ours encoder %.0fM, reader+head %.0fM | whisper encoder %.0fM, decoder %.0fM" % (
    sum(p.numel() for p in m.encoder.parameters())/1e6,
    sum(p.numel() for n,p in m.named_parameters() if not n.startswith("encoder"))/1e6,
    sum(p.numel() for p in wm.model.encoder.parameters())/1e6, sum(p.numel() for p in wm.model.decoder.parameters())/1e6), flush=True)

for label, aud in sets.items():
    secs = sum(len(a) for a in aud) / 16000
    lens = torch.tensor([len(a) for a in aud]); wav = torch.zeros(len(aud), int(lens.max()))
    for i, a in enumerate(aud): wav[i, :len(a)] = torch.from_numpy(a)
    with torch.no_grad():
        feats, fmask = m.features(wav, lens)
        t_feat, _ = t(lambda: m.features(wav, lens))
        t_enc, out = t(lambda: m.encoder(feats, torch.ones_like(fmask), output_hidden_states=True, return_dict=True))
        t_all, _ = t(lambda: m(wav, lens))
        x = proc([a for a in aud], sampling_rate=16000, return_tensors="pt").input_features
        tw_enc, _ = t(lambda: wm.model.encoder(x), reps=1)
        tw_all, ids = t(lambda: wm.generate(x, language="no", task="transcribe", num_beams=1, max_new_tokens=200), reps=1)
    ntok = int((ids != proc.tokenizer.pad_token_id).sum()) - 4 * len(aud)
    print(json.dumps({"clips": label, "audio_s": round(secs, 1),
        "ours": {"features": round(t_feat, 2), "encoder": round(t_enc, 2), "reader+head": round(t_all - t_feat - t_enc, 2), "total": round(t_all, 2)},
        "whisper": {"encoder (30 s windows)": round(tw_enc, 2), "decoder (autoregressive)": round(tw_all - tw_enc, 2), "total": round(tw_all, 2), "tokens": ntok},
        "speedup": round(tw_all / t_all, 1)}), flush=True)
