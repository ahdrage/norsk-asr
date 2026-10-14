"""Find encoder RMSNorm inputs with near-zero RMS (which amplify gradients by up to 1/sqrt(eps)).

    python scripts/diag_rms.py --train-dir data/train_small --batches 100
"""

import argparse
import collections

import torch

from norsk_asr.data import PackedSet, bucket_batches, collate
from norsk_asr.model import CTCModel, ModelConfig

p = argparse.ArgumentParser()
p.add_argument("--train-dir", default="data/train_small")
p.add_argument("--batches", type=int, default=100)
p.add_argument("--layers", type=int, nargs="*", default=[10, 11])
a = p.parse_args()

dev = torch.device("cuda")
ds = PackedSet.from_dir(a.train_dir, max_seconds=30)
m = CTCModel(ModelConfig(vocab_size=10, n_layers=1, head_init="random")).to(dev).eval()
cur = {}


def hook(name):
    def f(mod, inp, out):
        cur[name] = inp[0].detach().float().pow(2).mean(-1)  # [B, T]
    return f


for i in a.layers:
    for n, mod in m.encoder.layers[i].named_modules():
        if type(mod).__name__ == "Gemma4RMSNorm":
            mod.register_forward_hook(hook(f"L{i}.{n}"))

lowest = collections.defaultdict(lambda: (float("inf"), ""))
counts = collections.Counter()
for k, idx in enumerate(bucket_batches(ds, 300, seed=0)[: a.batches]):
    b = collate(ds, idx)
    wav, lens = b["wav"].to(dev), b["lengths"].to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        feats, fmask = m.features(wav, lens)
        cur.clear()
        m.encoder(feats, torch.ones_like(fmask), return_dict=True)
    valid = m.output_mask(fmask.sum(1), next(iter(cur.values())).shape[1])
    # Frames of digital silence inside the real audio: 40 ms windows with no non-zero sample.
    for name, ms in cur.items():
        for kind, sel in (("valid", valid), ("padded", ~valid)):
            if sel.any():
                v = ms[sel]
                mn = v.min().item()
                counts[(name, kind, "ms<1e-3")] += int((v < 1e-3).sum())
                if mn < lowest[(name, kind)][0]:
                    lowest[(name, kind)] = (mn, f"batch {k}")
for (name, kind), (mn, where) in sorted(lowest.items(), key=lambda x: x[1][0])[:16]:
    print(f"{name:40s} {kind:6s} min mean-square {mn:.2e} ({where}); frames with ms<1e-3: {counts[(name, kind, 'ms<1e-3')]}")
