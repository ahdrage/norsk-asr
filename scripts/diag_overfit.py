"""Sanity check: can the model memorise one fixed batch? Prints loss and a greedy decode.

    python scripts/diag_overfit.py --train-dir data/train_small --steps 300 [--model-kw head_init=random ...]
"""

import argparse
import time

import torch
import torch.nn.functional as F

from norsk_asr.data import PackedSet, collate
from norsk_asr.evaluate import greedy
from norsk_asr.model import CTCModel, ModelConfig
from norsk_asr.units import Units

p = argparse.ArgumentParser()
p.add_argument("--train-dir", default="data/train_small")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--unfreeze-at", type=int, default=-1)
p.add_argument("--encoder-lr", type=float, default=1e-5)
p.add_argument("--model-kw", nargs="*", default=[])
p.add_argument("--min-unit-count", type=int, default=20)
p.add_argument("--fresh-batches", action="store_true", help="new random batch each step instead of one fixed batch")
a = p.parse_args()

torch.manual_seed(0)
dev = torch.device("cuda")
ds = PackedSet.from_dir(a.train_dir, max_seconds=12)
units = Units.build((r["text"] for r in ds.rows), "google/embeddinggemma-2", a.min_unit_count)
print("units:", len(units), flush=True)
tg = [units.encode(r["text"]) for r in ds.rows]
kw = {"dropout": 0.0, "freq_masks": 0, "time_masks": 0}
for item in a.model_kw:
    k, v = item.split("=")
    kw[k] = type(getattr(ModelConfig, k))(v)
cfg = ModelConfig(vocab_size=len(units), **kw)
m = CTCModel(cfg, units.gemma_ids).to(dev)
m.set_trainable(encoder=a.unfreeze_at == 0)
opt = torch.optim.AdamW(m.param_groups(a.lr, a.encoder_lr, 0.0))
fixed = list(range(a.batch))
g = torch.Generator().manual_seed(0)
t0 = time.time()
for step in range(a.steps + 1):
    if step == a.unfreeze_at:
        m.set_trainable(encoder=True)
    for grp, lr in zip(opt.param_groups, (a.lr, a.encoder_lr)):
        grp["lr"] = lr * min(1.0, (step + 1) / 50)
    idx = torch.randint(len(ds), (a.batch,), generator=g).tolist() if a.fresh_batches else fixed
    b = collate(ds, idx, tg)
    m.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp, ol = m(b["wav"].to(dev), b["lengths"].to(dev))
    loss = F.ctc_loss(lp.transpose(0, 1), b["targets"].to(dev), ol, b["target_lengths"].to(dev), reduction="sum", zero_infinity=True) / b["target_lengths"].sum()
    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad], 1.0)
    if torch.isfinite(gn):
        opt.step()
    if step % 50 == 0:
        hyp = units.decode(greedy(lp[:1].detach(), ol[:1])[0])
        print(f"step {step:4d} loss {loss.item():7.3f} gn {gn.item():9.3g} t {time.time() - t0:5.0f}s | {hyp[:70]!r} <- {ds.rows[idx[0]]['text'][:50]!r}", flush=True)
