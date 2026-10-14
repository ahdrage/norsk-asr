"""Train the CTC reader on packed shards.

    python -m norsk_asr.train --train-dir data/train --eval-dir data/eval --out runs/pilot

Checkpoints hold only trainable weights (reader, layer mix, head, top encoder layers); the
frozen encoder is reloaded from the Hub, so a checkpoint is ~300 MB instead of ~1.3 GB.
"""

import argparse
import json
import math
import time
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import BatchLoader, PackedSet, bucket_batches
from .evaluate import evaluate_set, load_eval_sets
from .model import CTCModel, ModelConfig
from .units import Units


def save_checkpoint(path: Path, model: CTCModel, units: Units, step: int, extra: dict):
    top = {id(p) for p in model.top_encoder_params()}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad or id(p) in top}
    state = {k: v for k, v in model.state_dict().items() if k in trainable}
    torch.save({"config": model.cfg.to_dict(), "units": {"pieces": units.pieces, "gemma_ids": units.gemma_ids, "model_id": units.model_id}, "state": state, "step": step, **extra}, path)


def load_checkpoint(path, device) -> tuple[CTCModel, Units]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{f.name: ck["config"][f.name] for f in fields(ModelConfig) if f.name in ck["config"]})
    u = ck["units"]
    units = Units(u["pieces"], u["gemma_ids"], u["model_id"])
    model = CTCModel(cfg, units.gemma_ids)
    missing = set(ck["state"]) - set(model.state_dict())
    if missing:
        raise KeyError(f"checkpoint keys not in model: {sorted(missing)[:5]}")
    model.load_state_dict(ck["state"], strict=False)
    return model.to(device).eval(), units


def lr_at(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / warmup
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default="data/train")
    p.add_argument("--eval-dir", default="data/eval")
    p.add_argument("--out", default="runs/pilot")
    p.add_argument("--units", help="units.json; built from the training text if absent")
    p.add_argument("--unit-type", choices=["chars", "pieces"], default="chars",
                   help="chars: letters (word-initial ones carry the space); pieces: also Gemma pieces seen "
                        ">= --min-unit-count times. Pieces stayed stuck on the CTC blank plateau in the small run.")
    p.add_argument("--min-unit-count", type=int, default=20)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-batch-seconds", type=float, default=1000)
    p.add_argument("--max-utt-seconds", type=float, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--unfreeze-after", type=int, default=1000,
                   help="keep the top encoder layers frozen for this many steps, then warm them up over --warmup steps; "
                        "training them from step 0 against a random reader makes their gradients overflow")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-limit", type=int, default=500)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    for f in fields(ModelConfig):
        if f.name not in ("encoder_id", "vocab_size"):
            p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train = PackedSet.from_dir(Path(a.train_dir), max_seconds=a.max_utt_seconds)
    print(f"train: {len(train)} utts, {train.hours():.1f} h", flush=True)
    if a.units:
        units = Units.load(Path(a.units))
    else:
        min_count = a.min_unit_count if a.unit_type == "pieces" else None
        units = Units.build((r["text"] for r in train.rows), ModelConfig.encoder_id, min_count)
    units.save(out / "units.json")
    targets = [units.encode(r["text"]) for r in train.rows]
    # CTC needs at least one frame per unit (plus blanks between repeats); 25 frames/s.
    keep = [i for i, t in enumerate(targets) if 0 < len(t) <= 0.9 * train.rows[i]["n"] / 640]
    if len(keep) < len(train):
        print(f"dropping {len(train) - len(keep)} utts with too many units for their length", flush=True)
        train.rows = [train.rows[i] for i in keep]
        targets = [targets[i] for i in keep]
    print(f"units: {len(units)}", flush=True)

    cfg_kw = {f.name: getattr(a, f.name) for f in fields(ModelConfig) if f.name not in ("encoder_id", "vocab_size")}
    cfg = ModelConfig(vocab_size=len(units), **cfg_kw)
    model = CTCModel(cfg, units.gemma_ids).to(device)
    model.set_trainable(encoder=a.unfreeze_after == 0)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in model.top_encoder_params())
    print(f"trainable params: {n_train / 1e6:.1f}M now, +{n_enc / 1e6:.1f}M encoder from step {a.unfreeze_after}", flush=True)
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.encoder_lr, a.weight_decay), betas=(0.9, 0.98), fused=device.type == "cuda")
    base_lrs = [g["lr"] for g in opt.param_groups]

    eval_sets = load_eval_sets(Path(a.eval_dir), None, a.eval_limit)
    steps_per_epoch = len(bucket_batches(train, a.max_batch_seconds, seed=0))
    total = a.max_steps or int(a.epochs * steps_per_epoch)
    print(f"steps/epoch {steps_per_epoch}, total steps {total}", flush=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    log = open(out / "metrics.jsonl", "a")

    step, best = 0, float("inf")
    t_last, audio_s = time.time(), 0.0
    model.train()
    epoch = 0
    while step < total:
        batches = bucket_batches(train, a.max_batch_seconds, seed=a.seed + epoch)
        loader = torch.utils.data.DataLoader(
            BatchLoader(train, batches, targets), batch_size=None, shuffle=False, num_workers=a.workers,
            pin_memory=device.type == "cuda", persistent_workers=False, prefetch_factor=4 if a.workers else None,
        )
        for b in loader:
            if step >= total:
                break
            if step == a.unfreeze_after and a.unfreeze_after > 0 and a.unfreeze_top_k:
                model.set_trainable(encoder=True)
                print(f"step {step}: unfreezing top {a.unfreeze_top_k} encoder layers", flush=True)
            opt.param_groups[0]["lr"] = base_lrs[0] * lr_at(step, total, a.warmup)
            enc_step = step - a.unfreeze_after
            opt.param_groups[1]["lr"] = base_lrs[1] * lr_at(enc_step, total - a.unfreeze_after, a.warmup) if enc_step >= 0 else 0.0
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lp, olens = model(b["wav"].to(device, non_blocking=True), b["lengths"].to(device, non_blocking=True))
            loss = F.ctc_loss(
                lp.transpose(0, 1), b["targets"].to(device), olens, b["target_lengths"].to(device),
                blank=0, reduction="sum", zero_infinity=True,
            ) / b["target_lengths"].sum().to(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(gn):
                print(json.dumps({"step": step, "skipped": "non-finite grad norm"}), flush=True)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            opt.step()
            step += 1
            audio_s += float(b["lengths"].sum()) / 16000
            if step % 25 == 0 or step == 1:
                dt = time.time() - t_last
                rec = {"step": step, "epoch": epoch, "loss": loss.item(), "grad_norm": gn.item(), "lr": opt.param_groups[0]["lr"],
                       "encoder_lr": opt.param_groups[1]["lr"],
                       "audio_hours_per_s": audio_s / 3600 / dt, "s_per_step": dt / (25 if step > 1 else 1)}
                if device.type == "cuda":
                    rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                t_last, audio_s = time.time(), 0.0
            if step % a.eval_every == 0 or step == total:
                res = {name: evaluate_set(model, units, ds, device)[0] for name, ds in eval_sets.items()}
                avg = sum(r.get("wer_nodigit", r["wer"]) for r in res.values()) / len(res)
                rec = {"step": step, "eval": res, "avg_wer_nodigit": avg,
                       "layer_weights": torch.softmax(model.layer_weights, 0).tolist()}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                save_checkpoint(out / "last.pt", model, units, step, {"avg_wer_nodigit": avg})
                if avg < best:
                    best = avg
                    save_checkpoint(out / "best.pt", model, units, step, {"avg_wer_nodigit": avg})
                t_last, audio_s = time.time(), 0.0
        epoch += 1
    print(f"done; best avg_wer_nodigit {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
