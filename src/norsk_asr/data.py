"""Memory-mapped packed shards and duration-bucketed batching."""

import json
import random
from pathlib import Path

import numpy as np
import torch

SR = 16000


class PackedSet:
    def __init__(self, paths: list[Path], max_seconds: float | None = None, limit: int | None = None):
        self.pcm = {}
        self.rows = []
        for p in sorted(paths):
            p = Path(p)
            self.pcm[p.stem] = np.memmap(p.with_suffix(".i16"), dtype=np.int16, mode="r")
            for line in open(p):
                r = json.loads(line)
                if max_seconds and r["n"] > max_seconds * SR:
                    continue
                r["shard"] = p.stem
                self.rows.append(r)
        if limit:
            self.rows = random.Random(0).sample(self.rows, min(limit, len(self.rows)))

    @classmethod
    def from_dir(cls, d: Path, **kw) -> "PackedSet":
        return cls(list(Path(d).glob("*.jsonl")), **kw)

    def __len__(self):
        return len(self.rows)

    def audio(self, i: int) -> np.ndarray:
        r = self.rows[i]
        return self.pcm[r["shard"]][r["offset"] : r["offset"] + r["n"]]

    def hours(self) -> float:
        return sum(r["n"] for r in self.rows) / SR / 3600


def collate(dataset: PackedSet, idx: list[int], targets: list[list[int]] | None = None):
    lengths = torch.tensor([dataset.rows[i]["n"] for i in idx])
    wav = torch.zeros(len(idx), int(lengths.max()))
    for j, i in enumerate(idx):
        wav[j, : lengths[j]] = torch.from_numpy(dataset.audio(i).astype(np.float32) / 32768.0)
    batch = {"wav": wav, "lengths": lengths, "idx": idx}
    if targets is not None:
        t = [targets[i] for i in idx]
        batch["targets"] = torch.tensor([u for x in t for u in x], dtype=torch.long)
        batch["target_lengths"] = torch.tensor([len(x) for x in t])
    return batch


def bucket_batches(dataset: PackedSet, max_batch_seconds: float, seed: int, shuffle: bool = True) -> list[list[int]]:
    """Group utterances of similar length so padding stays small; cap total padded seconds."""
    rng = random.Random(seed)
    order = sorted(range(len(dataset)), key=lambda i: dataset.rows[i]["n"] + (rng.random() * SR if shuffle else 0))
    batches, cur, cur_max = [], [], 0
    for i in order:
        n = dataset.rows[i]["n"]
        if cur and max(cur_max, n) * (len(cur) + 1) > max_batch_seconds * SR:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


class BatchLoader(torch.utils.data.Dataset):
    """Wraps precomputed batches so a DataLoader can build them in worker processes."""

    def __init__(self, dataset: PackedSet, batches: list[list[int]], targets=None):
        self.dataset, self.batches, self.targets = dataset, batches, targets

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, k):
        return collate(self.dataset, self.batches[k], self.targets)
