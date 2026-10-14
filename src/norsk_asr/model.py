"""EmbeddingGemma 2 audio encoder + learned layer mix + conformer reader + CTC head.

Mirrors the QUAN-Transcribe-1.0 config: 12-layer encoder with the top `unfreeze_top_k`
layers trainable, softmax-weighted mix of all hidden states, a 6-layer d=768 reader with a
kernel-15 convolution module, and an output head initialised from Gemma text embeddings.
The reader uses full attention (recordings only, no streaming requirement).
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from .features import LogMel


@dataclass
class ModelConfig:
    encoder_id: str = "google/embeddinggemma-2"
    vocab_size: int = 0
    d_model: int = 768
    n_layers: int = 6
    n_heads: int = 12
    ff_mult: int = 4
    conv_kernel: int = 15
    dropout: float = 0.1
    unfreeze_top_k: int = 2
    head_init: str = "text"  # "text" or "random"
    freq_masks: int = 2
    freq_width: int = 27
    time_masks: int = 10
    time_width_frac: float = 0.04

    def to_dict(self) -> dict:
        return asdict(self)


def rotary(x: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to [B, H, T, Dh]."""
    t, dh = x.shape[-2], x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d * mult, d), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rotary(q), rotary(k)
        attn_mask = mask[:, None, None, :]  # key padding
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        return self.drop(self.out(y.transpose(1, 2).reshape(b, t, d)))


class ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.GroupNorm(1, d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = F.glu(self.pw1(self.norm(x)), dim=-1) * mask[..., None]
        y = F.silu(self.bn(self.dw(y.transpose(1, 2)))).transpose(1, 2)
        return self.drop(self.pw2(y))


class ConformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.ff1 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.attn = SelfAttention(d, cfg.n_heads, cfg.dropout)
        self.conv = ConvModule(d, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, mask)
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class SpecAugment(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, feats, mask):
        if not self.training:
            return feats
        c = self.cfg
        b, t, f = feats.shape
        dev = feats.device
        feats = feats.clone()
        lengths = mask.sum(1)
        fr = torch.arange(f, device=dev)[None, :]
        for _ in range(c.freq_masks):
            w = torch.randint(0, c.freq_width + 1, (b, 1), device=dev)
            s = (torch.rand(b, 1, device=dev) * (f - w)).long()
            feats.masked_fill_(((fr >= s) & (fr < s + w))[:, None, :], 0.0)
        tr = torch.arange(t, device=dev)[None, :]
        for _ in range(c.time_masks):
            w = (torch.rand(b, 1, device=dev) * c.time_width_frac * lengths[:, None]).long()
            s = (torch.rand(b, 1, device=dev) * (lengths[:, None] - w).clamp(min=1)).long()
            feats.masked_fill_(((tr >= s) & (tr < s + w))[:, :, None], 0.0)
        return feats


class CTCModel(nn.Module):
    def __init__(self, cfg: ModelConfig, gemma_ids: list[int] | None = None, load_encoder_weights: bool = True):
        super().__init__()
        self.cfg = cfg
        if load_encoder_weights:
            full = AutoModel.from_pretrained(cfg.encoder_id, dtype=torch.float32)
        else:
            from transformers import AutoConfig

            full = AutoModel.from_config(AutoConfig.from_pretrained(cfg.encoder_id))
        self.encoder = full.audio_tower
        self.encoder.output_proj = nn.Identity()  # unused: we read the 1024-d hidden states
        text_emb = full.language_model.embed_tokens.weight.detach() if cfg.head_init == "text" else None
        del full

        n_states = len(self.encoder.layers) + 1
        enc_dim = self.encoder.config.hidden_size
        self.features = LogMel(cfg.encoder_id)
        self.specaug = SpecAugment(cfg)
        self.layer_weights = nn.Parameter(torch.zeros(n_states))
        self.proj = nn.Sequential(nn.Linear(enc_dim, cfg.d_model), nn.Dropout(cfg.dropout))
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))

        # Head: project to the text embedding space and score against per-unit embeddings
        # (initialised from Gemma's text embeddings for the same pieces), plus a learned blank.
        emb_dim = text_emb.shape[1] if text_emb is not None else cfg.d_model
        self.head_proj = nn.Linear(cfg.d_model, emb_dim)
        self.unit_emb = nn.Parameter(torch.randn(cfg.vocab_size, emb_dim) * emb_dim**-0.5)
        self.head_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        if text_emb is not None and gemma_ids is not None:
            with torch.no_grad():
                rows = text_emb[torch.tensor(gemma_ids[1:])].float()
                rows = rows / rows.norm(dim=-1, keepdim=True)
                self.unit_emb[1:] = rows

        self.set_trainable()

    def top_encoder_params(self) -> list[nn.Parameter]:
        k = self.cfg.unfreeze_top_k
        return [p for layer in self.encoder.layers[len(self.encoder.layers) - k :] for p in layer.parameters()] if k else []

    def set_trainable(self, encoder: bool = True):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.top_encoder_params():
            p.requires_grad = encoder

    def param_groups(self, lr: float, encoder_lr: float, weight_decay: float):
        enc = self.top_encoder_params()
        rest = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
        return [
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
            {"params": enc, "lr": encoder_lr, "weight_decay": weight_decay},
        ]

    @staticmethod
    def output_mask(mel_frames: torch.Tensor, t: int) -> torch.Tensor:
        """Valid encoder frames after its two stride-2 subsampling convolutions."""
        n = ((mel_frames + 1) // 2 + 1) // 2
        return torch.arange(t, device=n.device)[None, :] < n[:, None]

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns CTC log-probs [B, T, V] (float32) and output lengths [B]."""
        feats, fmask = self.features(wav, lengths)
        feats = self.specaug(feats, fmask)
        # Padding is fed as silence with an all-valid mask: the encoder's masked attention gives
        # NaN gradients in the unfrozen layers on padded rows, and since it only attends left
        # (right context 0 beyond its 0.48 s chunk) trailing silence barely changes real frames.
        out = self.encoder(feats, torch.ones_like(fmask), output_hidden_states=True, return_dict=True)
        mask = self.output_mask(fmask.sum(1), out.hidden_states[0].shape[1])
        states = torch.stack([F.layer_norm(h, h.shape[-1:]) for h in out.hidden_states])
        x = (torch.softmax(self.layer_weights, 0)[:, None, None, None] * states).sum(0)
        x = self.proj(x) * mask[..., None]
        for block in self.blocks:
            x = block(x, mask) * mask[..., None]
        logits = self.head_proj(x) @ self.unit_emb.t() + self.head_bias
        return logits.float().log_softmax(-1), mask.sum(1)
