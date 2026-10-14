"""Batched log-mel features on the GPU, numerically matching Gemma4AudioFeatureExtractor.

The Hugging Face extractor runs per utterance in NumPy, which is far too slow to feed an H100
with a mostly frozen encoder. This reuses its window and mel filters and reproduces its framing:
half-frame left padding, frames of `frame_length + 1` samples (the last sample is dropped
when there is no preemphasis), magnitude rFFT, mel matmul, log(x + mel_floor).
"""

import math

import torch
from torch import nn
from transformers import AutoFeatureExtractor


class LogMel(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        fe = AutoFeatureExtractor.from_pretrained(model_id)
        if fe.preemphasis or fe.per_bin_mean is not None or fe.per_bin_stddev is not None or fe.dither:
            raise NotImplementedError("Feature extractor options not reproduced here")
        self.frame_length = fe.frame_length
        self.hop_length = fe.hop_length
        self.fft_length = fe.fft_length
        self.mel_floor = float(fe.mel_floor)
        self.register_buffer("window", torch.tensor(fe.window, dtype=torch.float32), persistent=False)
        self.register_buffer("mel_filters", torch.tensor(fe.mel_filters, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wav: [B, S] float in [-1, 1], zero padded; lengths: [B] valid samples.

        Returns features [B, T, 128] and mask [B, T]. Padded frames hold log(mel_floor), the
        value of digital silence, instead of the extractor's zeros: the encoder is run without a
        padding mask (see CTCModel.forward), so padding must look like silence.
        """
        pad_left = self.frame_length // 2
        wav = nn.functional.pad(wav.float(), (pad_left, 0))
        frames = wav.unfold(-1, self.frame_length + 1, self.hop_length)[..., :-1]
        spec = torch.fft.rfft(frames * self.window, n=self.fft_length).abs()
        feats = torch.log(spec @ self.mel_filters + self.mel_floor)
        # A frame is valid if its last sample (in the padded signal) is a real sample.
        ends = torch.arange(feats.shape[1], device=wav.device) * self.hop_length + self.frame_length
        mask = (ends[None, :] >= pad_left) & (ends[None, :] < (lengths[:, None] + pad_left))
        return feats.masked_fill(~mask[..., None], math.log(self.mel_floor)), mask
