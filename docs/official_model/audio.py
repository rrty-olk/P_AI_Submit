"""Speech to prefix embeddings: a log-mel front end, a 17-layer FastConformer, an MLP adapter and a residual.

16 kHz mono audio (up to 30 s; clips under 0.5 s are padded) -> 128 normalised log-mel features every 10 ms ->
8x subsampling -> conformer -> adapter 512 -> 1024 -> residual correction. One prefix embedding per 80 ms.
The conformer and adapter come from LFM2-Audio (NVIDIA Canary FastConformer lineage), with NeMo's module names.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE, MIN_SAMPLES, MAX_SECONDS = 16000, 8000, 30


def waveform(audio) -> torch.Tensor:
    """Mono 16 kHz int16 PCM or float samples in [-1, 1] -> float32 (1, N), cut to 30 s, padded to 0.5 s."""
    x = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
    if x.ndim != 1:
        raise ValueError("audio must be mono: a 1-D array of 16 kHz samples")
    x = x[: MAX_SECONDS * SAMPLE_RATE]
    x = x.astype(np.float32) / np.float32(32768.0) if x.dtype == np.int16 else x.astype(np.float32)
    if len(x) < MIN_SAMPLES:
        x = np.pad(x, (0, MIN_SAMPLES - len(x)))
    return torch.from_numpy(x)[None]


def slaney_filterbank(sr: int = SAMPLE_RATE, n_fft: int = 512, n_mels: int = 128) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels, norm="slaney") in float32, computed as librosa computes it."""
    f_sp, min_log_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp

    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)[()]
        return min_log_mel + np.log(f / min_log_hz) / logstep if f >= min_log_hz else f / f_sp

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        f = f_sp * m
        high = m >= min_log_mel
        f[high] = min_log_hz * np.exp(logstep * (m[high] - min_log_mel))
        return f

    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, np.fft.rfftfreq(n=n_fft, d=1.0 / sr))
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return weights


class MelFrontend(nn.Module):
    """NeMo's FilterbankFeatures in eval mode: preemphasis, |STFT|^2, Slaney mel, log, per-feature norm.
    Always float32, whatever the model's dtype."""

    def __init__(self, features: int = 128, n_fft: int = 512, window: int = 400, hop: int = 160):
        super().__init__()
        self.n_fft, self.window, self.hop, self.features = n_fft, window, hop, features
        self._fb = None  # built on first use, never a buffer, so it is never cast or put on the meta device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        n = torch.tensor([x.shape[1]], device=x.device)
        frames = torch.floor_divide(n + self.n_fft // 2 * 2 - self.n_fft, self.hop)
        x = torch.cat((x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]), dim=1)
        window = torch.hann_window(self.window, periodic=False).to(x.device)  # built on the CPU, as NeMo does
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.window, center=True,
                       window=window, return_complex=True, pad_mode="constant")
        x = torch.sqrt(torch.view_as_real(x).pow(2).sum(-1)).pow(2.0)
        if self._fb is None or self._fb.device != x.device:
            self._fb = torch.from_numpy(slaney_filterbank(n_fft=self.n_fft, n_mels=self.features))[None].to(x.device)
        x = torch.log(torch.matmul(self._fb, x) + 2**-24)
        valid = torch.arange(x.shape[2], device=x.device)[None] < frames[:, None]
        count = valid.sum(1)
        mean = torch.where(valid[:, None], x, 0.0).sum(2) / count[:, None]
        std = torch.sqrt(torch.where(valid[:, None], x - mean[:, :, None], 0.0).pow(2).sum(2) / (count[:, None] - 1.0))
        x = (x - mean[:, :, None]) / (std.masked_fill(std.isnan(), 0.0) + 1e-5)[:, :, None]
        return x.masked_fill(~valid[:, None], 0.0), frames


class ConvSubsampling(nn.Module):
    """8x time subsampling: a 3x3 stride-2 conv, then two depthwise-separable stride-2 convs, then a linear."""

    def __init__(self, feat_in: int, channels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 2, 1, groups=channels), nn.Conv2d(channels, channels, 1), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 2, 1, groups=channels), nn.Conv2d(channels, channels, 1), nn.ReLU(True))
        freq = feat_in
        for _ in range(3):
            freq = (freq - 1) // 2 + 1
        self.out = nn.Linear(channels * freq, d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.unsqueeze(1)
        lengths = lengths.float()
        for layer in self.conv:
            x = x * _time_mask(x, lengths)
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
                lengths = torch.div(lengths + 2 - 3, 2, rounding_mode="floor") + 1
        x = x * _time_mask(x, lengths)
        b, c, t, f = x.shape
        return self.out(x.transpose(1, 2).reshape(b, t, c * f)), lengths.long()


def _time_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    t = torch.arange(x.shape[2], device=x.device)[None] < lengths.long()[:, None]
    return t[:, None, :, None].to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))


class RelPositionAttention(nn.Module):
    """Transformer-XL relative-position attention with per-layer biases u and v."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.d_k = heads, d // heads
        self.linear_q, self.linear_k, self.linear_v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.linear_out = nn.Linear(d, d)
        self.linear_pos = nn.Linear(d, d, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(heads, self.d_k))

    def forward(self, x, pos_emb, mask):
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, t, self.h, self.d_k)
        k = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k).transpose(1, 2)
        ac = torch.matmul((q + self.pos_bias_u).transpose(1, 2), k.transpose(-2, -1))
        bd = torch.matmul((q + self.pos_bias_v).transpose(1, 2), p.transpose(-2, -1))
        bh, hh, qlen, pos_len = bd.shape
        bd = F.pad(bd, (1, 0)).view(bh, hh, pos_len + 1, qlen)[:, :, 1:].view(bh, hh, qlen, pos_len)
        scores = (ac + bd[:, :, :, :t]) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask[:, None], -10000.0)
        attn = torch.softmax(scores, dim=-1).masked_fill(mask[:, None], 0.0)
        return self.linear_out(torch.matmul(attn, v).transpose(1, 2).reshape(b, t, -1))


class ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, 1)
        self.depthwise_conv = nn.Conv1d(d, d, kernel, groups=d)
        self.batch_norm = nn.BatchNorm1d(d)
        self.pointwise_conv2 = nn.Conv1d(d, d, 1)
        self.pad = (kernel - 1) // 2

    def forward(self, x, pad_mask):
        x = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        x = self.depthwise_conv(F.pad(x.masked_fill(pad_mask[:, None], 0.0), (self.pad, self.pad)))
        return self.pointwise_conv2(F.silu(self.batch_norm(x))).transpose(1, 2)


class ConformerLayer(nn.Module):
    def __init__(self, d: int, d_ff: int, heads: int, kernel: int):
        super().__init__()
        self.norm_feed_forward1, self.feed_forward1 = nn.LayerNorm(d), FeedForward(d, d_ff)
        self.norm_self_att, self.self_attn = nn.LayerNorm(d), RelPositionAttention(d, heads)
        self.norm_conv, self.conv = nn.LayerNorm(d), ConvModule(d, kernel)
        self.norm_feed_forward2, self.feed_forward2 = nn.LayerNorm(d), FeedForward(d, d_ff)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x, pos_emb, att_mask, pad_mask):
        x = x + self.feed_forward1(self.norm_feed_forward1(x)) * 0.5
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, att_mask)
        x = x + self.conv(self.norm_conv(x), pad_mask)
        x = x + self.feed_forward2(self.norm_feed_forward2(x)) * 0.5
        return self.norm_out(x)


class Conformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.d_model = d
        self.pre_encode = ConvSubsampling(cfg["feat_in"], cfg["subsampling_conv_channels"], d)
        self.layers = nn.ModuleList(ConformerLayer(d, d * cfg["ff_expansion_factor"], cfg["n_heads"],
                                                   cfg["conv_kernel_size"]) for _ in range(cfg["n_layers"]))

    def pos_emb(self, t: int, device) -> torch.Tensor:
        """Sinusoids for relative positions t-1 .. -(t-1), built on the CPU, as NeMo does."""
        positions = torch.arange(t - 1, -t, -1, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(len(positions), self.d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * div), torch.cos(positions * div)
        return pe[None].to(device)

    def forward(self, mel: torch.Tensor, lengths: torch.Tensor):
        x, lengths = self.pre_encode(mel.transpose(1, 2), lengths)
        t = x.shape[1]
        valid = torch.arange(t, device=x.device)[None] < lengths[:, None]
        att_mask = ~(valid[:, None, :] & valid[:, :, None])
        pos_emb = self.pos_emb(t, x.device).to(x.dtype)
        for layer in self.layers:
            x = layer(x, pos_emb, att_mask, ~valid)
        return x, lengths


class Adapter(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.norm, self.linear_1, self.linear_2 = nn.LayerNorm(d_in), nn.Linear(d_in, d_out), nn.Linear(d_out, d_out)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(self.norm(x))))


class Residual(nn.Module):
    """x + up(GELU(down(LN(x)))), trained with the trunk to fit the adapter's output to it."""

    def __init__(self, d: int, width: int):
        super().__init__()
        self.ln, self.down, self.up = nn.LayerNorm(d), nn.Linear(d, width), nn.Linear(width, d)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.ln(x))))


class Audio(nn.Module):
    def __init__(self, cfg: dict, out: int):
        super().__init__()
        self.frontend = MelFrontend(cfg["feat_in"])
        self.encoder = Conformer(cfg)
        self.adapter = Adapter(cfg["d_model"], out)
        self.residual = Residual(out, cfg["residual_width"])

    def forward(self, audio) -> torch.Tensor:
        """One clip -> (1, P, D) prefix embeddings."""
        param = next(self.encoder.parameters())
        mel, frames = self.frontend(waveform(audio).to(param.device))
        x, lengths = self.encoder(mel.to(param.dtype), frames)
        return self.residual(self.adapter(x[:, : int(lengths[0])]))
