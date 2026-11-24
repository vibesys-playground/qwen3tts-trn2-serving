"""Streaming decoder of the Qwen3-TTS-Tokenizer-12Hz codec (codes -> 24 kHz audio), on the host CPU.

A port of `Qwen3TTSTokenizerV2Decoder` (qwen-tts 0.1.1) that keeps every causal state between
calls, so pushing the codes of an utterance in any chunking gives the same samples as one full
decode: causal convolutions keep their left context, the transposed convolutions keep the previous
input frame, and the pre-transformer keeps a key/value cache of its 72-frame sliding window with
absolute rotary positions. FP32, as the reference decode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

SAMPLES_PER_FRAME = 1920


class CodecWeights:
    """The decoder's weights (FP32) and its config, loaded once and shared by every stream."""

    def __init__(self, root: str | Path):
        root = Path(root) / "speech_tokenizer"
        cfg = json.loads((root / "config.json").read_text())["decoder_config"]
        self.cfg = cfg
        w = {}
        with safe_open(str(root / "model.safetensors"), "pt") as f:
            for k in f.keys():
                if k.startswith("decoder."):
                    w[k[len("decoder.") :]] = f.get_tensor(k).float().contiguous()
        self.w = w
        nq = cfg["num_quantizers"]
        # Codebook embeddings: embedding_sum / clamp(cluster_usage, eps).
        books = []
        for i in range(nq):
            pre = f"quantizer.rvq_first.vq.layers.{i}." if i == 0 else f"quantizer.rvq_rest.vq.layers.{i - 1}."
            emb = w[pre + "_codebook.embedding_sum"] / w[pre + "_codebook.cluster_usage"].clamp(min=1e-5)[:, None]
            books.append(emb)
        self.books = books
        self.first_out = w["quantizer.rvq_first.output_proj.weight"]  # [512, 256, 1]
        self.rest_out = w["quantizer.rvq_rest.output_proj.weight"]
        hd = cfg["head_dim"]
        self.inv_freq = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))


class _CausalConv:
    """Conv1d with left padding (kernel-1)*dilation, stride 1, carrying the left context."""

    def __init__(self, weight, bias, dilation=1, groups=1):
        self.w, self.b, self.d, self.g = weight, bias, dilation, groups
        self.ctx = (weight.shape[-1] - 1) * dilation
        self.state = None

    def __call__(self, x):
        if self.ctx == 0:
            return F.conv1d(x, self.w, self.b, dilation=self.d, groups=self.g)
        if self.state is None:
            self.state = x.new_zeros(x.shape[0], x.shape[1], self.ctx)
        full = torch.cat([self.state, x], -1)
        self.state = full[..., -self.ctx :]
        return F.conv1d(full, self.w, self.b, dilation=self.d, groups=self.g)


class _CausalTransConv:
    """ConvTranspose1d(kernel K, stride r) trimmed by K - r on the right, carrying one input frame."""

    def __init__(self, weight, bias, stride):
        self.w, self.b, self.r = weight, bias, stride
        self.k = weight.shape[-1]
        assert self.k in (self.r, 2 * self.r)
        self.prev = None

    def __call__(self, x):
        n = x.shape[-1]
        if self.k == self.r:
            return F.conv_transpose1d(x, self.w, self.b, stride=self.r)
        if self.prev is None:
            self.prev = x.new_zeros(x.shape[0], x.shape[1], 1)
        full = torch.cat([self.prev, x], -1)
        self.prev = x[..., -1:]
        y = F.conv_transpose1d(full, self.w, self.b, stride=self.r)
        return y[..., self.r : self.r + n * self.r]


def _snake(x, alpha, beta):
    a = torch.exp(alpha)[None, :, None]
    b = torch.exp(beta)[None, :, None]
    return x + (1.0 / (b + 1e-9)) * torch.pow(torch.sin(x * a), 2)


def _rms(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


class StreamDecoder:
    """One utterance's streaming decode: `push(codes [n, 16])` returns its n * 1920 samples."""

    def __init__(self, cw: CodecWeights):
        self.cw = cw
        w, cfg = cw.w, cw.cfg
        self.pos = 0
        self.pre_conv = _CausalConv(w["pre_conv.conv.weight"], w["pre_conv.conv.bias"])
        self.layers = cfg["num_hidden_layers"]
        self.window = cfg["sliding_window"]
        self.kv = [None] * self.layers  # (K, V) [B, heads, <= window-1, hd]
        self.upsample = []
        for i in range(len(cfg["upsampling_ratios"])):
            p = f"upsample.{i}."
            r = cfg["upsampling_ratios"][i]
            self.upsample.append(
                (
                    _CausalTransConv(w[p + "0.conv.weight"], w[p + "0.conv.bias"], r),
                    _CausalConv(w[p + "1.dwconv.conv.weight"], w[p + "1.dwconv.conv.bias"], groups=w[p + "1.dwconv.conv.weight"].shape[0]),
                    p + "1.",
                )
            )
        self.conv_in = _CausalConv(w["decoder.0.conv.weight"], w["decoder.0.conv.bias"])
        self.blocks = []
        for i in range(len(cfg["upsample_rates"])):
            p = f"decoder.{i + 1}.block."
            r = cfg["upsample_rates"][i]
            units = []
            for j, dil in enumerate((1, 3, 9)):
                q = f"{p}{j + 2}."
                units.append(
                    (
                        q,
                        _CausalConv(w[q + "conv1.conv.weight"], w[q + "conv1.conv.bias"], dilation=dil),
                        _CausalConv(w[q + "conv2.conv.weight"], w[q + "conv2.conv.bias"]),
                    )
                )
            self.blocks.append((p, _CausalTransConv(w[p + "1.conv.weight"], w[p + "1.conv.bias"], r), units))
        n = len(cfg["upsample_rates"]) + 1
        self.final_act = f"decoder.{n}."
        self.conv_out = _CausalConv(w[f"decoder.{n + 1}.conv.weight"], w[f"decoder.{n + 1}.conv.bias"])

    def _quantize(self, codes):  # codes [B, 16, T] -> [B, 512, T]
        cw = self.cw
        first = F.embedding(codes[:, 0], cw.books[0]).transpose(1, 2)
        rest = None
        for i in range(1, len(cw.books)):
            e = F.embedding(codes[:, i], cw.books[i]).transpose(1, 2)
            rest = e if rest is None else rest + e
        return F.conv1d(first, cw.first_out) + F.conv1d(rest, cw.rest_out)

    def _transformer(self, x):  # x [B, T, 1024]
        w, cfg = self.cw.w, self.cw.cfg
        B, T, _ = x.shape
        nh, hd, eps = cfg["num_attention_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
        x = F.linear(x, w["pre_transformer.input_proj.weight"], w["pre_transformer.input_proj.bias"])
        pos = torch.arange(self.pos, self.pos + T, dtype=torch.float32)
        freqs = pos[:, None] * self.cw.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), -1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        for i in range(self.layers):
            p = f"pre_transformer.layers.{i}."
            r = x
            h = _rms(x, w[p + "input_layernorm.weight"], eps)
            q = F.linear(h, w[p + "self_attn.q_proj.weight"]).view(B, T, nh, hd).transpose(1, 2)
            k = F.linear(h, w[p + "self_attn.k_proj.weight"]).view(B, T, nh, hd).transpose(1, 2)
            v = F.linear(h, w[p + "self_attn.v_proj.weight"]).view(B, T, nh, hd).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            if self.kv[i] is not None:
                K = torch.cat([self.kv[i][0], k], 2)
                V = torch.cat([self.kv[i][1], v], 2)
            else:
                K, V = k, v
            P = K.shape[2] - T  # cached positions before this chunk
            keep = self.window - 1
            self.kv[i] = (K[:, :, -keep:], V[:, :, -keep:])
            qpos = torch.arange(T)[:, None] + P
            kpos = torch.arange(K.shape[2])[None, :]
            allowed = (kpos <= qpos) & (kpos > qpos - self.window)
            mask = torch.zeros(T, K.shape[2]).masked_fill(~allowed, torch.finfo(torch.float32).min)
            s = torch.matmul(q, K.transpose(2, 3)) * (hd**-0.5) + mask
            a = torch.softmax(s, -1, dtype=torch.float32)
            o = torch.matmul(a, V).transpose(1, 2).reshape(B, T, nh * hd)
            o = F.linear(o, w[p + "self_attn.o_proj.weight"])
            x = r + w[p + "self_attn_layer_scale.scale"] * o
            r = x
            h = _rms(x, w[p + "post_attention_layernorm.weight"], eps)
            m = F.linear(F.silu(F.linear(h, w[p + "mlp.gate_proj.weight"])) * F.linear(h, w[p + "mlp.up_proj.weight"]), w[p + "mlp.down_proj.weight"])
            x = r + w[p + "mlp_layer_scale.scale"] * m
        x = _rms(x, w["pre_transformer.norm.weight"], eps)
        return F.linear(x, w["pre_transformer.output_proj.weight"], w["pre_transformer.output_proj.bias"])

    @torch.inference_mode()
    def push(self, codes) -> np.ndarray:
        """Decode the next frames: codes [n, 16] ints -> n * 1920 float32 samples in [-1, 1]."""
        w = self.cw.w
        c = torch.as_tensor(np.asarray(codes), dtype=torch.long).clamp(min=0)
        if c.numel() == 0:
            return np.zeros(0, dtype=np.float32)
        c = c.T.unsqueeze(0)  # [1, 16, n]
        n = c.shape[-1]
        h = self._quantize(c)
        h = self.pre_conv(h).transpose(1, 2)
        h = self._transformer(h).permute(0, 2, 1)
        self.pos += n
        for tconv, dw, p in self.upsample:
            h = tconv(h)
            res = h
            h = dw(h).permute(0, 2, 1)
            h = F.layer_norm(h, (h.shape[-1],), w[p + "norm.weight"], w[p + "norm.bias"], eps=1e-6)
            h = F.linear(h, w[p + "pwconv1.weight"], w[p + "pwconv1.bias"])
            h = F.gelu(h)
            h = F.linear(h, w[p + "pwconv2.weight"], w[p + "pwconv2.bias"])
            h = (w[p + "gamma"] * h).permute(0, 2, 1)
            h = res + h
        h = self.conv_in(h)
        for p, tconv, units in self.blocks:
            h = _snake(h, w[p + "0.alpha"], w[p + "0.beta"])
            h = tconv(h)
            for q, c1, c2 in units:
                r = h
                y = _snake(h, w[q + "act1.alpha"], w[q + "act1.beta"])
                y = c1(y)
                y = _snake(y, w[q + "act2.alpha"], w[q + "act2.beta"])
                y = c2(y)
                h = y + r
        h = _snake(h, w[self.final_act + "alpha"], w[self.final_act + "beta"])
        h = self.conv_out(h).clamp(min=-1, max=1)
        out = h[0, 0].numpy()
        assert out.shape[0] == n * SAMPLES_PER_FRAME, (out.shape, n)
        return out


def to_pcm16(wav: np.ndarray) -> bytes:
    w = np.asarray(wav, dtype=np.float32)
    return np.clip(np.round(w * 32767), -32768, 32767).astype("<i2").tobytes()

