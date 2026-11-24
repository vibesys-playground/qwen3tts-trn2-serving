"""CPU test of paged KV rows: DecodeStep with `pages` on a small random model.

A batch with an S-slot cache; two streams grow past their leader row and borrow free rows of the
batch as pages (one at page_at = S - 1, one at a small page_at as the startup self-test does);
every live row's logits must match a plain DecodeStep with one 64-slot cache (same function: the
FP32 log-sum-exp merge differs from one softmax in rounding only), for both row-gather forms
("gather", "mm") and for P = 1 and P = 3 partner slots. Page rows start with garbage (a fresh page
must be fully masked), and nothing is non-finite.
    python tests/test_pages_cpu.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.model import DecodeStep, page_inputs  # noqa: E402

torch.manual_seed(0)
L, H, nh, kvh, hd, I, Vv = 2, 64, 4, 2, 16, 96, 50
cfg = {"num_hidden_layers": L, "hidden_size": H, "num_attention_heads": nh, "num_key_value_heads": kvh, "head_dim": hd,
       "rms_norm_eps": 1e-6, "rope_theta": 10000.0}
w = {}
for i in range(L):
    p = f"m.layers.{i}."
    w[p + "self_attn.q_proj.weight"] = torch.randn(nh * hd, H) * 0.2
    w[p + "self_attn.k_proj.weight"] = torch.randn(kvh * hd, H) * 0.2
    w[p + "self_attn.v_proj.weight"] = torch.randn(kvh * hd, H) * 0.2
    w[p + "self_attn.o_proj.weight"] = torch.randn(H, nh * hd) * 0.2
    w[p + "self_attn.q_norm.weight"] = torch.ones(hd)
    w[p + "self_attn.k_norm.weight"] = torch.ones(hd)
    w[p + "input_layernorm.weight"] = torch.ones(H)
    w[p + "post_attention_layernorm.weight"] = torch.ones(H)
    w[p + "mlp.gate_proj.weight"] = torch.randn(I, H) * 0.2
    w[p + "mlp.up_proj.weight"] = torch.randn(I, H) * 0.2
    w[p + "mlp.down_proj.weight"] = torch.randn(H, I) * 0.2
w["m.norm.weight"] = torch.ones(H)
SR = 64
st = DecodeStep(w, "m.", cfg, torch.randn(1, Vv, H) * 0.2, SR)


def run(mode, P, S, page_at, steps):
    B = 10
    live = [0, 1, 2]  # rows 0 and 1 grow long (row 1 starts later); row 2 stays short; rows 3..9 free
    start = {0: 0, 1: 4, 2: steps - S + 1}
    Kr = torch.zeros(L, B, kvh, SR, hd, dtype=torch.bfloat16)
    Vr = torch.zeros_like(Kr)
    K = (torch.randn(L, B, kvh, S, hd) * 3).to(torch.bfloat16)  # garbage everywhere not written: must be masked
    V = (torch.randn(L, B, kvh, S, hd) * 3).to(torch.bfloat16)
    xs = torch.randn(steps, B, H).to(torch.bfloat16)
    free = list(range(3, B))
    pg = {0: [], 1: []}
    worst, fails, nsplit = 0.0, [], 0
    for t in range(steps):
        pos = [max(0, t - start[r]) if r in live else 0 for r in range(B)]
        _, lr, Kr, Vr = st(xs[t], torch.tensor(pos), Kr, Vr)
        for ld in pg:
            need = 0 if pos[ld] < page_at else 1 + (pos[ld] - page_at) // (S - 1)
            assert need <= P, (mode, P, t, ld, need)
            while len(pg[ld]) < need:
                pg[ld].append(free.pop(0))
        groups = {ld: list(v) for ld, v in pg.items() if v}
        nsplit += bool(groups)
        ins = page_inputs(pos, groups, P=P, mode=mode, page_at=page_at, S=S)
        _, lt, K, V = st(xs[t], ins[0], K, V, inplace=True, pages=(mode,) + tuple(ins[1:]))
        for r in live:
            if pos[r] == 0 and t < start[r]:
                continue
            if not bool(lt[r].isfinite().all()):
                fails.append(f"{mode} P={P} step {t} row {r}: non-finite logits")
                continue
            d = float((lt[r] - lr[r]).abs().max())
            worst = max(worst, d)
            if d > 0.05:
                fails.append(f"{mode} P={P} S={S} page_at={page_at} step {t} row {r} pos {pos[r]} pages {pg.get(r)}: logits differ by {d:.2e}")
    print(f"{mode} P={P} S={S} page_at={page_at}: {steps} steps ({nsplit} with groups), pages {pg}, worst |logit diff| {worst:.2e}")
    if nsplit < steps // 3:
        fails.append(f"{mode} P={P}: too few paged steps {nsplit}")
    return fails


fails = []
fails += run("gather", 1, 16, 15, 30)   # the serving split (page_at = S - 1): positions up to 29
fails += run("gather", 1, 16, 5, 20)    # a small split (the self-test's): 5 + 15
fails += run("mm", 1, 16, 15, 30)
fails += run("gather", 3, 8, 3, 24)     # 3 pages of 7: positions up to 3 + 21
fails += run("mm", 3, 8, 3, 24)
for f in fails[:20]:
    print("FAIL", f)
sys.exit(1 if fails else 0)
