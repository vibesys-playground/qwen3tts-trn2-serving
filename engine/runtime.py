"""The language-model side of the engine: prompt prefill, frame generation and teacher-forced scoring.

One `LM` holds three compiled graphs on one device: the one-call prompt prefill (engine.model.
PrefillStep, slots 0..9 of the talker cache), the talker step (the prefill self-test's reference)
and the frame step (engine.model.FrameStep: sample code 0, the 16 code-predictor positions with in-graph sampling
of codes 1-15, the frame embedding and the next talker step). Per frame the host makes one copy in
(noise, text slot, forced codes), one launch and one copy out (the frame's 16 codes); everything
else stays on the device. `generate` and `score` run the same frame graph; only the choice of
each code differs (sampled vs given). `LM` runs one sequence at a time (B = 1). In the server
it provides the shared weights and prompt embedder to the continuous batch (engine.batch), the
B = 1 frame graph with a 2048-slot cache that long streams fall back to, and the host BF16
reference path that the replica's startup self-test compares the batched graphs with.
"""

from __future__ import annotations

import time

import torch

from .model import BF, PrefillStep, build_frame, build_steps, load_checkpoint
from .prompt import Embedder
from .sampling import NEG, TEMPERATURE, TOP_K, code0_scores, gumbel_noise, gumbel_pick

MAX_POSITIONS = 2048  # generation stops when prompt ids plus generated frames reach this
TALKER_SLOTS = 2048  # talker KV cache slots (prefill 10 + frames <= 2048 - prompt ids < 2048)
PREFILL_LEN = 10  # talker prompt positions for a CustomVoice speaker with no instruct text (engine/prompt.py)


class LM:
    """Checkpoint, prompt embedder and the B = 1 graphs (prefill, talker step, frame step)."""

    def __init__(self, root: str, log=print):
        """Load the checkpoint and build both step modules on the host (call to_device next)."""
        self.log = log
        t0 = time.time()
        cfg, w = load_checkpoint(root)
        self.cfg = cfg
        tc = cfg["talker_config"]
        self.n_groups = tc["num_code_groups"]
        self.emb = Embedder(cfg, w, root)
        self.eos_id = tc["codec_eos_token_id"]
        self.talker_mod, self.pred_mod = build_steps(cfg, w, TALKER_SLOTS)
        self.prefill_mod = PrefillStep(self.talker_mod, PREFILL_LEN)  # shares the talker's Parameters
        self.frame_mod = build_frame(cfg, w, self.talker_mod, self.pred_mod)
        del w
        self.B = 1
        t, p = self.talker_mod, self.pred_mod
        self.t_shape = (t.L, self.B, t.kvh, TALKER_SLOTS, t.hd)
        self.p_shape = (p.L, self.B, p.kvh, self.n_groups, p.hd)
        self._set_device("cpu", compile_=False)
        log(f"[lm] checkpoint loaded on the host in {time.time() - t0:.1f}s")

    def _set_device(self, device: str, compile_: bool):
        self.device = torch.device(device)
        self.talker_mod = self.talker_mod.to(self.device).eval()
        self.pred_mod = self.pred_mod.to(self.device).eval()
        self.prefill_mod = self.prefill_mod.to(self.device).eval()  # its step is talker_mod (moved above)
        assert self.prefill_mod.step is self.talker_mod
        self.frame_mod = self.frame_mod.to(self.device).eval()  # shares the two step modules
        assert self.frame_mod.talker is self.talker_mod and self.frame_mod.pred is self.pred_mod
        if compile_:
            self.talker = torch.compile(self.talker_mod, backend="neuron", fullgraph=True, dynamic=False)
            self.prefill_graph = torch.compile(self.prefill_mod, backend="neuron", fullgraph=True, dynamic=False)
            self.frame = torch.compile(self.frame_mod, backend="neuron", fullgraph=True, dynamic=False)
        else:
            self.talker, self.prefill_graph, self.frame = self.talker_mod, self.prefill_mod, self.frame_mod
        # Positions as device tensors, made once each (no host->device copy per step).
        self._pos = {}

    def to_device(self, device: str, compile_: bool = True):
        t0 = time.time()
        self._set_device(device, compile_)
        self.log(f"[lm] weights moved to {device} in {time.time() - t0:.1f}s")

    def pos(self, i: int) -> torch.Tensor:
        t = self._pos.get(i)
        if t is None:
            t = self._pos[i] = torch.tensor([i], dtype=torch.long, device=self.device)
        return t

    # ------------------------------------------------------------------ device helpers
    def _dev(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.B, -1).to(self.device)

    def new_talker_cache(self):
        z = torch.zeros(self.t_shape, dtype=BF, device=self.device)
        return [z, torch.zeros(self.t_shape, dtype=BF, device=self.device)]

    def talker_step(self, x, pos: int, cache):
        h, logits, cache[0], cache[1] = self.talker(x, self.pos(pos), cache[0], cache[1])
        return h, logits

    # ------------------------------------------------------------------ frame loop
    def frame_state(self, pos: int):
        """Per-sequence device state of the frame loop after a prefill of `pos` positions:
        [talker position, codes generated so far (presence mask), frames generated so far]."""
        d = self.device
        return [torch.full((self.B,), pos, dtype=torch.long, device=d),
                torch.zeros((self.B, self.frame_mod.V0), dtype=torch.float32, device=d),
                torch.zeros((self.B,), dtype=torch.long, device=d)]

    def host_input(self, trailing: torch.Tensor, step: int, gen=None, forced: list[int] | None = None) -> torch.Tensor:
        """The frame graph's host input [B, width] FP32: noise, text slot, forced codes, flag."""
        f = self.frame_mod
        x = torch.empty((self.B, f.width), dtype=torch.float32)
        if forced is None:
            x[:, : f.o_trail] = gumbel_noise((self.B, f.o_trail), gen)
            x[:, f.o_forced :] = 0.0
        else:
            x[:, : f.o_trail] = 0.0
            x[:, f.o_forced : f.o_forced + self.n_groups] = torch.tensor(forced, dtype=torch.float32)
            x[:, f.o_forced + self.n_groups] = 1.0
        x[:, f.o_trail : f.o_forced] = trailing[step] if step < trailing.shape[0] else self.emb.tts_pad
        return x

    def frame_step(self, h, logits, cache, state, hostin):
        """One frame graph launch: returns (codes [B, 16] on the device, h', logits', sub_logits);
        the talker cache and the frame state are updated in place (lists)."""
        codes, h, logits, state[0], cache[0], cache[1], state[1], state[2], sub = self.frame(
            h, logits, state[0], cache[0], cache[1], state[1], state[2], hostin.to(self.device))
        return codes, h, logits, sub

    def prefill_x(self, x: torch.Tensor):
        """Run the prompt embeddings x [T, H] BF16 (host) through the one-call prefill graph.

        Returns (cache, hidden [B, H], logits [B, vocab]) of the last prompt position."""
        assert x.shape[0] == PREFILL_LEN, f"prefill graph is built for {PREFILL_LEN} prompt positions, got {x.shape[0]}"
        K0, V0 = self.new_talker_cache()
        h, logits, K, V = self.prefill_graph(x.reshape(self.B, PREFILL_LEN, -1).to(self.device), K0, V0)
        return [K, V], h, logits

    def prefill_x_stepwise(self, x: torch.Tensor):
        """The reference path the prefill graph replaces: one decode step per prompt position."""
        cache = self.new_talker_cache()
        h = logits = None
        for i in range(x.shape[0]):
            h, logits = self.talker_step(self._dev(x[i]), i, cache)
        return cache, h, logits

    def prefill(self, text: str, speaker: str, language: str):
        prefill, trailing, n_ids = self.emb.prompt(text, speaker, language)
        cache, h, logits = self.prefill_x(prefill)
        return cache, h, logits, trailing, n_ids, prefill.shape[0]

    # ------------------------------------------------------------------ requests
    @torch.inference_mode()
    def generate(self, text: str, speaker: str, language: str, cancelled=lambda: False, gen=None):
        """Yield frames ([16] ints) as they are generated; stops at EOS or when prompt ids plus
        generated frames reach MAX_POSITIONS."""
        cache, h, logits, trailing, n_ids, pos = self.prefill(text, speaker, language)
        state = self.frame_state(pos)
        for t in range(MAX_POSITIONS - n_ids):
            if cancelled():
                return
            codes, h, logits, _ = self.frame_step(h, logits, cache, state, self.host_input(trailing, t, gen))
            codes = codes[0].cpu().tolist()  # the one device->host sync per frame
            if codes[0] == self.eos_id:
                return
            yield codes

    @torch.inference_mode()
    def score(self, text: str, speaker: str, language: str, codes: list[list[int]]):
        """Teacher-forced raw-logit log-probabilities of `codes` [T][16] and of EOS after them."""
        cache, h, logits, trailing, _, pos = self.prefill(text, speaker, language)
        state = self.frame_state(pos)
        out = []
        for t, fr in enumerate(codes):
            fr = [int(c) for c in fr]
            lp0 = torch.log_softmax(logits[0].float().cpu(), -1)
            _, h, logits, sub = self.frame_step(h, logits, cache, state, self.host_input(trailing, t, forced=fr))
            lps = torch.log_softmax(sub[0].float().cpu(), -1)
            out.append([float(lp0[fr[0]])] + [float(lps[g - 1, fr[g]]) for g in range(1, self.n_groups)])
        eos = float(torch.log_softmax(logits[0].float().cpu(), -1)[self.eos_id])
        return out, eos

    @torch.inference_mode()
    def selftest_sampler(self, text: str, n_frames: int, gen=None, tol: float = 1e-3):
        """Check the in-graph sampler against a host Gumbel-max on the graph's own logits and noise.

        Runs `n_frames` frames of generation (ignoring EOS) and recomputes every code on the host
        from the frame's input logits, presence and frame count (code 0) and its sub_logits
        (codes 1-15) with the same noise. Returns (checked, mismatches); a mismatch whose host
        winner leads the runner-up by less than `tol` (a rounding tie) is not counted."""
        f = self.frame_mod
        cache, h, logits, trailing, n_ids, pos = self.prefill(text, "Ryan", "English")
        state = self.frame_state(pos)
        checked, bad = 0, []
        sup = f.suppress.cpu()

        def host(scores, noise):
            c = int(gumbel_pick(scores, noise)[0])
            s = scores.float() / TEMPERATURE
            kth = torch.topk(s, TOP_K, dim=-1).values[..., -1:]
            v = torch.where(s >= kth, s, NEG) + noise
            top2 = torch.topk(v[0], 2).values
            return c, float(top2[0] - top2[1])

        for t in range(n_frames):
            l_in, p_in, n_in = logits.cpu(), state[1].cpu(), state[2].cpu()
            hostin = self.host_input(trailing, t, gen)
            codes, h, logits, sub = self.frame_step(h, logits, cache, state, hostin)
            codes, sub = codes.cpu(), sub.cpu()
            want = [host(code0_scores(l_in, p_in, n_in, sup, self.eos_id), hostin[:, : f.V0])]
            for g in range(1, self.n_groups):
                o = f.o_g1 + (g - 1) * f.V1
                want.append(host(sub[:, g - 1], hostin[:, o : o + f.V1]))
            for g, (c, margin) in enumerate(want):
                checked += 1
                if int(codes[0, g]) != c and margin >= tol:
                    bad.append((t, g, int(codes[0, g]), c, margin))
        return checked, bad

