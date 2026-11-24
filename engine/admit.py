"""Admission control for text /generate, in front of the codec.

Two gates: DeadlineGate (the default, QWEN3TTS_ADMIT_MODE=deadline) and AdmissionGate, an older
codec frame token bucket (QWEN3TTS_ADMIT_MODE=bucket). Both are length-neutral. The rest of this
docstring describes the token bucket; DeadlineGate has its own.

The codec is the ceiling resource: once offered frames/s pass its capacity C, the EDF codec
scheduler makes every stream late (no request meets the SLO). The gate keeps the admitted frame
rate under rho x C and answers the rest with 503 at once (a rejection is a miss, but it costs no
LM or codec work, so the admitted requests stay on time).

A token bucket of codec frames: it refills at rho x fps_cap tokens/s (fps_cap: the codec's live
measured frames per busy second, CodecService.fps_cap) up to a depth of DEPTH_S x rho x fps_cap.
Admitting a request charges est = ceil(chars x f) frames (f: a decayed ratio of actual frames to
characters over finished requests, seeded from the model's speech rate); when it finishes, the
estimate is replaced by the actual count (refund or charge est - actual). A request is rejected
only when tokens < THRESH (a fixed number of frames, the same for every request) AND the codec's
backlog (frames pushed, not yet decoded) exceeds BACKLOG_S x fps_cap AND at least NMIN codec
streams are open. The decision is length-neutral: it reads only the gate's and the codec's state,
never the request's text, length or estimated cost, which is charged only after admission (so the
bucket may go into debt, refilled before the next admission). Admission is FIFO by arrival.
"""

from __future__ import annotations

import math
import os
import threading
import time

F0 = 12.5 * 0.08  # frames per character: 12.5 Hz x ~0.075-0.08 s of speech per character (Ryan)


class AdmissionGate:
    """The codec frame token bucket (QWEN3TTS_ADMIT_MODE=bucket); see the module docstring."""

    def __init__(self, rho: float | None = None, nmin: int | None = None, depth_s: float = 1.5,
                 backlog_s: float = 0.25, thresh: float = 0.0, f0: float = F0, f_decay: float = 0.98, clock=time.monotonic):
        self.rho = float(os.environ.get("QWEN3TTS_ADMIT_RHO", "0.9")) if rho is None else rho
        self.nmin = int(os.environ.get("QWEN3TTS_ADMIT_NMIN", "64")) if nmin is None else nmin
        self.depth_s, self.backlog_s, self.clock = depth_s, backlog_s, clock
        self.thresh = thresh  # frames of budget needed to admit anyone; never depends on the request
        self.lock = threading.Lock()
        self.tokens: float | None = None  # starts full on the first call
        self.t = clock()
        # f = num / den, both decayed by f_decay per finished request (~50-request memory),
        # seeded with the weight of 20 requests of 100 characters at F0
        self.f_decay = f_decay
        self.f_num, self.f_den = 20 * 100 * f0, 20 * 100.0
        self.admitted = self.rejected = 0

    @property
    def f(self) -> float:
        return self.f_num / self.f_den

    def _refill(self, fps_cap: float) -> None:
        now = self.clock()
        depth = self.depth_s * self.rho * fps_cap
        if self.tokens is None:
            self.tokens = depth
        else:
            self.tokens = min(depth, self.tokens + self.rho * fps_cap * max(0.0, now - self.t))
            self.tokens = max(self.tokens, -4 * depth)  # bound the debt a burst below NMIN leaves
        self.t = now

    def admit(self, chars: int, fps_cap: float, backlog: float, open_streams: int):
        """(admitted, est): est frames are charged when admitted; pass est to finish()."""
        with self.lock:
            self._refill(fps_cap)
            # length-neutral: the decision never reads `chars` or est
            if (open_streams >= self.nmin and self.tokens < self.thresh and backlog > self.backlog_s * fps_cap):
                self.rejected += 1
                return False, 0
            est = max(2, math.ceil(chars * self.f))  # charged only once admitted
            self.tokens -= est
            self.admitted += 1
            return True, est

    def finish(self, est: int, chars: int, frames: int, completed: bool) -> None:
        """A request admitted with `est` ended after `frames` frames; a completed one updates f."""
        with self.lock:
            if self.tokens is not None:
                self.tokens += est - frames
            if completed and chars > 0 and frames > 0:
                self.f_num = self.f_decay * self.f_num + frames
                self.f_den = self.f_decay * self.f_den + chars

    def stats(self, fps_cap: float) -> str:
        with self.lock:
            tok = self.tokens if self.tokens is not None else float("nan")
            return (f"admitted {self.admitted} rejected {self.rejected} tokens {tok:.0f} "
                    f"fps_cap {fps_cap:.0f} f {self.f:.3f} rho {self.rho} nmin {self.nmin}")


class DeadlineGate:
    """Deadline-feasibility admission (the default gate). Admit a text /generate iff the codec can deliver
    every frame due within the next H seconds at its current measured cost per call:

        reject iff open_streams >= NMIN and urgent(H) + ALLOW > U x cap_H

    urgent(H): CodecService.urgent, the frames every open stream (including streams still in
    prefill, due from arrival + TTFA) needs within H: those whose deadline falls before now + H, and
    at least its playback rate x H for a stream ahead of its player (it needs them once its buffer
    runs down: an H-horizon count alone lets ahead-of-time streams hide committed work, and a
    simulation then admits ~96% of capacity, 45% of frames late). Late frames all count, so a
    backlog that has fallen behind closes the gate. cap_H = H x 64 / per_call(64), where
    per_call(64) comes from a least-squares fit of call cycle time against active slots over the
    last 64 calls (CodecService.call_full_s), so a host slowdown cuts capacity within ~2 s. ALLOW is
    the arriving stream's own due frames within H (first at TTFA, then one per frame), a constant:
    the decision never reads the request's text, its length or any estimate from them."""

    def __init__(self, h: float | None = None, u: float | None = None, nmin: int | None = None,
                 ttfa_s: float = 0.2, frame_s: float = 0.08):
        self.h = float(os.environ.get("QWEN3TTS_ADMIT_H", "0.5")) if h is None else h
        # U = 0.8, measured on trn2 in a Poisson A/B at 30 req/s offered: SLO-met goodput 22.08 req/s
        # vs 22.12 at U = 0.9 (the token bucket: 18.79), but attainment among admitted requests 99.2%
        # vs 97.5% and half the codec backlog (p50 1146 vs 2239 frames), so more margin at the
        # sweep's higher rates
        self.u = float(os.environ.get("QWEN3TTS_ADMIT_U", "0.8")) if u is None else u
        self.nmin = int(os.environ.get("QWEN3TTS_ADMIT_NMIN", "64")) if nmin is None else nmin
        # the arriving stream's frames within H: its playback rate, or its first frame at TTFA and
        # one per frame after it, whichever is more; the same constant for every request
        self.allow = max(self.h / frame_s, 1 + max(0, math.floor((self.h - ttfa_s) / frame_s + 1e-9)))
        self.lock = threading.Lock()
        self.admitted = self.rejected = 0
        self.last = (0, 0.0, 0.0)  # (urgent, cap_H, per_call) of the last gated decision

    def decide(self, svc) -> bool:
        """svc: the CodecService. Reads only its state (streams, deadlines, call timings)."""
        with self.lock:
            if len(svc.streams) < self.nmin:
                self.admitted += 1
                return True
            per_call = svc.call_full_s()[0]
            cap = self.h * svc.max_groups * 8 / per_call
            urg = svc.urgent(self.h)
            self.last = (urg, cap, per_call)
            if urg + self.allow > self.u * cap:
                self.rejected += 1
                return False
            self.admitted += 1
            return True

    def stats(self, svc) -> str:
        per_call, a, b, n = svc.call_full_s()
        cap = self.h * svc.max_groups * 8 / per_call
        tot = self.admitted + self.rejected
        return (f"mode deadline admitted {self.admitted} rejected {self.rejected} "
                f"(reject {100 * self.rejected / max(tot, 1):.1f}%) urgent {svc.urgent(self.h):.0f} cap_H {cap:.0f} "
                f"per_call64 {1000 * per_call:.2f} ms a {1000 * a:.2f} ms b {1000 * b:.3f} ms/slot (n {n}) "
                f"H {self.h} U {self.u} allow {self.allow} nmin {self.nmin}")
