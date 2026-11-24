"""The streaming TTS client shared by the check and the benchmark (trusted task code).

One request is `POST {url}/generate` with a JSON body; the answer is HTTP 200 and a body of raw
16-bit little-endian mono PCM at 24 kHz, streamed as it is generated, ending when the utterance
ends. The client timestamps every network chunk on a monotonic clock and models a player that
starts at the first audio byte and pauses whenever the next audio has not arrived (VoxServe v2,
Appendix B).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import aiohttp

SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2
SPEAKER = "Ryan"
LANGUAGE = "English"
TTFA_SLO_S = 0.200  # TTFA target T
STALL_SLO_S = 0.010  # a request stalls if its player's pauses add up to 10 ms or more
REQUEST_TIMEOUT_S = 900.0


@dataclass
class Result:
    text: str
    sent: float  # monotonic seconds
    status: int | None = None
    ttfa: float | None = None
    audio_bytes: int = 0
    pause: float = 0.0
    done: float | None = None
    complete: bool = False
    error: str | None = None
    pcm: bytearray | None = None
    arrivals: list[tuple[float, int]] = field(default_factory=list)

    @property
    def audio_seconds(self) -> float:
        return self.audio_bytes / BYTES_PER_SECOND

    @property
    def meets_slo(self) -> bool:
        return (
            self.complete
            and self.audio_bytes > 0
            and self.ttfa is not None
            and self.ttfa <= TTFA_SLO_S
            and self.pause < STALL_SLO_S
        )

    def summary(self) -> dict:
        return {
            "status": self.status,
            "ttfa": self.ttfa,
            "audio_s": round(self.audio_seconds, 4),
            "pause": round(self.pause, 4),
            "latency": None if self.done is None else round(self.done - self.sent, 4),
            "complete": self.complete,
            "slo": self.meets_slo,
            "chars": len(self.text),
            "error": self.error,
        }


def body(text: str, **extra) -> dict:
    return {"text": text, "speaker": SPEAKER, "language": LANGUAGE, "stream": True, **extra}


async def request(
    session: aiohttp.ClientSession,
    url: str,
    res: Result,
    *,
    keep_pcm: bool = False,
    **extra,
) -> Result:
    """One streamed request, filled into `res` (so a cancelled request keeps what it got).

    `res.sent` is the request's scheduled send time: on an absolute schedule a late send counts
    against TTFA."""
    text = res.text
    if keep_pcm:
        res.pcm = bytearray()
    play_end = 0.0
    try:
        async with session.post(f"{url}/generate", json=body(text, **extra)) as resp:
            res.status = resp.status
            if resp.status != 200:
                res.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return res
            async for chunk in resp.content.iter_any():
                now = time.monotonic()
                if not chunk:
                    continue
                n = len(chunk)
                if res.ttfa is None:
                    res.ttfa = now - res.sent
                    play_end = now
                elif now > play_end:
                    res.pause += now - play_end
                    play_end = now
                play_end += n / BYTES_PER_SECOND
                res.audio_bytes += n
                res.arrivals.append((now, n))
                if keep_pcm:
                    res.pcm.extend(chunk)
            res.complete = True
            if res.audio_bytes % 2:
                res.complete = False
                res.error = f"odd number of PCM bytes ({res.audio_bytes})"
    except asyncio.CancelledError:
        res.complete = False
        res.error = "timed out"
        raise
    except Exception as e:  # noqa: BLE001 - any transport failure is a miss
        res.error = f"{type(e).__name__}: {e}"[:300]
    finally:
        res.done = time.monotonic()
    return res


def session(limit: int = 0) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=limit, limit_per_host=limit, force_close=False),
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=REQUEST_TIMEOUT_S),
    )
