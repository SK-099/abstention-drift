"""Audio overview: speak a two-host script with Kokoro (Kokoro-FastAPI, OpenAI-compatible /v1/audio/speech) and
join the turns into one WAV file with short pauses between speakers."""

from __future__ import annotations

import io
import wave
from pathlib import Path

from .settings import Settings


class TTSUnavailable(RuntimeError):
    pass


def speak(text: str, voice: str, s: Settings) -> bytes:
    import httpx

    if not s.tts_url:
        raise TTSUnavailable(
            "the voice service is not configured: start it with `docker compose --profile tts up -d kokoro` "
            "and set TTS_URL=http://kokoro:8880"
        )
    try:
        r = httpx.post(
            f"{s.tts_url.rstrip('/')}/v1/audio/speech",
            # raw 16-bit mono PCM (Kokoro: 24 kHz); a streamed WAV header can carry placeholder sizes
            json={
                "model": "kokoro",
                "input": text,
                "voice": voice,
                "response_format": "pcm",
                "speed": s.tts_speed,
                "stream": False,
            },
            timeout=s.tts_timeout_s,
        )
    except httpx.HTTPError as e:
        raise TTSUnavailable(
            f"the voice service ({s.tts_url}) is not running. Start it with "
            "`docker compose --profile tts up -d kokoro`, then press Speak again."
        ) from e
    r.raise_for_status()
    return r.content


def join_wavs(parts: list[bytes], pause_s: float = 0.35) -> tuple[bytes, float]:
    """Concatenate WAV clips (same format) with silence between them. Returns (wav bytes, seconds)."""
    out, params, frames = io.BytesIO(), None, []
    for p in parts:
        with wave.open(io.BytesIO(p)) as w:
            pr = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            if params is None:
                params = pr
            elif pr != params:
                raise ValueError(f"audio clips have different formats: {pr} vs {params}")
            frames.append(w.readframes(w.getnframes()))
    if params is None:
        raise ValueError("no audio")
    ch, sw, rate = params
    silence = b"\x00" * int(rate * pause_s) * ch * sw
    with wave.open(out, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(sw)
        w.setframerate(rate)
        for i, f in enumerate(frames):
            if i:
                w.writeframes(silence)
            w.writeframes(f)
    total = sum(len(f) for f in frames) + len(silence) * (len(frames) - 1)
    return out.getvalue(), total / (rate * ch * sw)


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    if pcm[:4] == b"RIFF":  # the server sent a WAV anyway: keep only the samples after the "data" chunk header
        i = pcm.find(b"data")
        pcm = pcm[i + 8 :] if i >= 0 else pcm[44:]
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm[: len(pcm) // 2 * 2])
    return out.getvalue()


def synthesize(turns: list[dict], dest: Path, s: Settings) -> float:
    """Speak every turn (Alex / Sam voices) and write one WAV. Returns the duration in seconds."""
    voices = {"ALEX": s.tts_voice_a, "SAM": s.tts_voice_b}
    clips = [pcm_to_wav(speak(t["text"], voices.get(t["speaker"], s.tts_voice_a), s), s.tts_sample_rate) for t in turns]
    data, secs = join_wavs(clips)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return secs
