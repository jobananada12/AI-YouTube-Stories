from __future__ import annotations

import math
import wave
from pathlib import Path


class ProceduralMusicGenerator:
    """Create a simple original ambient WAV without external samples."""

    def generate(self, duration_seconds: float, output_dir: str | Path, filename: str = 'background_ambient.wav') -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        target = output / filename
        rate = 22050
        frames = max(1, int(duration_seconds * rate))
        notes = (110.0, 146.83, 164.81, 220.0)
        with wave.open(str(target), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            block = bytearray()
            for i in range(frames):
                t = i / rate
                note = notes[int(t // 8) % len(notes)]
                value = 0.055 * math.sin(2 * math.pi * note * t) + 0.025 * math.sin(2 * math.pi * note * 2 * t)
                envelope = min(1.0, t / 2.0) * min(1.0, max(0.0, (duration_seconds - t) / 2.0))
                sample = int(max(-1.0, min(1.0, value * envelope)) * 32767)
                block += sample.to_bytes(2, 'little', signed=True)
                if len(block) >= 8192:
                    wav.writeframes(block)
                    block.clear()
            if block:
                wav.writeframes(block)
        return target
