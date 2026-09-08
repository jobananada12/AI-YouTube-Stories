from __future__ import annotations

import math
import wave
from pathlib import Path


class ProceduralSFXGenerator:
    """Generate optional original one-shot SFX without external samples."""

    def generate_hit(self, output_dir: str | Path, filename: str = 'transition_hit.wav') -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        target = output / filename
        rate, duration = 22050, 0.45
        with wave.open(str(target), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
            for i in range(int(rate * duration)):
                t = i / rate
                freq = 180 - 130 * (t / duration)
                env = max(0.0, 1.0 - t / duration) ** 2
                sample = int(math.sin(2 * math.pi * freq * t) * env * 0.18 * 32767)
                wav.writeframes(sample.to_bytes(2, 'little', signed=True))
        return target
