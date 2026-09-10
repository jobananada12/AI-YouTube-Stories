import subprocess
import sys
import wave
from pathlib import Path

from app.config import settings
from core.scene_schema import ScenePlan
from core.tts_schema import AudioSegment, NarrationManifest


class NarrationGenerator:
    """Generate Ukrainian scene narration with the local Piper voice model."""

    def __init__(self, model_dir: str | Path = "tts/models"):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.voice = settings.tts_voice
        self._voice = None

    def _ensure_voice(self):
        if self._voice is not None:
            return self._voice

        self.model_dir.mkdir(parents=True, exist_ok=True)
        model = self.model_dir / f"{self.voice}.onnx"
        config = self.model_dir / f"{self.voice}.onnx.json"

        if not model.exists() or not config.exists():
            print(f"Завантажую голос Piper: {self.voice}...", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "piper.download_voices",
                    "--data-dir",
                    str(self.model_dir),
                    self.voice,
                ],
                check=True,
            )

        if not model.exists() or not config.exists():
            raise FileNotFoundError(f"Piper voice model not found: {model}")

        from piper import PiperVoice
        self._voice = PiperVoice.load(str(model))
        return self._voice

    @staticmethod
    def _wav_duration(path: Path) -> float:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
        return frames / rate if rate else 0.0

    def generate(
        self,
        scene_plan: ScenePlan,
        output_dir: str | Path,
        voice_profile: str = "mykyta",
        rate: int = 136,
        volume: float = 1.0,
    ) -> NarrationManifest:
        voice = self._ensure_voice()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        segments: list[AudioSegment] = []
        total = 0.0

        from piper import SynthesisConfig
        # 136 is intentionally slower than the old 170 rate.
        length_scale = max(0.55, min(1.55, 170.0 / max(80, int(rate))))
        syn_config = SynthesisConfig(
            length_scale=length_scale,
            volume=max(0.0, min(2.0, float(volume))),
        )

        for index, scene in enumerate(scene_plan.scenes, 1):
            text = scene.narration.strip()
            if not text:
                continue
            target = output / f"narration_{scene.number:03d}.wav"
            print(f"Озвучую сцену {scene.number}/{len(scene_plan.scenes)}...", flush=True)
            with wave.open(str(target), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file, syn_config=syn_config)
            duration = self._wav_duration(target)
            segments.append(
                AudioSegment(
                    scene_number=scene.number,
                    text=text,
                    file=target.name,
                    duration_seconds=duration,
                    voice_profile=voice_profile,
                    rate=int(rate),
                    volume=float(volume),
                )
            )
            total += duration
            print(f"  [{index}/{len(scene_plan.scenes)}] {duration:.2f}s", flush=True)

        return NarrationManifest(
            provider="Piper local",
            voice_profile=voice_profile,
            segments=segments,
            total_duration_seconds=total,
        )
