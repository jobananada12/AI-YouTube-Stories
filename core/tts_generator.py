import sys
import wave
from pathlib import Path

from app.config import settings
from core.scene_schema import ScenePlan
from core.tts_schema import AudioSegment, NarrationManifest


class NarrationGenerator:
    """Generate Ukrainian narration through the local FilmDubUA Piper engine."""

    def __init__(self, filmdubua_path: str | Path | None = None):
        self.filmdubua_path = Path(filmdubua_path or settings.filmdubua_path).expanduser().resolve()
        self._tts = None

    def _load_filmdubua_tts(self):
        if not self.filmdubua_path.exists():
            raise FileNotFoundError(f"FilmDubUA не знайдено: {self.filmdubua_path}. Вкажи FILMDUBUA_PATH у .env.")
        core_path = self.filmdubua_path / "core"
        if not (core_path / "tts.py").exists():
            raise FileNotFoundError(f"У {self.filmdubua_path} немає core/tts.py. Потрібна актуальна версія FilmDubUA.")
        if str(self.filmdubua_path) not in sys.path:
            sys.path.insert(0, str(self.filmdubua_path))
        try:
            from core import tts
        except Exception as exc:
            raise RuntimeError(f"Не вдалося підключити FilmDubUA TTS: {exc}") from exc
        self._tts = tts
        return tts

    @staticmethod
    def _wav_duration(path: Path) -> float:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
        return frames / rate if rate else 0.0

    def generate(self, scene_plan: ScenePlan, output_dir: str | Path, voice_profile: str = "neutral", rate: int = 170, volume: float = 1.0) -> NarrationManifest:
        tts = self._load_filmdubua_tts()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        segments: list[AudioSegment] = []
        total = 0.0
        for scene in scene_plan.scenes:
            text = scene.narration.strip()
            if not text:
                continue
            target = output / f"narration_{scene.number:03d}.wav"
            print(f"Озвучую сцену {scene.number}/{len(scene_plan.scenes)}...")
            tts.synthesize_ukrainian(text=text, output_wav=str(target), rate=rate, volume=volume, profile=voice_profile)
            duration = self._wav_duration(target)
            segments.append(AudioSegment(scene_number=scene.number, text=text, file=target.name, duration_seconds=duration, voice_profile=voice_profile, rate=rate, volume=volume))
            total += duration
        return NarrationManifest(provider="FilmDubUA/Piper", voice_profile=voice_profile, segments=segments, total_duration_seconds=total)
