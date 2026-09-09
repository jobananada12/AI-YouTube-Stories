from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    filmdubua_path: str = os.getenv("FILMDUBUA_PATH", r"C:\FilmDubUA")
    filmdubua_voice: str = os.getenv("FILMDUBUA_VOICE", "")
    filmdubua_voice_profile: str = os.getenv("FILMDUBUA_VOICE_PROFILE", "neutral")
    tts_rate: int = int(os.getenv("TTS_RATE", "170"))
    tts_volume: float = float(os.getenv("TTS_VOLUME", "1.0"))

    # Local AI image generation. GT 1030 is VRAM-constrained, so the defaults
    # deliberately generate a small 16:9 frame and let FFmpeg upscale it later.
    image_api_url: str = os.getenv("IMAGE_API_URL", "http://127.0.0.1:7860")
    image_model: str = os.getenv("IMAGE_MODEL", "")
    image_width: int = int(os.getenv("IMAGE_WIDTH", "640"))
    image_height: int = int(os.getenv("IMAGE_HEIGHT", "360"))
    image_steps: int = int(os.getenv("IMAGE_STEPS", "16"))
    image_cfg_scale: float = float(os.getenv("IMAGE_CFG_SCALE", "7.0"))
    image_timeout: int = int(os.getenv("IMAGE_TIMEOUT", "900"))

    ffmpeg_bin: str = os.getenv("FFMPEG_BIN", "ffmpeg")
    output_fps: int = int(os.getenv("OUTPUT_FPS", "30"))
    video_width: int = int(os.getenv("VIDEO_WIDTH", "1920"))
    video_height: int = int(os.getenv("VIDEO_HEIGHT", "1080"))


settings = Settings()
