from __future__ import annotations

import argparse
from pathlib import Path

from app.main import load_authored_content
from app.config import settings
from core.image_generator import ImageGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate authored scene images without regenerating narration")
    parser.add_argument("--content", default="content/room_without_plan.json")
    parser.add_argument("--project", default="room_without_plan_01")
    args = parser.parse_args()

    story, bible, scene_plan = load_authored_content(args.content)
    output_dir = Path("projects") / args.project / "images"
    print(f"Історія: {story.title}")
    print(f"Сцен: {len(scene_plan.scenes)}")
    print(f"Розмір генерації: {settings.image_width}x{settings.image_height}")
    print("TTS НЕ запускається — наявні WAV залишаються без змін.")

    images = ImageGenerator().generate(scene_plan, bible, output_dir)
    print(f"ГОТОВО: згенеровано {len(images)} зображень у {output_dir}")


if __name__ == "__main__":
    main()
