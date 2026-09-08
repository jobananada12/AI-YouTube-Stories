from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont, ImageFilter


class ThumbnailGenerator:
    """Create a 1280x720 YouTube thumbnail from a generated scene image."""

    WIDTH = 1280
    HEIGHT = 720

    def __init__(self, font_path: str | None = None):
        self.font_path = font_path

    @staticmethod
    def _clean_title(title: str) -> str:
        title = re.sub(r"\s+", " ", title).strip()
        return title[:90]

    def _font(self, size: int):
        candidates = []
        if self.font_path:
            candidates.append(self.font_path)
        candidates += [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
            r"C:\Windows\Fonts\DejaVuSans-Bold.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        return ImageFont.load_default()

    def generate(self, title: str, source_image: str | Path, output_dir: str | Path) -> Path:
        source = Path(source_image)
        if not source.exists():
            raise FileNotFoundError(f"Не знайдено зображення для thumbnail: {source}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "thumbnail.jpg"

        image = Image.open(source).convert("RGB")
        image.thumbnail((self.WIDTH, self.HEIGHT), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.WIDTH, self.HEIGHT))
        x = (self.WIDTH - image.width) // 2
        y = (self.HEIGHT - image.height) // 2
        canvas.paste(image, (x, y))

        # Subtle darkening keeps the title readable without altering the source scene.
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, self.WIDTH, self.HEIGHT), fill=(0, 0, 0, 70))
        draw.rectangle((0, 0, self.WIDTH, 190), fill=(0, 0, 0, 145))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(canvas)
        font = self._font(62)
        text = self._clean_title(title)

        # Word-wrap to a maximum of three lines.
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textbbox((0, 0), candidate, font=font)[2] <= self.WIDTH - 100:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        lines = lines[:3]

        total_h = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines) + 12 * (len(lines) - 1)
        y = max(30, (190 - total_h) // 2)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=3)
            tw = bbox[2] - bbox[0]
            x = (self.WIDTH - tw) // 2
            draw.text((x, y), line, font=font, fill="white", stroke_width=3, stroke_fill="black")
            y += bbox[3] - bbox[1] + 12

        # Small original-studio label; no external branding.
        label_font = self._font(26)
        label = "ОРИГІНАЛЬНА ІСТОРІЯ"
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((40, self.HEIGHT - 55), label, font=label_font, fill="white", stroke_width=2, stroke_fill="black")

        canvas.save(output, "JPEG", quality=94, optimize=True)
        return output
