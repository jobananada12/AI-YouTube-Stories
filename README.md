# AI-YouTube-Stories

Local-first studio for producing original long-form Ukrainian YouTube stories.

## Complete pipeline

1. Original story concept via local Ollama
2. Long-form narration script
3. Character Bible for visual consistency
4. Scene/director plan
5. Local AI scene images through an Automatic1111-compatible API
6. Dedicated local Ukrainian narrator through Piper TTS
7. Original procedural background music and optional SFX
8. FFmpeg MP4 assembly
9. 1280x720 YouTube thumbnail
10. Story-specific YouTube title, description, keywords, tags and hashtags
11. Final validation and ZIP project package

The pipeline is designed to avoid external copyrighted media. Generated media is created locally where configured; this does not constitute a legal guarantee against every possible copyright or platform claim.

## Local narrator

The project has its own narrator module in `tts/`. It is independent of FilmDubUA and keeps one stable Ukrainian male synthetic voice for future stories.

Default voice: `uk_UA-oleksa-high`.

Setup:

```powershell
.\\tts\\setup_tts.ps1
```

Test:

```powershell
.\\.venv\\Scripts\\python.exe tts\\generate_narration.py --input narration_ua_30min.txt --output projects\\_tts_preview\\narration_ua.wav
```

The narrator is a distinct synthetic voice and is not intended to clone or impersonate a real reference speaker.

## Configuration

Copy `.env.example` to `.env` and configure the local services.

### Ollama

`OLLAMA_BASE_URL` defaults to `http://127.0.0.1:11434`.

### FilmDubUA

`FILMDUBUA_PATH` points to the local FilmDubUA repository. Real credentials/tokens stay only in `.env`.

### Images

The integrated image generator uses an Automatic1111-compatible `/sdapi/v1/txt2img` endpoint.

```text
IMAGE_API_URL=http://127.0.0.1:7860
IMAGE_MODEL=
IMAGE_STEPS=20
IMAGE_CFG_SCALE=7.0
```

If no local image server is running, the program stops at image generation instead of silently inventing placeholder artwork.

## Run

Full production run:

```powershell
python -m app.main "Таємнича історія про покинутий будинок у Карпатах" --minutes 30 --package
```

Useful development flags:

```powershell
python -m app.main "Тема" --minutes 30 --no-images --no-render
python -m app.main "Тема" --minutes 30 --no-music --no-sfx
```

`--no-images` is intended for development/testing only; the final validation gate still requires actual scene PNG files.

## Output

Each run creates:

```text
projects/<project_id>/
  story.json
  script.md
  character_bible.json
  scenes.json
  narration.json
  audio/
  images/
  music/
  sfx/
  thumbnail/thumbnail.jpg
  youtube/youtube.json
  youtube/youtube.md
  final/story.mp4
  final/render.json
  project_manifest.json
```

With `--package`, the completed project is archived under `projects/packages/`.

## Security

Never commit tokens, passwords, cookies or private credentials. Use `.env`, which is ignored by Git.
