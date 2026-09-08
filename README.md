# AI-YouTube-Stories

Local-first studio for creating original long-form narrated YouTube stories.

## Goals

- Generate original stories, characters, scenes and narration scripts.
- Produce Ukrainian narration through a local FilmDubUA-compatible TTS provider.
- Generate original AI visuals for scenes and a YouTube thumbnail.
- Use copyright-conscious/original music and sound effects.
- Assemble the story automatically into an MP4 with FFmpeg.
- Generate story-specific YouTube title, description, keywords, tags and hashtags.
- Keep secrets and API tokens outside Git (`.env`).

## Pipeline status

1. Story concept — implemented
2. Long-form script — implemented
3. Character bible — implemented
4. Scene breakdown — implemented
5. Image prompts/images — **in progress**
6. Narration audio — implemented
7. Music/SFX — **in progress**
8. Video rendering — implemented
9. Thumbnail — implemented when a scene image exists
10. YouTube metadata — **in progress**
11. Final project package — implemented
12. Final validation / packaging gate — **implemented**

## Step 12 — Final validation gate

Before a project can be marked ready, the packager checks the core story files plus actual generated narration WAV segments and scene PNG files. A final MP4 is required when normal rendering is enabled.

The manifest is written to `projects/<project_id>/project_manifest.json` and records which optional assets are present, including thumbnail, YouTube metadata, music and SFX.

`--package` now uses the same validation mode as the current run, so `--no-render --package` can create an asset-only ZIP when the required non-video assets are present, while a normal run requires `final/story.mp4`.

## Video rendering

The local FFmpeg renderer:

1. validates the required scene images and narration WAV files;
2. normalizes each scene image to the configured 16:9 output size;
3. creates a continuous visual timeline from scene durations;
4. concatenates narration segments;
5. optionally mixes generated background music;
6. encodes H.264 video + AAC audio with `+faststart`.

The final file is written to `projects/<project_id>/final/story.mp4`, with a render manifest at `final/render.json`.

## Run

Full run:

```powershell
python -m app.main "Твоя оригінальна тема" --minutes 30
```

Asset-only run:

```powershell
python -m app.main "Твоя оригінальна тема" --minutes 30 --no-render
```

Create a ZIP after the current validation passes:

```powershell
python -m app.main "Твоя оригінальна тема" --minutes 30 --package
```

FFmpeg must be installed and available through `FFMPEG_BIN` (default: `ffmpeg`).

## Security

Never commit tokens, passwords, cookies, or private credentials. Use `.env` locally and keep it ignored by Git.
