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
5. Image prompts/images — in progress
6. Narration audio — implemented
7. Music/SFX — in progress
8. Video rendering — **implemented**
9. Thumbnail — planned
10. YouTube metadata — planned
11. Final project package — planned

## Video rendering

Step 8 uses the local FFmpeg executable to:

1. validate the required scene images and narration WAV files;
2. normalize each scene image to the configured 16:9 output size;
3. create a continuous visual timeline from the scene durations;
4. concatenate all narration segments;
5. optionally mix the generated background music;
6. encode the final MP4 as H.264 video + AAC audio with `+faststart`.

The final file is written to `projects/<project_id>/final/story.mp4`, with a render manifest at `final/render.json`.

Run the full pipeline with:

```powershell
python -m app.main "Твоя оригінальна тема" --minutes 30
```

If you only want to generate the project assets without rendering the MP4 yet:

```powershell
python -m app.main "Твоя оригінальна тема" --minutes 30 --no-render
```

FFmpeg must be installed and available through `FFMPEG_BIN` (default: `ffmpeg`).

## Security

Never commit tokens, passwords, cookies, or private credentials. Use `.env` locally and keep it ignored by Git.
