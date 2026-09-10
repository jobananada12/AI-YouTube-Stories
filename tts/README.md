# Local Ukrainian Narrator

This module gives AI-YouTube-Stories its own stable local narrator, independent of FilmDubUA.

## Voice

Default: `uk_UA-oleksa-high` — Ukrainian, male, high-quality Piper voice. The model is Apache-2.0 licensed. An alternative is `uk_UA-mykyta-high`.

The narrator is designed as a distinct synthetic voice with a warm, calm, lower-register storytelling target. It is **not** intended to clone or impersonate the identity of the YouTube reference speaker.

Piper provides local ONNX TTS and supports Ukrainian. Official documentation describes CLI synthesis, voice downloads, sentence silence and a persistent HTTP server. See the Piper documentation before changing model/license choices.

## Setup on Windows

From `C:\AI-YouTube-Stories`:

```powershell
.\tts\setup_tts.ps1
```

The setup installs `piper-tts` into the project's `.venv` and downloads the voice into `tts\models`.

## Test narration

```powershell
.\.venv\Scripts\python.exe tts\generate_narration.py --input narration_ua_30min.txt --output projects\_tts_preview\narration_ua.wav
```

For a short test, create `tts\test.txt` with a few paragraphs and run:

```powershell
.\.venv\Scripts\python.exe tts\generate_narration.py --input tts\test.txt --output projects\_tts_preview\test.wav
```

Alternative male voice:

```powershell
.\.venv\Scripts\python.exe tts\generate_narration.py --voice uk_UA-mykyta-high --input tts\test.txt --output projects\_tts_preview\mykyta_test.wav
```

## Tuning

`tts/narrator_config.json` controls:

- `speed`: 0.96 is intentionally slightly slower than neutral.
- `sentence_silence`: target pause between sentences.
- `paragraph_silence`: target pause between generated chunks.
- `voice`: stable voice identity for future stories.

Do not regenerate the narrator model per story. Keep the same voice and tune only pacing/pauses unless a deliberate voice change is wanted.
