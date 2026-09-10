from google.cloud import texttospeech
from pathlib import Path

OUT = Path("tts/tests")
OUT.mkdir(parents=True, exist_ok=True)

text = """Існують будинки, які старіють разом із містами. Їхні стіни пам'ятають голоси людей, яких давно немає. Але іноді серед звичайних кімнат з'являється одна, якої ніколи не було на жодному плані.

Ця історія почалася тихого осіннього вечора, коли чоловік повернувся до старого будинку, щоб розібрати речі свого батька. Він ще не знав, що цієї ночі знайде двері, яких раніше не існувало."""

voices = [
    "uk-UA-Chirp3-HD-Achird",
    "uk-UA-Chirp3-HD-Charon",
    "uk-UA-Chirp3-HD-Rasalgethi",
    "uk-UA-Chirp3-HD-Enceladus",
]

client = texttospeech.TextToSpeechClient()

for voice_name in voices:
    print(f"Generating {voice_name}...")

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="uk-UA",
        name=voice_name,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.92,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    filename = OUT / f"{voice_name}.mp3"
    filename.write_bytes(response.audio_content)

    print(f"  OK: {filename}")

print("\nALL 4 VOICES GENERATED.")
