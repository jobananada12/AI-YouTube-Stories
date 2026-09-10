from ukrainian_tts.tts import TTS, Voices, Stress
from pathlib import Path

OUT = Path("tts/uktts_tests")
OUT.mkdir(parents=True, exist_ok=True)

text = """
Існують будинки, які старіють разом із містами.
Їхні стіни пам'ятають голоси людей, яких давно немає.

Але іноді серед звичайних кімнат з'являється одна,
якої ніколи не було на жодному плані.

Ця історія почалася тихого осіннього вечора,
коли чоловік повернувся до старого будинку,
щоб розібрати речі свого батька.

Він ще не знав, що цієї ночі знайде двері,
яких раніше не існувало.
"""

voices = [
    ("oleksa", Voices.Oleksa.value),
    ("dmytro", Voices.Dmytro.value),
    ("mykyta", Voices.Mykyta.value),
]

print("=== UKRAINIAN TTS — 3 MALE VOICES ===")
print()

tts = TTS(device="cpu")

for name, voice in voices:
    output = OUT / f"{name}.wav"

    print(f"Generating: {name}")

    with open(output, "wb") as f:
        _, accented = tts.tts(
            text,
            voice,
            Stress.Dictionary.value,
            f
        )

    print(f"DONE: {output}")
    print()

print("ALL TESTS COMPLETE")
