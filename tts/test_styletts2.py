import soundfile as sf
from styletts2_inference.models import StyleTTS2
from ukrainian_word_stress import Stressifier
from ipa_uk import ipa
from unicodedata import normalize
import torch
import re

OUT = "tts\styletts2_test.wav"

text = """
Існують будинки, які старіють разом із містами. Їхні стіни пам'ятають голоси людей, яких давно немає.

Але іноді серед звичайних кімнат з'являється одна, якої ніколи не було на жодному плані.

Ця історія почалася тихого осіннього вечора, коли чоловік повернувся до старого будинку, щоб розібрати речі свого батька.

Він ще не знав, що цієї ночі знайде двері, яких раніше не існувало.
"""

print("=== STYLE TTS 2 UKRAINIAN TEST ===")
print("Model: patriotyk/styletts2_ukrainian_single")
print("Device: CPU")
print()

device = "cpu"

print("Loading model...")
model = StyleTTS2(
    hf_path="patriotyk/styletts2_ukrainian_single",
    device=device
)

print("Model loaded.")

# Нормалізація Unicode
text = normalize("NFC", text.strip())

# Українські наголоси
print("Applying Ukrainian stress...")
stressify = Stressifier()
text = stressify(text)

# Українська IPA
print("Converting to IPA...")
phonemes = ipa(text)

# Токенізація
tokens = model.tokenizer.encode(phonemes)

print("Synthesizing...")

with torch.no_grad():
    wav = model(tokens)

wav = wav.detach().cpu().numpy()

sf.write(
    OUT,
    wav,
    24000,
    subtype="PCM_16"
)

print()
print("DONE!")
print(f"Output: {OUT}")
print(f"Duration: {len(wav) / 24000:.2f} sec")
