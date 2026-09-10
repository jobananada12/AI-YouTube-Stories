import wave as wavfile
from pathlib import Path
from tts_uk.inference import synthesis

OUT = Path("tts/tests")
OUT.mkdir(parents=True, exist_ok=True)

text = """Існують будинки, які старіють разом із містами. Їхні стіни пам'ятають голоси людей, яких давно немає. Але іноді серед звичайних кімнат з'являється одна, якої ніколи не було на жодному плані.

Ця історія почалася тихого осіннього вечора, коли чоловік повернувся до старого будинку, щоб розібрати речі свого батька. Він ще не знав, що цієї ночі знайде двері, яких раніше не існувало."""

print("=== TTS-UK / MYKYTA TEST ===")
print("Voice: mykyta")
print("Sample rate: 44100 Hz")
print("CPU mode")
print()

mels, wave, stats = synthesis(
    text=text,
    voice="mykyta",
    n_takes=1,
    use_latest_take=False,
    token_dur_scaling=1,
    f0_mean=0,
    f0_std=0,
    energy_mean=0,
    energy_std=0,
    sigma_decoder=0.8,
    sigma_token_duration=0.666,
    sigma_f0=1,
    sigma_energy=1,
)

print()
print("STATS:")
print(stats)

# Перетворюємо tensor у PCM16 без TorchCodec
audio = wave.detach().cpu().squeeze().numpy()

# Нормалізація
audio = audio.clip(-1.0, 1.0)
pcm16 = (audio * 32767).astype("<i2")

output = OUT / "mykyta_test.wav"

with wavfile.open(str(output), "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(44100)
    f.writeframes(pcm16.tobytes())

print()
print(f"DONE: {output}")
print(f"SIZE: {output.stat().st_size / 1024 / 1024:.2f} MB")
