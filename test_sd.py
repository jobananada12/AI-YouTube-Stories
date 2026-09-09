import torch
from diffusers import StableDiffusionPipeline

MODEL_PATH = r"C:\Users\AI\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
OUTPUT = "test_sd.png"

print("Loading Stable Diffusion 1.5...")
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    local_files_only=True,
    safety_checker=None,
)

# Для 2 GB VRAM — максимально економний режим
pipe.enable_sequential_cpu_offload()

pipe.enable_attention_slicing()

prompt = (
    "cinematic original fantasy scene, "
    "an abandoned wooden house in the Carpathian mountains at night, "
    "moonlight, mist, mysterious atmosphere, detailed environment, "
    "dramatic lighting, realistic cinematic composition"
)

negative_prompt = (
    "text, watermark, logo, celebrity, famous character, "
    "copyrighted character, deformed, blurry, low quality"
)

print("Generating...")

with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=512,
        num_inference_steps=15,
        guidance_scale=7.0,
    ).images[0]

image.save(OUTPUT)

print()
print("SUCCESS!")
print("Image:", OUTPUT)