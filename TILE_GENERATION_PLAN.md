# Coherent 1920x1080 scene generation

The previous implementation generated eight independent Stable Diffusion images (4 columns x 2 rows) and stitched them together. That approach was removed because each tile was a separate diffusion generation, so the final result could look like a storyboard/contact sheet or contain unrelated repeated subjects.

## Current design

Each scene is generated exactly once as **one coherent composition** at a VRAM-safe base resolution, then enlarged with ordinary image processing to the required final canvas:

```text
ONE SD generation
        ↓
coherent 16:9 graphite scene
        ↓
grayscale / contrast normalization
        ↓
Lanczos upscale
        ↓
1920x1080 ONE PNG
```

The local server uses VAE tiling only as an internal memory optimization. VAE tiling does not mean eight independent scenes; the diffusion model still receives one prompt and produces one image.

Default low-VRAM generation size:

```text
768x432 → 1920x1080
```

For a stronger GPU, override with `SD_BASE_WIDTH` and `SD_BASE_HEIGHT`, keeping both dimensions divisible by 8. For example:

```text
1024x576 → 1920x1080
```

## Visual requirements

Every scene must be:

- one single continuous image;
- black and white graphite pencil drawing;
- hand-drawn pencil sketch;
- visible graphite strokes and cross-hatching;
- coherent 16:9 composition;
- no collage, grid, split screen, contact sheet, storyboard or comic panels;
- no color or photorealistic photography.

The final file remains exactly `1920x1080`.
