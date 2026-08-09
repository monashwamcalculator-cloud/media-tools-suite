# MediaForge Test Report

Tested on: 2026-08-09

## Automated result

`python -m pytest -q` → **3 passed**

The 3 test groups exercise all 10 processing endpoints plus health/frontend delivery.

## Tool checks

| Tool | Processing check | Output validation | Result |
|---|---|---|---|
| Background Remover | PNG sample processed | RGBA + original dimensions | PASS |
| Image Upscaler | 2× upscale | 160×120 → 320×240 | PASS |
| Image Enhancer | Enhancement pipeline | Valid JPEG + dimensions | PASS |
| Image Compressor | WebP compression | Valid WebP | PASS |
| Image Converter | PNG → JPG | Valid JPEG | PASS |
| Video Upscaler | 2× FFmpeg upscale | 160×120 → 320×240 via ffprobe | PASS |
| Video Enhancer | FFmpeg filter chain | Valid video stream via ffprobe | PASS |
| Video Compressor | H.264 compression | Valid video stream via ffprobe | PASS |
| Video to MP3 | Audio extraction | MP3 codec via ffprobe | PASS |
| Video Converter | MP4 → WebM | WebM container via ffprobe | PASS |

## Frontend checks

- Root page returns HTTP 200.
- Static JavaScript returns HTTP 200.
- Frontend registry contains 5 image tools and 5 video tools.
- `node --check app/static/app.js` passes.
- Health endpoint confirms FFmpeg is available and reports 10 tools.

## Important implementation note

Background removal automatically uses `rembg` when the optional AI dependency is installed. The base package was tested with its offline GrabCut fallback so the project remains runnable without downloading an AI model.
