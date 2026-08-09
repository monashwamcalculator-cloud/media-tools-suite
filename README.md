# MediaForge — 10 Image & Video Tools

A deployable FastAPI + FFmpeg media tools website with a responsive frontend.

## Included tools

### Image
1. Background Remover — optional `rembg` AI path + tested offline GrabCut fallback
2. Image Upscaler — 2× / 4×
3. Image Enhancer — denoise, local contrast, sharpening
4. Image Compressor — WebP / JPG / optimized PNG
5. Image Converter — JPG / PNG / WebP

### Video
6. Video Upscaler — 2× / 4×
7. Video Enhancer — denoise, contrast, saturation, sharpen
8. Video Compressor — 3 presets
9. Video to MP3 — 128/192/256/320 kbps
10. Video Converter — MP4 / WebM / MOV / MKV

## Requirements

- Python 3.11+
- FFmpeg installed and available on PATH

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open `http://localhost:8000`.

## Optional AI background removal

For higher-quality background removal:

```bash
pip install -r requirements-ai.txt
```

`rembg` downloads its model on first use. If unavailable, the app automatically uses the built-in GrabCut fallback.

## Production notes

- Put the app behind Nginx/Cloudflare.
- Add authentication/rate limiting if needed.
- Configure a cron/job to delete old files from `app/uploads` and `app/outputs`.
- Set upload/body limits at the reverse proxy too.
- For heavy video workloads, move FFmpeg jobs to a queue/worker setup.
- Do not claim ML/AI video enhancement unless you integrate a real model/API; the included enhancer is FFmpeg-based.

## Existing watermark tools

Your already-built image/video watermark remover can be linked from the same navigation or merged as two additional endpoints. This package intentionally focuses on the 10 complementary tools requested after the watermark remover.
