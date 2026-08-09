from __future__ import annotations

import io
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageFilter

import tempfile

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

def get_writable_dir(dir_name: str) -> Path:
    target = BASE_DIR / dir_name
    try:
        target.mkdir(parents=True, exist_ok=True)
        test_file = target / f".write_test_{uuid.uuid4().hex}"
        test_file.touch()
        test_file.unlink()
        return target
    except (PermissionError, OSError):
        tmp_target = Path(tempfile.gettempdir()) / "mediaforge" / dir_name
        tmp_target.mkdir(parents=True, exist_ok=True)
        return tmp_target

UPLOAD_DIR = get_writable_dir("uploads")
OUTPUT_DIR = get_writable_dir("outputs")

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 300 * 1024 * 1024

app = FastAPI(title="MediaForge Tools", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def safe_suffix(filename: str | None, fallback: str) -> str:
    if not filename:
        return fallback
    suffix = Path(filename).suffix.lower()
    return suffix if suffix and len(suffix) <= 8 else fallback


async def save_upload(upload: UploadFile, kind: Literal["image", "video", "audio"]) -> Path:
    suffix = safe_suffix(upload.filename, ".bin")
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    limit = MAX_IMAGE_BYTES if kind == "image" else MAX_VIDEO_BYTES
    with path.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large. Max {limit // (1024*1024)} MB.")
            f.write(chunk)
    return path


def output_path(ext: str) -> Path:
    ext = ext.lower().lstrip(".")
    return OUTPUT_DIR / f"{uuid.uuid4().hex}.{ext}"


def download_response(path: Path, download_name: str | None = None) -> FileResponse:
    if not path.exists():
        raise HTTPException(500, "Output file was not created")
    return FileResponse(path, filename=download_name or path.name, media_type="application/octet-stream")


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=180)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Processing timed out")
    except FileNotFoundError:
        raise HTTPException(500, "FFmpeg is not installed or available on PATH on this server.")
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", errors="replace")[-1500:]
        raise HTTPException(500, f"FFmpeg error: {msg}")


def read_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")


def smart_remove_background(pil_img: Image.Image) -> Image.Image:
    # Optional AI path when rembg is installed.
    try:
        from rembg import remove  # type: ignore
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        result = remove(buf.getvalue())
        return Image.open(io.BytesIO(result)).convert("RGBA")
    except Exception:
        pass

    # Tested offline fallback: Fast scaled GrabCut with a safe centered-subject rectangle.
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if h < 10 or w < 10:
        raise HTTPException(400, "Image is too small")

    # Downscale for ultra-fast GrabCut computation if image is large
    max_dim = 600
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small_w = max(2, int(w * scale))
        small_h = max(2, int(h * scale))
        proc_bgr = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        proc_bgr = bgr

    ph, pw = proc_bgr.shape[:2]
    margin_x = max(1, int(pw * 0.04))
    margin_y = max(1, int(ph * 0.04))
    rect = (margin_x, margin_y, max(2, pw - 2 * margin_x), max(2, ph - 2 * margin_y))
    mask = np.zeros((ph, pw), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(proc_bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fgmask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
    except cv2.error:
        # Color-distance fallback using corner median as approximate background.
        corners = np.vstack([
            proc_bgr[: max(1,ph//10), : max(1,pw//10)].reshape(-1,3),
            proc_bgr[: max(1,ph//10), -max(1,pw//10):].reshape(-1,3),
            proc_bgr[-max(1,ph//10):, : max(1,pw//10)].reshape(-1,3),
            proc_bgr[-max(1,ph//10):, -max(1,pw//10):].reshape(-1,3),
        ])
        bg = np.median(corners, axis=0)
        dist = np.linalg.norm(proc_bgr.astype(np.float32) - bg.astype(np.float32), axis=2)
        fgmask = np.clip((dist - 15) * 8, 0, 255).astype(np.uint8)

    if scale != 1.0:
        fgmask = cv2.resize(fgmask, (w, h), interpolation=cv2.INTER_LINEAR)

    fgmask = cv2.GaussianBlur(fgmask, (0, 0), 1.2)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = fgmask
    return Image.fromarray(rgba)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "ffmpeg": shutil.which("ffmpeg") is not None, "tools": 10}


@app.post("/api/image/background-remove")
async def image_background_remove(file: UploadFile = File(...)):
    src = await save_upload(file, "image")
    try:
        img = read_image(src)
        out = output_path("png")
        smart_remove_background(img).save(out, "PNG", optimize=True)
        return download_response(out, "background-removed.png")
    finally:
        src.unlink(missing_ok=True)


@app.post("/api/image/upscale")
async def image_upscale(file: UploadFile = File(...), scale: int = Form(2)):
    if scale not in (2, 4):
        raise HTTPException(400, "Scale must be 2 or 4")
    src = await save_upload(file, "image")
    try:
        img = read_image(src).convert("RGB")
        w, h = img.size
        if w * h * scale * scale > 60_000_000:
            raise HTTPException(400, "Output dimensions are too large")
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        up = cv2.resize(arr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(up, (0, 0), 1.0)
        up = cv2.addWeighted(up, 1.18, blur, -0.18, 0)
        out = output_path("png")
        Image.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB)).save(out, "PNG", optimize=True)
        return download_response(out, f"upscaled-{scale}x.png")
    finally:
        src.unlink(missing_ok=True)


@app.post("/api/image/enhance")
async def image_enhance(file: UploadFile = File(...), strength: float = Form(1.0)):
    strength = max(0.25, min(float(strength), 2.0))
    src = await save_upload(file, "image")
    try:
        img = read_image(src).convert("RGB")
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6 + strength * 0.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        arr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        denoised = cv2.bilateralFilter(arr, d=5, sigmaColor=20 * strength, sigmaSpace=20 * strength)
        blur = cv2.GaussianBlur(denoised, (0, 0), 1.1)
        arr = cv2.addWeighted(denoised, 1.0 + 0.18 * strength, blur, -0.18 * strength, 0)
        out = output_path("jpg")
        Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).save(out, "JPEG", quality=94, optimize=True)
        return download_response(out, "enhanced.jpg")
    finally:
        src.unlink(missing_ok=True)


@app.post("/api/image/compress")
async def image_compress(
    file: UploadFile = File(...),
    quality: int = Form(75),
    format: str = Form("webp"),
):
    quality = max(20, min(int(quality), 95))
    fmt = format.lower()
    if fmt not in {"jpg", "jpeg", "webp", "png"}:
        raise HTTPException(400, "Format must be jpg, webp, or png")
    src = await save_upload(file, "image")
    try:
        img = read_image(src)
        ext = "jpg" if fmt in {"jpg", "jpeg"} else fmt
        out = output_path(ext)
        if ext == "jpg":
            img.convert("RGB").save(out, "JPEG", quality=quality, optimize=True, progressive=True)
        elif ext == "webp":
            img.save(out, "WEBP", quality=quality, method=6)
        else:
            img.save(out, "PNG", optimize=True, compress_level=9)
        return download_response(out, f"compressed.{ext}")
    finally:
        src.unlink(missing_ok=True)


@app.post("/api/image/convert")
async def image_convert(file: UploadFile = File(...), format: str = Form("png")):
    fmt = format.lower()
    if fmt not in {"jpg", "jpeg", "png", "webp"}:
        raise HTTPException(400, "Unsupported output format")
    src = await save_upload(file, "image")
    try:
        img = read_image(src)
        ext = "jpg" if fmt in {"jpg", "jpeg"} else fmt
        out = output_path(ext)
        if ext == "jpg":
            img.convert("RGB").save(out, "JPEG", quality=95, optimize=True)
        elif ext == "webp":
            img.save(out, "WEBP", quality=92, method=6)
        else:
            img.save(out, "PNG", optimize=True)
        return download_response(out, f"converted.{ext}")
    finally:
        src.unlink(missing_ok=True)


@app.post("/api/video/upscale")
async def video_upscale(file: UploadFile = File(...), scale: int = Form(2)):
    if scale not in (2, 4):
        raise HTTPException(400, "Scale must be 2 or 4")
    src = await save_upload(file, "video")
    try:
        out = output_path("mp4")
        vf = f"scale=iw*{scale}:ih*{scale}:flags=lanczos,unsharp=5:5:0.5:5:5:0"
        run_ffmpeg(["-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)])
        return download_response(out, f"video-upscaled-{scale}x.mp4")
    finally:
        src.unlink(missing_ok=True)


@app.post("/api/video/enhance")
async def video_enhance(file: UploadFile = File(...), strength: float = Form(1.0)):
    strength = max(0.5, min(float(strength), 2.0))
    src = await save_upload(file, "video")
    try:
        out = output_path("mp4")
        contrast = 1.0 + 0.06 * strength
        saturation = 1.0 + 0.08 * strength
        brightness = 0.01 * strength
        vf = f"hqdn3d=1.5:1.5:6:6,eq=contrast={contrast:.3f}:saturation={saturation:.3f}:brightness={brightness:.3f},unsharp=5:5:{0.35*strength:.3f}:5:5:0"
        run_ffmpeg(["-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)])
        return download_response(out, "video-enhanced.mp4")
    finally:
        src.unlink(missing_ok=True)


@app.post("/api/video/compress")
async def video_compress(file: UploadFile = File(...), level: str = Form("balanced")):
    presets = {
        "high": (31, "veryfast"),
        "balanced": (26, "medium"),
        "quality": (22, "slow"),
    }
    if level not in presets:
        raise HTTPException(400, "Level must be high, balanced, or quality")
    crf, preset = presets[level]
    src = await save_upload(file, "video")
    try:
        out = output_path("mp4")
        run_ffmpeg(["-i", str(src), "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)])
        return download_response(out, "compressed-video.mp4")
    finally:
        src.unlink(missing_ok=True)


@app.post("/api/video/to-mp3")
async def video_to_mp3(file: UploadFile = File(...), bitrate: str = Form("192k")):
    if bitrate not in {"128k", "192k", "256k", "320k"}:
        raise HTTPException(400, "Unsupported bitrate")
    src = await save_upload(file, "video")
    try:
        out = output_path("mp3")
        run_ffmpeg(["-i", str(src), "-vn", "-c:a", "libmp3lame", "-b:a", bitrate, str(out)])
        return download_response(out, "audio.mp3")
    finally:
        src.unlink(missing_ok=True)


@app.post("/api/video/convert")
async def video_convert(file: UploadFile = File(...), format: str = Form("mp4")):
    fmt = format.lower()
    if fmt not in {"mp4", "webm", "mov", "mkv"}:
        raise HTTPException(400, "Unsupported output format")
    src = await save_upload(file, "video")
    try:
        out = output_path(fmt)
        if fmt == "webm":
            args = ["-i", str(src), "-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-c:a", "libopus", str(out)]
        elif fmt == "mov":
            args = ["-i", str(src), "-c:v", "libx264", "-crf", "20", "-c:a", "aac", str(out)]
        elif fmt == "mkv":
            args = ["-i", str(src), "-c:v", "libx264", "-crf", "20", "-c:a", "aac", str(out)]
        else:
            args = ["-i", str(src), "-c:v", "libx264", "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", str(out)]
        run_ffmpeg(args)
        return download_response(out, f"converted.{fmt}")
    finally:
        src.unlink(missing_ok=True)


@app.exception_handler(HTTPException)
def http_exception_handler(_request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
