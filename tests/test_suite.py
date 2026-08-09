from pathlib import Path
import io
import json
import subprocess
import tempfile
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app

ROOT = Path(__file__).resolve().parent
FIX = ROOT / "fixtures"
FIX.mkdir(exist_ok=True)
client = TestClient(app)


def make_image():
    p = FIX / "sample.png"
    img = Image.new("RGB", (160, 120), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((45, 20, 115, 95), fill=(40, 110, 210))
    d.rectangle((65, 70, 95, 112), fill=(235, 90, 80))
    img.save(p)
    return p


def make_video():
    p = FIX / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=12:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)
    ], check=True)
    return p


def post_file(url, p, data=None):
    with p.open("rb") as f:
        return client.post(url, files={"file": (p.name, f)}, data=data or {})


def probe_bytes(content, suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        path = f.name
    try:
        proc = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=format_name,duration:stream=codec_type,codec_name,width,height",
            "-of", "json", path
        ], check=True, stdout=subprocess.PIPE, text=True)
        return json.loads(proc.stdout)
    finally:
        Path(path).unlink(missing_ok=True)


def test_health_and_frontend():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {'status':'ok','ffmpeg':True,'tools':10}
    page = client.get('/')
    assert page.status_code == 200
    assert b'MediaForge' in page.content
    js = client.get('/static/app.js')
    assert js.status_code == 200
    assert js.text.count("type:'image'") == 5
    assert js.text.count("type:'video'") == 5


def test_all_image_tools_deep():
    p = make_image()

    r = post_file('/api/image/background-remove', p)
    assert r.status_code == 200, r.text
    out = Image.open(io.BytesIO(r.content))
    assert out.mode == 'RGBA'
    assert out.size == (160,120)

    r = post_file('/api/image/upscale', p, {'scale':'2'})
    assert r.status_code == 200, r.text
    out = Image.open(io.BytesIO(r.content))
    assert out.size == (320,240)

    r = post_file('/api/image/enhance', p, {'strength':'1'})
    assert r.status_code == 200, r.text
    out = Image.open(io.BytesIO(r.content))
    assert out.format == 'JPEG' and out.size == (160,120)

    r = post_file('/api/image/compress', p, {'quality':'70','format':'webp'})
    assert r.status_code == 200, r.text
    out = Image.open(io.BytesIO(r.content))
    assert out.format == 'WEBP'

    r = post_file('/api/image/convert', p, {'format':'jpg'})
    assert r.status_code == 200, r.text
    out = Image.open(io.BytesIO(r.content))
    assert out.format == 'JPEG'


def test_all_video_tools_deep():
    p = make_video()

    r = post_file('/api/video/upscale', p, {'scale':'2'})
    assert r.status_code == 200, r.text
    info = probe_bytes(r.content, '.mp4')
    vs = next(s for s in info['streams'] if s['codec_type']=='video')
    assert (vs['width'],vs['height']) == (320,240)

    r = post_file('/api/video/enhance', p, {'strength':'1'})
    assert r.status_code == 200, r.text
    info = probe_bytes(r.content, '.mp4')
    assert any(s['codec_type']=='video' for s in info['streams'])

    r = post_file('/api/video/compress', p, {'level':'balanced'})
    assert r.status_code == 200, r.text
    info = probe_bytes(r.content, '.mp4')
    assert any(s['codec_type']=='video' for s in info['streams'])

    r = post_file('/api/video/to-mp3', p, {'bitrate':'192k'})
    assert r.status_code == 200, r.text
    info = probe_bytes(r.content, '.mp3')
    assert any(s['codec_type']=='audio' and s['codec_name']=='mp3' for s in info['streams'])

    r = post_file('/api/video/convert', p, {'format':'webm'})
    assert r.status_code == 200, r.text
    info = probe_bytes(r.content, '.webm')
    assert 'webm' in info['format']['format_name']
