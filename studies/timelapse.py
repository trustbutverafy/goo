"""A painting session as a time-lapse. DIR holds frames.npz and receipts.json (from a painting script)."""
import json, os, subprocess, glob
import numpy as np
from PIL import Image, ImageDraw

DIR = os.environ.get("DIR", "out/stilllife"); NAME = os.environ.get("NAME", "timelapse")
d = np.load(f"{DIR}/frames.npz"); rgb, t = d["rgb"], d["t"]
marks = [(e["t"], e["stage"]) for e in json.load(open(f"{DIR}/receipts.json"))["log"]]
H, W = rgb.shape[1:3]; scale = max(1, 960 // W)
ims = []
for k in np.argsort(t, kind="stable"):
    im = Image.fromarray(rgb[k]).resize((W * scale, H * scale), Image.LANCZOS)
    stage = next((name for tt, name in marks if tt >= t[k]), marks[-1][1])
    dr = ImageDraw.Draw(im); dr.rectangle((0, im.height - 26, im.width, im.height), fill=(15, 15, 16))
    dr.text((10, im.height - 20), f"tick {t[k]:6d}   {stage}", fill=(230, 225, 215))
    ims.append(im)
ims += [ims[-1]] * 30
tmp = f"{DIR}/_tl"; os.makedirs(tmp, exist_ok=True)
for i, im in enumerate(ims): im.save(f"{tmp}/{i:04d}.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", f"{tmp}/%04d.png", "-c:v", "libx264", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-crf", "20", f"{DIR}/{NAME}.mp4"], check=True)
for f in glob.glob(f"{tmp}/*.png"): os.remove(f)
os.rmdir(tmp)
print(len(ims), "frames ->", f"{DIR}/{NAME}.mp4")
