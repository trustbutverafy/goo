"""The painting session as a time-lapse (frames recorded every ~30 engine steps)."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw

d = np.load("out/pear-frames.npz"); rgb, t = d["rgb"], d["t"]
labels = [(0, "charcoal"), (1700, "imprimatura"), (2600, "background"), (5000, "ledge"), (6200, "pear"),
          (7550, "impasto, stem"), (7700, "sgraffito"), (7850, "curing")]
import json
log = json.load(open("out/pear-receipts.json"))["log"]
marks = [(e["t"], e["stage"]) for e in log]
def label(ti):
    s = ""
    for tt, name in marks:
        if ti >= tt - 1: s = name
    return s
# frames are recorded before each stage's mark; show the stage being worked on
order = np.argsort(t, kind="stable")
ims = []
for k in order:
    im = Image.fromarray(rgb[k]).resize((640, 480), Image.LANCZOS)
    nxt = next((name for tt, name in marks if tt >= t[k]), marks[-1][1])
    dr = ImageDraw.Draw(im); dr.rectangle((0, 452, 640, 480), fill=(15, 15, 16))
    dr.text((10, 458), f"step {t[k]:5d}   {nxt}", fill=(230, 225, 215))
    ims.append(im)
ims += [ims[-1]] * 24
for i, im in enumerate(ims): im.save(f"out/tl_{i:04d}.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "18", "-i", "out/tl_%04d.png", "-c:v", "libx264", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-crf", "20", "out/pear-timelapse.mp4"], check=True)
small = [im.resize((400, 300), Image.LANCZOS) for im in ims[::2]]
small[0].save("out/pear-timelapse.gif", save_all=True, append_images=small[1:], duration=110, loop=0, optimize=True)
subprocess.run("rm out/tl_*.png", shell=True)
print(len(ims), "frames")
