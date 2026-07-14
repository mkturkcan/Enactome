#!/usr/bin/env python3
"""Composite subtitles onto every scene's frames, then stitch scenes + music into the video.

Reads frames/<scene_id>/*.png (produced by record_ui.py and make_frames.py), burns a
subtitle bar with the scene caption onto each frame, encodes one clip per scene, concatenates
in storyboard order, and muxes the elevator-music bed. Output: enactome_demo.mp4.
"""
import json, os, subprocess, textwrap, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SB = json.load(open(HERE / "storyboard.json"))
W, H, FPS = SB["meta"]["width"], SB["meta"]["height"], SB["meta"]["fps"]
FR = HERE / "frames"; SUB = HERE / "sub"; CLIPS = HERE / "clips"
for d in (SUB, CLIPS): d.mkdir(exist_ok=True)

def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def burn(src, dst, caption):
    img = Image.open(src).convert("RGB")
    if img.size != (W, H): img = img.resize((W, H))
    if caption:
        d = ImageDraw.Draw(img, "RGBA")
        f = font(26)
        lines = textwrap.wrap(caption, width=78)
        bar_h = 26 + len(lines)*34 + 20
        d.rectangle([0, H-bar_h, W, H], fill=(0,0,0,180))
        y = H - bar_h + 16
        for ln in lines:
            w = d.textlength(ln, font=f)
            d.text(((W-w)/2, y), ln, font=f, fill=(240,244,250)); y += 34
    img.save(dst)

def scene_dir(sc):
    return FR / (f"ui_{sc['id']}" if sc["kind"]=="ui" else sc["id"])

def main():
    clip_list = []
    for sc in SB["scenes"]:
        sd = scene_dir(sc)
        frames = sorted(sd.glob("*.png"))
        if not frames:
            print(f"WARN: no frames for scene {sc['id']} ({sd}); skipping"); continue
        od = SUB / sc["id"]; od.mkdir(exist_ok=True)
        for i, fp in enumerate(frames):
            burn(fp, od / f"{i:04d}.png", sc.get("caption",""))
        clip = CLIPS / f"{sc['id']}.mp4"
        subprocess.run(["ffmpeg","-y","-framerate",str(FPS),"-i",str(od/"%04d.png"),
                        "-c:v","libx264","-pix_fmt","yuv420p","-r",str(FPS),str(clip)],
                       check=True, capture_output=True)
        clip_list.append(clip)
    # concat
    concat = HERE / "concat.txt"
    concat.write_text("".join(f"file '{c}'\n" for c in clip_list))
    silent = HERE / "silent.mp4"
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(concat),
                    "-c","copy",str(silent)], check=True, capture_output=True)
    # mux music (trim to video length, fade out)
    out = HERE / "enactome_demo.mp4"
    music = HERE / "assets" / "music.wav"
    if music.exists():
        subprocess.run(["ffmpeg","-y","-i",str(silent),"-i",str(music),
                        "-c:v","copy","-c:a","aac","-shortest",
                        "-af","afade=t=out:st=%d:d=2" % (sum(s['dur'] for s in SB['scenes'])-2),
                        str(out)], check=True, capture_output=True)
    else:
        shutil.copy(silent, out)
    print("wrote", out)

if __name__ == "__main__":
    main()
