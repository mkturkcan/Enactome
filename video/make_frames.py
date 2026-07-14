#!/usr/bin/env python3
"""Generate card frames and paper-scroll frames for scenes that are not live UI.

Cards: title / section / outro (rendered with PIL). Paper: renders the manuscript PDF
pages to a tall strip and pans down it. Run after record_ui.py; produces frames/<id>/.
Works anywhere (no browser needed) so the non-UI scenes render even in a headless box.
"""
import json, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SB = json.load(open(HERE / "storyboard.json"))
W, H, FPS = SB["meta"]["width"], SB["meta"]["height"], SB["meta"]["fps"]
BG = (15, 20, 32); FG = (205, 214, 230); ACC = (77, 163, 255); ORA = (224, 130, 20)

def font(sz, bold=False):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def card_frame(heading, sub):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    hf, sf = font(88, True), font(34)
    # brand split color for ENACTOME
    if heading.upper() == "ENACTOME":
        a, b = "ENACT", "OME"
        wa = d.textlength(a, font=hf); wb = d.textlength(b, font=hf)
        x0 = (W - wa - wb) / 2; y = H/2 - 90
        d.text((x0, y), a, font=hf, fill=FG); d.text((x0+wa, y), b, font=hf, fill=ACC)
    else:
        w = d.textlength(heading, font=hf)
        d.text(((W-w)/2, H/2-90), heading, font=hf, fill=FG)
    w = d.textlength(sub, font=sf)
    d.text(((W-w)/2, H/2+30), sub, font=sf, fill=(150,160,180))
    return img

def emit_cards():
    for sc in SB["scenes"]:
        if sc["kind"] != "card":
            continue
        d = HERE / "frames" / sc["id"]; d.mkdir(parents=True, exist_ok=True)
        img = card_frame(sc["heading"], sc["sub"])
        n = int(sc["dur"] * FPS)
        for i in range(n):
            img.save(d / f"{i:04d}.png")   # static card (subtitles composited later)

def emit_paper():
    import pypdfium2 as pdfium
    sc = next(s for s in SB["scenes"] if s["kind"] == "paper")
    pdf_path = HERE / "assets" / "manuscript.pdf"
    doc = pdfium.PdfDocument(str(pdf_path))
    # render each page to W-wide image, stack vertically
    pages = []
    for i in range(len(doc)):
        p = doc[i]; pw, ph = p.get_size()
        scale = W / pw
        im = p.render(scale=scale).to_pil().convert("RGB")
        pages.append(im)
    strip_h = sum(p.height for p in pages)
    strip = Image.new("RGB", (W, strip_h), (255,255,255))
    y = 0
    for p in pages:
        strip.paste(p, (0, y)); y += p.height
    d = HERE / "frames" / sc["id"]; d.mkdir(parents=True, exist_ok=True)
    n = int(sc["dur"] * FPS)
    max_off = max(1, strip_h - H)
    for i in range(n):
        off = int(max_off * (i / (n - 1)))
        frame = strip.crop((0, off, W, off + H))
        frame.save(d / f"{i:04d}.png")

if __name__ == "__main__":
    emit_cards()
    emit_paper()
    print("card + paper frames written under frames/")
