"""Renders public/og.png (1200x630): the tool name in Japanese and English on a plain background.

usage: python3 make_og.py [font file] [output]
Needs Pillow and a font with Japanese glyphs (e.g. /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc, or
Noto Sans JP). Colours follow the site's light theme.
"""
import sys

from PIL import Image, ImageDraw, ImageFont

font_path = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
out = sys.argv[2] if len(sys.argv) > 2 else "public/og.png"

W, H = 1200, 630
BG, FG, MUTED, ACCENT = (246, 246, 244), (31, 31, 30), (94, 94, 90), (47, 111, 222)
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# icon: a note with a bar across it (same idea as favicon.svg)
x0, y0, s = 96, 170, 150
d.rounded_rectangle([x0, y0, x0 + s, y0 + s], radius=34, fill=FG)
line = int(s * 0.07)
d.line([(x0 + s * 0.41, y0 + s * 0.66), (x0 + s * 0.41, y0 + s * 0.31), (x0 + s * 0.66, y0 + s * 0.27), (x0 + s * 0.66, y0 + s * 0.61)], fill=BG, width=line, joint="curve")
for cx, cy in ((x0 + s * 0.31, y0 + s * 0.66), (x0 + s * 0.56, y0 + s * 0.61)):
    r = s * 0.08
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=BG, width=line)
d.line([(x0 + s * 0.22, y0 + s * 0.22), (x0 + s * 0.78, y0 + s * 0.78)], fill=ACCENT, width=int(s * 0.09))

big = ImageFont.truetype(font_path, 112)
mid = ImageFont.truetype(font_path, 64)
small = ImageFont.truetype(font_path, 34)
tx = x0 + s + 56
d.text((tx, 150), "BGM排除くん", font=big, fill=FG)
d.text((tx, 290), "BGM Buster", font=mid, fill=MUTED)
d.text((96, 430), "動画のBGMだけを、ブラウザで消す。", font=small, fill=FG)
d.text((96, 480), "Remove background music from video in your browser.", font=small, fill=MUTED)
d.rectangle([0, H - 14, W, H], fill=ACCENT)
img.save(out, "PNG", optimize=True)
print(out, img.size)
