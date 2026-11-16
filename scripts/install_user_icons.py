#!/usr/bin/env python3
"""
Installs and integrates user-provided Fotodit icons from Fotodit-icon-pack.zip.

Integrates:
- Fotodit.ico -> assets/app-icon/fotodit.ico
- Fotodit.icns -> assets/app-icon/fotodit.icns
- Fotodit-aperture.png -> raw source & hicolor tile renders (16 to 512 px)
- 1024px render -> assets/app-icon/fotodit-1024.png
"""

import os
import shutil
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(
    r"C:\Users\USER\.gemini\antigravity\brain\0888c2d5-b556-4fc4-bb30-606fd24b135b\scratch\user_icons"
)
APP_ICON_DIR = os.path.join(PROJECT_ROOT, "assets", "app-icon")

def create_hicolor_tile(size: int, aperture_img: Image.Image) -> Image.Image:
    scale = 4 if size <= 256 else 2
    canvas_size = size * scale
    rx = int(112 * (canvas_size / 512.0))

    # Base tile with dark slate/sapphire gradient harmonized with aperture colors
    tile = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    for y in range(canvas_size):
        t = y / float(canvas_size)
        r = int(15 * (1 - t) + 24 * t)
        g = int(23 * (1 - t) + 32 * t)
        b = int(42 * (1 - t) + 64 * t)
        tile_draw.line([(0, y), (canvas_size, y)], fill=(r, g, b, 255))

    mask = Image.new("L", (canvas_size, canvas_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, canvas_size - 1, canvas_size - 1], radius=rx, fill=255)

    base = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    base.paste(tile, (0, 0), mask=mask)

    # Scale aperture to fit with pleasing margin inside rounded rect (78% of tile)
    ap_size = int(canvas_size * 0.78)
    ap_resized = aperture_img.resize((ap_size, ap_size), Image.Resampling.LANCZOS)
    offset = (canvas_size - ap_size) // 2
    base.paste(ap_resized, (offset, offset), mask=ap_resized)

    return base.resize((size, size), Image.Resampling.LANCZOS)

def create_macos_1024(aperture_img: Image.Image) -> Image.Image:
    # Apple 1024px icon with standard 824px squircle content area
    canvas_size = 1024
    content_size = 824
    rx = int(185) # Apple standard radius for 824px tile

    tile = Image.new("RGBA", (content_size, content_size), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    for y in range(content_size):
        t = y / float(content_size)
        r = int(15 * (1 - t) + 24 * t)
        g = int(23 * (1 - t) + 32 * t)
        b = int(42 * (1 - t) + 64 * t)
        tile_draw.line([(0, y), (content_size, y)], fill=(r, g, b, 255))

    mask = Image.new("L", (content_size, content_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, content_size - 1, content_size - 1], radius=rx, fill=255)

    content = Image.new("RGBA", (content_size, content_size), (0, 0, 0, 0))
    content.paste(tile, (0, 0), mask=mask)

    ap_size = int(content_size * 0.82)
    ap_resized = aperture_img.resize((ap_size, ap_size), Image.Resampling.LANCZOS)
    offset = (content_size - ap_size) // 2
    content.paste(ap_resized, (offset, offset), mask=ap_resized)

    # Place centered on 1024x1024
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    pos = (canvas_size - content_size) // 2
    canvas.paste(content, (pos, pos), mask=content)
    return canvas

def main():
    os.makedirs(APP_ICON_DIR, exist_ok=True)

    # 1. Copy user's Fotodit.ico
    user_ico = os.path.join(SRC_DIR, "Fotodit.ico")
    dest_ico = os.path.join(APP_ICON_DIR, "fotodit.ico")
    shutil.copy2(user_ico, dest_ico)
    print(f"Installed Windows ICO: {dest_ico}")

    # 2. Copy user's Fotodit.icns
    user_icns = os.path.join(SRC_DIR, "Fotodit.icns")
    dest_icns = os.path.join(APP_ICON_DIR, "fotodit.icns")
    shutil.copy2(user_icns, dest_icns)
    print(f"Installed macOS ICNS: {dest_icns}")

    # 3. Process Fotodit-aperture.png
    user_png = os.path.join(SRC_DIR, "Fotodit-aperture.png")
    dest_raw_png = os.path.join(APP_ICON_DIR, "fotodit-aperture.png")
    shutil.copy2(user_png, dest_raw_png)
    aperture_img = Image.open(user_png).convert("RGBA")

    # 4. Generate macOS 1024px render
    macos_1024 = create_macos_1024(aperture_img)
    dest_1024 = os.path.join(APP_ICON_DIR, "fotodit-1024.png")
    macos_1024.save(dest_1024, "PNG")
    print(f"Generated macOS 1024px: {dest_1024}")

    # 5. Generate Hicolor PNG suite (16, 24, 32, 48, 64, 128, 256, 512)
    sizes = [16, 24, 32, 48, 64, 128, 256, 512]
    for sz in sizes:
        tile = create_hicolor_tile(sz, aperture_img)
        sz_dir = os.path.join(APP_ICON_DIR, "hicolor", f"{sz}x{sz}", "apps")
        os.makedirs(sz_dir, exist_ok=True)
        dest_png = os.path.join(sz_dir, "com.plyra.fotodit.png")
        tile.save(dest_png, "PNG")
        print(f"Generated hicolor {sz}x{sz}: {dest_png}")

    print("All user icons successfully integrated!")

if __name__ == "__main__":
    main()
