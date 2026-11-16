#!/usr/bin/env python3
"""
Generates high-fidelity visual assets for Fotodit (com.plyra.fotodit).

Outputs:
- SVG Masters: assets/app-icon/fotodit.svg, fotodit-small.svg, scalable SVG
- PNGs: assets/app-icon/hicolor/<size>x<size>/apps/com.plyra.fotodit.png (16 to 512)
- macOS 1024px render: assets/app-icon/fotodit-1024.png
- Windows Multi-resolution ICO: assets/app-icon/fotodit.ico (16, 24, 32, 48, 64, 128, 256 px)
"""

import math
import os
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_ICON_DIR = os.path.join(PROJECT_ROOT, "assets", "app-icon")

SVG_CONTENT = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs>
    <clipPath id="tile-clip">
      <rect width="512" height="512" rx="112" />
    </clipPath>
    <linearGradient id="bg-grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#181e4b" />
      <stop offset="100%" stop-color="#2563eb" />
    </linearGradient>
    <linearGradient id="blade-a" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="100%" stop-color="#0284c7" />
    </linearGradient>
    <linearGradient id="blade-b" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#818cf8" />
      <stop offset="100%" stop-color="#4f46e5" />
    </linearGradient>
    <radialGradient id="lens-center" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#ffffff" />
      <stop offset="60%" stop-color="#93c5fd" />
      <stop offset="100%" stop-color="#3b82f6" />
    </radialGradient>
  </defs>
  <g clip-path="url(#tile-clip)">
    <rect width="512" height="512" fill="url(#bg-grad)" />
    <!-- Outer Lens Ring -->
    <circle cx="256" cy="256" r="176" fill="none" stroke="#93c5fd" stroke-width="4" opacity="0.6" />
    <circle cx="256" cy="256" r="164" fill="#0f172a" fill-opacity="0.85" stroke="#3b82f6" stroke-width="6" />
    <!-- Aperture Blades -->
    <path d="M 256 100 L 290 200 L 370 125 Z" fill="url(#blade-a)" />
    <path d="M 370 125 L 330 235 L 420 220 Z" fill="url(#blade-b)" />
    <path d="M 420 220 L 315 275 L 365 375 Z" fill="url(#blade-a)" />
    <path d="M 365 375 L 245 290 L 220 410 Z" fill="url(#blade-b)" />
    <path d="M 220 410 L 195 305 L 105 340 Z" fill="url(#blade-a)" />
    <path d="M 105 340 L 195 235 L 125 150 Z" fill="url(#blade-b)" />
    <!-- Lens Center Aperture -->
    <circle cx="256" cy="256" r="48" fill="url(#lens-center)" />
    <circle cx="256" cy="256" r="24" fill="#60a5fa" />
    <!-- Focus Marks -->
    <line x1="256" y1="52" x2="256" y2="76" stroke="#f8fafc" stroke-width="4" stroke-linecap="round" />
    <line x1="256" y1="436" x2="256" y2="460" stroke="#f8fafc" stroke-width="4" stroke-linecap="round" />
    <line x1="52" y1="256" x2="76" y2="256" stroke="#f8fafc" stroke-width="4" stroke-linecap="round" />
    <line x1="436" y1="256" x2="460" y2="256" stroke="#f8fafc" stroke-width="4" stroke-linecap="round" />
  </g>
</svg>
'''

def render_fotodit_icon(size: int) -> Image.Image:
    # Render with 4x supersampling for high-quality downsampling
    scale = 4 if size <= 256 else (2 if size <= 512 else 1)
    canvas_size = size * scale
    
    rx = int(112 * (canvas_size / 512.0))
    
    tile = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    
    # Vertical gradient
    for y in range(canvas_size):
        t = y / float(canvas_size)
        r = int(24 * (1 - t) + 37 * t)
        g = int(30 * (1 - t) + 99 * t)
        b = int(75 * (1 - t) + 235 * t)
        tile_draw.line([(0, y), (canvas_size, y)], fill=(r, g, b, 255))
        
    mask = Image.new("L", (canvas_size, canvas_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([0, 0, canvas_size - 1, canvas_size - 1], radius=rx, fill=255)
    
    base_tile = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    base_tile.paste(tile, (0, 0), mask=mask)
    
    lens_draw = ImageDraw.Draw(base_tile)
    cx, cy = canvas_size / 2.0, canvas_size / 2.0
    r_outer = canvas_size * 0.32
    r_inner = canvas_size * 0.16
    
    # Outer ring
    lens_draw.ellipse(
        [cx - r_outer * 1.08, cy - r_outer * 1.08, cx + r_outer * 1.08, cy + r_outer * 1.08],
        outline=(147, 197, 253, 160),
        width=max(1, int(3 * scale))
    )
    lens_draw.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        fill=(15, 23, 42, 220),
        outline=(59, 130, 246, 255),
        width=max(1, int(5 * scale))
    )
    
    # 6 Aperture blades
    num_blades = 6
    for i in range(num_blades):
        angle = i * (2 * math.pi / num_blades)
        x1 = cx + r_outer * 0.95 * math.cos(angle)
        y1 = cy + r_outer * 0.95 * math.sin(angle)
        
        angle_next = (i + 1) * (2 * math.pi / num_blades)
        x2 = cx + r_inner * math.cos(angle + math.pi / 6)
        y2 = cy + r_inner * math.sin(angle + math.pi / 6)
        
        x3 = cx + r_outer * 0.95 * math.cos(angle_next)
        y3 = cy + r_outer * 0.95 * math.sin(angle_next)
        
        blade_color = (56, 189, 248, 230) if i % 2 == 0 else (129, 140, 248, 230)
        lens_draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=blade_color)

    # Center aperture opening
    lens_draw.ellipse(
        [cx - r_inner * 0.8, cy - r_inner * 0.8, cx + r_inner * 0.8, cy + r_inner * 0.8],
        fill=(248, 250, 252, 255)
    )
    lens_draw.ellipse(
        [cx - r_inner * 0.45, cy - r_inner * 0.45, cx + r_inner * 0.45, cy + r_inner * 0.45],
        fill=(96, 165, 250, 255)
    )
    
    # Focus crosshairs
    mark_len = canvas_size * 0.06
    mark_dist = r_outer * 1.18
    for angle in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
        mx1 = cx + mark_dist * math.cos(angle)
        my1 = cy + mark_dist * math.sin(angle)
        mx2 = cx + (mark_dist + mark_len) * math.cos(angle)
        my2 = cy + (mark_dist + mark_len) * math.sin(angle)
        lens_draw.line([(mx1, my1), (mx2, my2)], fill=(248, 250, 252, 200), width=max(1, int(3 * scale)))

    if scale > 1:
        return base_tile.resize((size, size), Image.Resampling.LANCZOS)
    return base_tile

def main():
    os.makedirs(APP_ICON_DIR, exist_ok=True)
    
    # 1. Write SVG files
    svg_master_path = os.path.join(APP_ICON_DIR, "fotodit.svg")
    svg_small_path = os.path.join(APP_ICON_DIR, "fotodit-small.svg")
    scalable_dir = os.path.join(APP_ICON_DIR, "hicolor", "scalable", "apps")
    os.makedirs(scalable_dir, exist_ok=True)
    svg_scalable_path = os.path.join(scalable_dir, "com.plyra.fotodit.svg")
    
    for path in [svg_master_path, svg_small_path, svg_scalable_path]:
        with open(path, "w", encoding="utf-8") as f:
            f.write(SVG_CONTENT)
        print(f"Generated: {os.path.relpath(path, PROJECT_ROOT)}")
        
    # 2. Generate PNGs for each hicolor size
    sizes = [16, 24, 32, 48, 64, 128, 256, 512]
    rendered_images = {}
    for sz in sizes:
        img = render_fotodit_icon(sz)
        rendered_images[sz] = img
        sz_dir = os.path.join(APP_ICON_DIR, "hicolor", f"{sz}x{sz}", "apps")
        os.makedirs(sz_dir, exist_ok=True)
        png_path = os.path.join(sz_dir, "com.plyra.fotodit.png")
        img.save(png_path, "PNG")
        print(f"Generated: {os.path.relpath(png_path, PROJECT_ROOT)} ({sz}x{sz})")

    # 3. Generate 1024px render
    img1024 = render_fotodit_icon(1024)
    png1024_path = os.path.join(APP_ICON_DIR, "fotodit-1024.png")
    img1024.save(png1024_path, "PNG")
    print(f"Generated: {os.path.relpath(png1024_path, PROJECT_ROOT)} (1024x1024)")

    # 4. Generate Windows multi-resolution ICO (16, 24, 32, 48, 64, 128, 256)
    ico_path = os.path.join(APP_ICON_DIR, "fotodit.ico")
    ico_img = rendered_images[256]
    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    ico_img.save(ico_path, format="ICO", sizes=ico_sizes)
    print(f"Generated: {os.path.relpath(ico_path, PROJECT_ROOT)} (multi-res ICO)")

if __name__ == "__main__":
    main()
