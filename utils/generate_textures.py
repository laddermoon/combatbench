#!/usr/bin/env python3
"""
Generate texture images for CombatBench arena.

Usage:
    python generate_textures.py

Requirements:
    pip install Pillow
"""

import os
from PIL import Image, ImageDraw, ImageFont

# Ensure textures directory exists
TEXTURES_DIR = os.path.join(os.path.dirname(__file__), 'textures')
os.makedirs(TEXTURES_DIR, exist_ok=True)


def generate_wall_texture(width=1024, height=1024):
    """
    Generate wall texture with off-white background and "CombatBench" text.

    Args:
        width: Texture width in pixels
        height: Texture height in pixels
    """
    # Create image with off-white (米白色) background
    # Off-white: RGB(245, 245, 238) - warm off-white
    img = Image.new('RGB', (width, height), color=(245, 245, 238))
    draw = ImageDraw.Draw(img)

    # Add subtle grid pattern for visual interest
    grid_size = 64
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=(242, 242, 235), width=1)
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=(242, 242, 235), width=1)

    # Try to use a nice font, fallback to default if not available
    try:
        # Try to use a bold font for the text
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120)
    except:
        try:
            font_large = ImageFont.truetype("arial.ttf", 120)
        except:
            # Use default font
            font_large = ImageFont.load_default()

    # Get text bounding box for main title
    text_main = "CombatBench"
    text_sub = "powered by D-Robotics"
    
    bbox_main = draw.textbbox((0, 0), text_main, font=font_large)
    main_width = bbox_main[2] - bbox_main[0]
    main_height = bbox_main[3] - bbox_main[1]
    
    # Try to load a smaller font for subtitle
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60)
    except:
        try:
            font_small = ImageFont.truetype("arial.ttf", 60)
        except:
            font_small = font_large
            
    bbox_sub = draw.textbbox((0, 0), text_sub, font=font_small)
    sub_width = bbox_sub[2] - bbox_sub[0]
    sub_height = bbox_sub[3] - bbox_sub[1]

    # Calculate positions
    # Combine heights with some padding between them
    text_spacing = 30
    total_text_height = main_height + text_spacing + sub_height
    max_text_width = max(main_width, sub_width)
    
    border_margin = 60
    pattern_height = total_text_height + 2 * border_margin
    pattern_center_y = height * 4 // 5
    
    # Starting Y for the entire pattern
    pattern_y = pattern_center_y - pattern_height // 2
    
    # Y positions for texts
    y_main = pattern_y + border_margin
    y_sub = y_main + main_height + text_spacing

    # X positions for texts (centered)
    x_main = (width - main_width) // 2
    x_sub = (width - sub_width) // 2

    # Box dimensions for decorations
    box_x1 = (width - max_text_width) // 2 - border_margin
    box_y1 = pattern_y
    box_x2 = (width + max_text_width) // 2 + border_margin
    box_y2 = pattern_y + pattern_height

    # Draw main text shadow
    draw.text((x_main + 3, y_main + 3), text_main, fill=(220, 220, 215), font=font_large)
    # Draw main text with a rich but not too dark blue/slate color
    draw.text((x_main, y_main), text_main, fill=(70, 100, 140), font=font_large)
    
    # Draw sub text shadow
    draw.text((x_sub + 2, y_sub + 2), text_sub, fill=(220, 220, 215), font=font_small)
    # Draw sub text with a slightly lighter, warm grey/brown color
    draw.text((x_sub, y_sub), text_sub, fill=(140, 130, 120), font=font_small)

    # Add a decorative border
    draw.rectangle(
        [box_x1, box_y1, box_x2, box_y2],
        outline=(180, 190, 200),  # Slight blue tint to match main text
        width=5
    )
    draw.rectangle(
        [box_x1 + 5, box_y1 + 5, box_x2 - 5, box_y2 - 5],
        outline=(200, 205, 210),
        width=2
    )

    # Add corner decorations
    corner_size = 20
    corner_offset = 40
    corners = [
        (box_x1 - corner_offset, box_y1 - corner_offset),
        (box_x2 + corner_offset, box_y1 - corner_offset),
        (box_x1 - corner_offset, box_y2 + corner_offset),
        (box_x2 + corner_offset, box_y2 + corner_offset),
    ]
    for cx, cy in corners:
        draw.rectangle(
            [cx - corner_size, cy - corner_size, cx + corner_size, cy + corner_size],
            fill=(180, 190, 200)
        )

    # Save the image
    output_path = os.path.join(TEXTURES_DIR, 'wall.png')
    img.save(output_path)
    print(f"Generated wall texture: {output_path}")
    return output_path


def generate_floor_texture(width=1024, height=1024):
    """
    Generate floor texture with gray background and "D-Robotics" text.

    Args:
        width: Texture width in pixels
        height: Texture height in pixels
    """
    # Create image with gray background
    # Gray: RGB(180, 180, 180) - medium gray
    img = Image.new('RGB', (width, height), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)

    # Add subtle grid pattern for floor tiles
    tile_size = 128
    for x in range(0, width, tile_size):
        draw.line([(x, 0), (x, height)], fill=(170, 170, 170), width=2)
    for y in range(0, height, tile_size):
        draw.line([(0, y), (width, y)], fill=(170, 170, 170), width=2)

    # Try to use a nice font, fallback to default if not available
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
    except:
        try:
            font_large = ImageFont.truetype("arial.ttf", 100)
        except:
            font_large = ImageFont.load_default()

    # Get text bounding box
    text = ""
    bbox = draw.textbbox((0, 0), text, font=font_large)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Draw text in center
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Draw text shadow
    draw.text((x + 3, y + 3), text, fill=(160, 160, 160), font=font_large)
    # Draw text with dark gray color
    draw.text((x, y), text, fill=(60, 60, 60), font=font_large)

    # Add boxing ring circle in the center
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3
    draw.ellipse(
        [center_x - radius, center_y - radius,
         center_x + radius, center_y + radius],
        outline=(140, 140, 140),
        width=3
    )
    draw.ellipse(
        [center_x - radius + 5, center_y - radius + 5,
         center_x + radius - 5, center_y + radius - 5],
        outline=(150, 150, 150),
        width=1
    )

    # Add center line (like boxing ring center line)
    draw.line(
        [(center_x - radius, center_y), (center_x + radius, center_y)],
        fill=(140, 140, 140),
        width=2
    )

    # Add corner markers (boxing ring style - red/blue corners)
    corner_size = 40
    corner_positions = [
        ((50, 50), (220, 100, 100)),  # Top-left - Red corner
        ((width - 50, 50), (100, 100, 220)),  # Top-right - Blue corner
        ((50, height - 50), (100, 100, 220)),  # Bottom-left - Blue corner
        ((width - 50, height - 50), (220, 100, 100)),  # Bottom-right - Red corner
    ]
    for (cx, cy), color in corner_positions:
        # Outer ring
        draw.ellipse(
            [cx - corner_size, cy - corner_size, cx + corner_size, cy + corner_size],
            outline=color,
            width=4
        )
        # Inner dot
        draw.ellipse(
            [cx - 8, cy - 8, cx + 8, cy + 8],
            fill=color
        )

    # Save the image
    output_path = os.path.join(TEXTURES_DIR, 'floor.png')
    img.save(output_path)
    print(f"Generated floor texture: {output_path}")
    return output_path


def generate_ceiling_texture(width=1024, height=1024):
    """
    Generate ceiling texture with subtle pattern.

    Args:
        width: Texture width in pixels
        height: Texture height in pixels
    """
    # Create image with light gray background
    img = Image.new('RGB', (width, height), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)

    # Add panel pattern
    panel_size = 256
    for x in range(0, width, panel_size):
        draw.line([(x, 0), (x, height)], fill=(220, 220, 220), width=3)
    for y in range(0, height, panel_size):
        draw.line([(0, y), (width, y)], fill=(220, 220, 220), width=3)

    # Add light fixture representations at corners
    light_positions = [
        (width // 4, height // 4),
        (3 * width // 4, height // 4),
        (width // 4, 3 * height // 4),
        (3 * width // 4, 3 * height // 4),
    ]
    for lx, ly in light_positions:
        # Light fixture glow
        for r in range(40, 0, -5):
            alpha = int(255 * (1 - r / 40))
            draw.ellipse(
                [lx - r, ly - r, lx + r, ly + r],
                fill=(255, 255, 240)
            )
        # Light center
        draw.ellipse(
            [lx - 15, ly - 15, lx + 15, ly + 15],
            fill=(255, 255, 220)
        )

    # Save the image
    output_path = os.path.join(TEXTURES_DIR, 'ceiling.png')
    img.save(output_path)
    print(f"Generated ceiling texture: {output_path}")
    return output_path


def main():
    """Generate all required textures."""
    print("Generating CombatBench arena textures...")
    print(f"Output directory: {TEXTURES_DIR}")
    print()

    # Generate textures
    generate_wall_texture()
    generate_floor_texture()
    generate_ceiling_texture()

    print()
    print("Texture generation complete!")
    print()
    print("Generated files:")
    print("  - textures/wall.png    (Off-white background with 'CombatBench')")
    print("  - textures/floor.png   (Gray background with 'D-Robotics')")
    print("  - textures/ceiling.png (Light gray with light fixtures)")


if __name__ == "__main__":
    main()
