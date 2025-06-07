# image_processing.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
from io import BytesIO
from config import CLASS_COLORS, CLASSES

def overlay_mask_on_image(image, mask):
    """Overlay segmentation mask with color on the original image."""
    image_np = np.array(image).copy()
    for idx, cls in enumerate(CLASSES):
        color = CLASS_COLORS[cls]
        image_np[mask == idx] = color
    return Image.fromarray(image_np)

def split_by_class(mask):
    """Generate per-class segmentation images with labels and pixel info."""
    labeled_images = []
    if mask is None or len(mask.shape) < 2:
        return labeled_images
        
    width, height = mask.shape[1], mask.shape[0]
    title_height = 50
    total_pixels = width * height

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()

    for idx, class_name in enumerate(CLASSES):
        binary_mask = (mask == idx).astype(np.uint8) * 255
        pixel_count = int(np.sum(mask == idx))
        percent = (pixel_count / total_pixels) * 100 if total_pixels > 0 else 0

        # Create tinted image for class
        tint = np.zeros((height, width, 3), dtype=np.uint8)
        tint[binary_mask == 255] = CLASS_COLORS[class_name]
        class_img = Image.fromarray(tint)

        # Add label banner
        labeled_img = Image.new("RGB", (width, height + title_height), (255, 255, 255))
        labeled_img.paste(class_img, (0, title_height))

        label_text = f"{class_name} - {pixel_count} px ({percent:.2f}%)"
        draw = ImageDraw.Draw(labeled_img)
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_x = (width - (bbox[2] - bbox[0])) // 2
        draw.text((text_x, (title_height - (bbox[3] - bbox[1])) // 2), 
                 label_text, fill=(0, 0, 0), font=font)

        labeled_images.append(labeled_img)

    return labeled_images

