#yolov11

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

class YOLOPredictor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = {
            0: "Buildings",
            1: "Hills",
            2: "Land",
            3: "Road",
            4: "Vegetation",
            5: "Water"
        }
        self.class_colors = {
            0: (235, 16, 16),
            1: (136, 96, 11),
            2: (247, 120, 7),
            3: (0, 0, 0),
            4: (28, 106, 11),
            5: (19, 158, 244)
        }

    def predict(self, image):
        original_size = image.size
        image_resized = image.resize((512, 512))
        img = np.array(image_resized)
        results = self.model(img)

        if results[0].masks is None or results[0].boxes is None:
            raise ValueError("No segmentation masks or boxes found in YOLO output.")

        masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
        classes = results[0].boxes.cls.cpu().numpy().astype(np.uint8)  # (N,)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)

        # Resize masks to original size
        semantic_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for i in range(len(masks)):
            semantic_mask[masks[i] > 0.5] = classes[i]
        semantic_mask = Image.fromarray(semantic_mask).resize(original_size, Image.NEAREST)
        semantic_mask_np = np.array(semantic_mask)

        # Create color overlay
        color_mask = np.zeros((semantic_mask_np.shape[0], semantic_mask_np.shape[1], 3), dtype=np.uint8)
        for class_idx, color in self.class_colors.items():
            color_mask[semantic_mask_np == class_idx] = color

        # Prepare overlay and drawing
        overlay = Image.blend(image.convert("RGB"), Image.fromarray(color_mask), alpha=0.5)
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()

        # Scale boxes back to original image size and draw labels
        w_ratio = original_size[0] / 512
        h_ratio = original_size[1] / 512
        for i in range(len(boxes)):
            cls_id = classes[i]
            label = self.class_names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1 * w_ratio), int(y1 * h_ratio), int(x2 * w_ratio), int(y2 * h_ratio)

            draw.rectangle([x1, y1, x2, y2], outline=self.class_colors[cls_id], width=2)
            draw.text((x1, y1 - 10), label, fill=self.class_colors[cls_id], font=font)

        return overlay, semantic_mask_np
