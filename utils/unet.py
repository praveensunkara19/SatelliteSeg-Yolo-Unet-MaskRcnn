#unet.py

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from scipy.ndimage import label, center_of_mass

class UnetPredictor:
    def __init__(self, model_path, num_classes=6):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.num_classes = num_classes
        self.input_size = (256, 256)  # This must match training

        self.class_names = {
            0: "Buildings",
            1: "Hills",
            2: "Land",
            3: "Road",
            4: "Vegetation",
            5: "Water"
        }

        self.class_colors = {
            0: (235, 16, 16),   # Buildings
            1: (136, 96, 11),   # Hills
            2: (247, 120, 7),   # Land
            3: (0, 0, 0),       # Road
            4: (28, 106, 11),   # Vegetation
            5: (19, 158, 244)   # Water
        }

    def preprocess(self, image):
        image = image.resize(self.input_size)
        image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)

    def predict(self, image):
        original_size = image.size
        input_array = self.preprocess(image)

        pred = self.model.predict(input_array)[0]
        pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)

        # Resize to original size
        semantic_mask_img = Image.fromarray(pred_mask).resize(original_size, Image.NEAREST)
        semantic_mask = np.array(semantic_mask_img)

        # Create overlay
        color_mask = np.zeros((semantic_mask.shape[0], semantic_mask.shape[1], 3), dtype=np.uint8)
        for class_idx, color in self.class_colors.items():
            color_mask[semantic_mask == class_idx] = color

        overlay = Image.blend(image.convert("RGB"), Image.fromarray(color_mask), alpha=0.5)
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()

        # Draw labels at centers of class regions
        for class_idx, class_name in self.class_names.items():
            binary_mask = (semantic_mask == class_idx).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue

            labeled_mask, num_features = label(binary_mask)
            centers = center_of_mass(binary_mask, labeled_mask, range(1, num_features + 1))

            for center in centers:
                y, x = map(int, center)
                draw.text((x, y), class_name, fill=self.class_colors[class_idx], font=font)

        return overlay, semantic_mask
