import numpy as np
from PIL import Image, ImageDraw, ImageFont
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

class MaskRCNNPredictor:
    def __init__(self, model_path):
        self.model_path = model_path

        # Define class names and class colors (no longer from config.py)
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

        # Setup Detectron2 config
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_names)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(cfg)

    def predict(self, image):
        original_size = image.size
        image_np = np.array(image.convert("RGB"))

        outputs = self.predictor(image_np[:, :, ::-1])  # BGR for Detectron2
        instances = outputs["instances"].to("cpu")

        if not instances.has("pred_masks") or not instances.has("pred_classes"):
            raise ValueError("No masks or classes predicted by Mask R-CNN.")

        masks = instances.pred_masks.numpy()  # (N, H, W)
        classes = instances.pred_classes.numpy().astype(np.uint8)  # (N,)
        boxes = instances.pred_boxes.tensor.numpy()  # (N, 4)

        h, w = image_np.shape[:2]
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(len(masks)):
            mask = masks[i] > 0.5
            cls_id = classes[i]
            semantic_mask[mask] = cls_id
            color_mask[mask] = self.class_colors.get(cls_id, (255, 255, 255))

        # Overlay mask
        overlay = Image.blend(image.convert("RGB"), Image.fromarray(color_mask), alpha=0.5)
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()

        for i in range(len(masks)):
            cls_id = classes[i]
            label = self.class_names.get(cls_id, str(cls_id))
            color = self.class_colors.get(cls_id, (255, 255, 255))
            x1, y1, x2, y2 = boxes[i].astype(int)

            # Draw bounding box and label
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 10), label, fill=color, font=font)

        return overlay, semantic_mask
