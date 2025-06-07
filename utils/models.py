#models.py

import os
from utils.yolov11 import YOLOPredictor
from utils.unet import UnetPredictor
from utils.maskrcnn import MaskRCNNPredictor

def load_models():
    models = {}
    try:
        models["YOLOv11"] = YOLOPredictor(os.path.join("models", "yolov11_best.pt"))
        print("YOLOv11 model loaded successfully.")
    except Exception as e:
        print("YOLOv11 load error:", e)

    try:
        models["UNet"] = UnetPredictor(os.path.join("models", "unet_best_model.h5"))
        print("UNet model loaded successfully.")
    except Exception as e:
        print("UNet load error:", e)

    try:
        models["MaskRCNN"] = MaskRCNNPredictor(os.path.join("models", "maskrcnn_model_final.pth"))
        print("MaskRCNN model loaded successfully.")
    except Exception as e:
        print("MaskRCNN load error:", e)

    return models
