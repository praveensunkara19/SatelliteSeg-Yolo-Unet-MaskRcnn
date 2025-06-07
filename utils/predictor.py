# predictor.py

from utils.tile_fetcher import fetch_stitched_map
from utils.models import load_models
from utils.image_processing import overlay_mask_on_image, split_by_class
from utils.area_calculator import calculate_area

# Load all models at module level
models = load_models()

def predict_image(image, model_type):
    model = models.get(model_type)
    if model is None:
        return {"error": f"{model_type} model not found"}

    try:
        overlay, mask = model.predict(image)
        areas = calculate_area(mask)
        split_images = split_by_class(mask)

        return {
            "original": image,
            "overlay": overlay,
            "mask": mask,
            "split_images": split_images,
            "areas": areas
        }

    except Exception as e:
        print(f"Prediction failed for {model_type}: {e}")
        return {"error": str(e)}

def predict_map(model_type, location_tuple, zoom):
    try:
        lat, lon = location_tuple
        image = fetch_stitched_map(lat, lon, zoom, num_tiles=3)
        return predict_image(image, model_type)
    except Exception as e:
        print(f"Map prediction failed: {e}")
        return {"error": str(e)}
