
# Satellite Image Segmentation Space 🌍

This project allows you to upload or capture satellite images, predict segmentations using YOLOv11, UNet, and Mask-RCNN models, and visualize the segmented classes and areas separately.

## Features

- Upload custom images or capture live maps
- Predict segmentation masks
- Split segmented output into 6 images (Buildings, Hills, Land, Road, Vegetation, Water)
- Calculate area covered by each class

## Installation steps

git clone https://github.com/praveensunkara19/SatelliteSeg-Yolo-Unet-MaskRcnn.git
cd SatelliteSeg-Yolo-Unet-MaskRcnn

1.Set up virtual environment (optional but recommended)
```bash
python -m venv myenv
myenv\Scripts\activate    # On Windows

2.Install dependencies
```bash
pip install -r requirements.txt

3.How to run
```bash
py app.py
