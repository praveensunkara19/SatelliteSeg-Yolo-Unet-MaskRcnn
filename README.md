
# Satellite Image Segmentation Space 🌍

This project allows you to upload or capture satellite images, predict segmentations using YOLOv11, UNet, and Mask-RCNN models, and visualize the segmented classes and areas separately.

## Features

- Upload custom images or capture live maps
- Predict segmentation masks
- Split segmented output into 6 images (Buildings, Hills, Land, Road, Vegetation, Water)
- Calculate area covered by each class

##Installation steps

git clone https://github.com/praveensunkara19/ObjectTracker.git
cd ObjectTracker

2. Set up virtual environment (optional but recommended)
```bash
python -m venv myenv
myenv\Scripts\activate    # On Windows

3. Install dependencies
```bash
pip install -r requirements.txt

4. How to run
py app.py
