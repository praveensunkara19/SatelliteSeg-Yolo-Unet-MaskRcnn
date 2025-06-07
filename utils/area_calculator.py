#area_calculator.py
import numpy as np
from config import CLASSES

def calculate_area(mask):
    return {cls: int(np.sum(mask == idx)) for idx, cls in enumerate(CLASSES)}
