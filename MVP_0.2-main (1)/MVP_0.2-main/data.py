import numpy as np
import os
import pandas as pd
from pathlib import Path

def load_custom_dataset(image_dir, label_csv, detector):
    # ... (copy function from original file)

def create_memory_efficient_training_data(detector, num_samples=None):
    # You may want to wrap this around the detector's method for easier access if needed.
    return detector.create_memory_efficient_training_data(num_samples=num_samples)
