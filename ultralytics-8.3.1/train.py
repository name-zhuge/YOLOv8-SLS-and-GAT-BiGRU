import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import YOLO

from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Load model configuration
model = YOLO("/root/autodl-tmp/ultralytics-8.3.1/ultralytics-8.3.1/ultralytics/cfg/models/v8/yolov8-SLS.yaml")

# Train the model
results = model.train(data=r"/root/autodl-tmp/ultralytics/ultralytics-main/fish.v5i.yolov8/data.yaml",
                      imgsz=640,
                      epochs=500,
                      batch=8,
                      device=0,
                      optimizer="SGD",
                      workers=0,
                      patience=0)
# Validate model performance
metrics = model.val()

# Validate the model on the test set
test_results = model.val(data=r'/root/autodl-tmp/ultralytics/ultralytics-main/fish.v5i.yolov8/data.yaml',
                         split='test',
                         imgsz=640,
                         batch=8)