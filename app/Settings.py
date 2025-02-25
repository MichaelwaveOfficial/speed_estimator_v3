from pathlib import Path
import os 

''' Main Application Constants. '''

APPLICATION_PATH = Path(__file__).resolve().parent

''' DIRECTORY PATH CONSTANTS. '''

ASSETS_DIR = './assets/'
ICONS_DIR_PATH = os.path.join(APPLICATION_PATH, ASSETS_DIR)
CAPTURE_DIR = './captures/'
CAPTURES_DIR_PATH = os.path.join(APPLICATION_PATH, CAPTURE_DIR)

''' MODELS FOR INFERENCE. '''

# Model Dir Path.
MODELS_PATH = 'detection_models/'
# Available models.
YOLO_V11 = os.path.join(MODELS_PATH, 'yolo11s.pt')
# Set path to run inference on.
DETECTION_MODEL_PATH = os.path.join(APPLICATION_PATH, YOLO_V11)
# One size fits all confidence threshold before adjustment. 
BASE_YOLO_CONFIDENCE_THRESHOLD = 0.85