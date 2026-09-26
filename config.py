"""
config.py — File Cấu Hình Trung Tâm
=======================================
Tất cả các siêu tham số (hyperparameters) và đường dẫn được lưu ở đây.
Đề tài: Nghiên cứu và xây dựng chương trình nhận diện biển số xe.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
PLATES_DIR = os.path.join(DATASET_DIR, 'plates')
LABELS_FILE = os.path.join(DATASET_DIR, 'labels.csv')

SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
DETECTOR_PATH = os.path.join(SAVED_MODELS_DIR, 'plate_detector.pth')
RECOGNIZER_PATH = os.path.join(SAVED_MODELS_DIR, 'plate_recognizer.pth')

RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLATES_DIR, exist_ok=True)

BLANK_CHAR = '_'
ALPHABET = '_-0123456789ABCDEFGHKLMNPSTUVXYZ'
NUM_CLASSES = len(ALPHABET)
CHAR2IDX = {ch: i for i, ch in enumerate(ALPHABET)}
IDX2CHAR = {i: ch for i, ch in enumerate(ALPHABET)}

PLATE_WIDTH = 192
PLATE_HEIGHT = 64
IMG_CHANNELS = 1
MAX_LABEL_LEN = 10

BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_EPOCHS = 100
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-06
PATIENCE = 15
MIN_DELTA = 0.001

CNN_CHANNELS = [64, 128, 256, 512]
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3

MIN_PLATE_AREA_RATIO = 0.005
MAX_PLATE_AREA_RATIO = 0.5
MIN_ASPECT = 0.8
MAX_ASPECT = 6.0
CANNY_LOW = 50
CANNY_HIGH = 150

RANDOM_SEED = 42

ESP_DEFAULT_IP   = '192.168.137.49'
ESP_DEFAULT_PORT = 80
ESP_TIMEOUT      = 2.0
ESP_ALERT_PATH   = '/alert'
ESP_STATUS_PATH  = '/status'
