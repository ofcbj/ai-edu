"""
Configuration and constants for Tokyo rent prediction models
"""
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data file paths
DATA_FILE = 'tokyo_rent_data_v2.csv'
FONT_FILE = 'meiryo.ttc'

# Model file paths
MODEL_BASIC_BEST = 'best_rent_model_basic.pth'
MODEL_ATTENTION_BEST = 'best_rent_model_attention.pth'
MODEL_BASIC_FINAL = 'rent_prediction_model_basic.pth'
MODEL_ATTENTION_FINAL = 'rent_prediction_model_attention.pth'

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0

# Train/validation/test split
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Model architecture parameters
EMBEDDING_DIM = 10
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
DROPOUT_RATE = 0.3

# Tokyo 23 wards information
WARD_INFO = {
    '千代田区': {'tier': 'premium', 'avg_rent': 180000},
    '中央区': {'tier': 'premium', 'avg_rent': 170000},
    '港区': {'tier': 'premium', 'avg_rent': 200000},
    '新宿区': {'tier': 'premium', 'avg_rent': 150000},
    '文京区': {'tier': 'high', 'avg_rent': 140000},
    '台東区': {'tier': 'mid', 'avg_rent': 120000},
    '墨田区': {'tier': 'mid', 'avg_rent': 110000},
    '江東区': {'tier': 'mid', 'avg_rent': 130000},
    '品川区': {'tier': 'high', 'avg_rent': 145000},
    '目黒区': {'tier': 'high', 'avg_rent': 155000},
    '大田区': {'tier': 'mid', 'avg_rent': 115000},
    '世田谷区': {'tier': 'high', 'avg_rent': 135000},
    '渋谷区': {'tier': 'premium', 'avg_rent': 190000},
    '中野区': {'tier': 'mid', 'avg_rent': 105000},
    '杉並区': {'tier': 'mid', 'avg_rent': 110000},
    '豊島区': {'tier': 'mid', 'avg_rent': 115000},
    '北区': {'tier': 'mid', 'avg_rent': 100000},
    '荒川区': {'tier': 'mid', 'avg_rent': 95000},
    '板橋区': {'tier': 'mid', 'avg_rent': 95000},
    '練馬区': {'tier': 'mid', 'avg_rent': 100000},
    '足立区': {'tier': 'low', 'avg_rent': 85000},
    '葛飾区': {'tier': 'low', 'avg_rent': 85000},
    '江戸川区': {'tier': 'low', 'avg_rent': 90000}
}

# Building structure types
BUILDING_STRUCTURES = ['木造', 'RC造', '鉄骨造', 'SRC造']

# Property types
PROPERTY_TYPES = ['マンション', 'アパート', 'ハイツ', 'コーポ']

# Feature columns
FEATURE_COLUMNS = [
    'ward_encoded',
    'room_size',
    'station_distance',
    'building_age',
    'building_structure_encoded',
    'property_type_encoded'
]

TARGET_COLUMN = 'rent'
