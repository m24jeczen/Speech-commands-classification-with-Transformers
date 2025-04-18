# # config.py

# # Dataset
# DATA_DIR = 'data'
# TRAIN_SUBDIR = 'train'
# TEST_SUBDIR = 'test'
# AUDIO_SUBFOLDER = 'audio'

# # Training
# NUM_LABELS = 35         # Adjust if you have a different number
# SAMPLE_RATE = 16000
# DURATION = 1.0
# BATCH_SIZE = 8
# EPOCHS = 3
# LEARNING_RATE = 1e-4

# # Model
# MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
# DEVICE = "cuda"  # or "cpu"

# # Save paths
# MODEL_SAVE_PATH = "checkpoints/ast_model.pt"
# PREDICTIONS_SAVE_PATH = "predictions.csv"

SAMPLE_RATE = 16000
LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
AUDIO_DIR = "data/train/audio/"
