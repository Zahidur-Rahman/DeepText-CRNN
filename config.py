import os

class Config:
    # --- PATHS ---
    # Update this to where your images actually are
    # DATA_DIR = "./mnt/ramdisk/max/90kDICT32px"
    # TRAIN_LABEL_FILE = "my_train.txt"
    # VAL_LABEL_FILE = "my_val.txt"
    DATA_DIR = "" 
    
    TRAIN_LABEL_FILE = "my_train.txt"
    VAL_LABEL_FILE = "my_val.txt"
    
    SAVE_DIR = "checkpoints"
    MODEL_NAME = "best_ocr_model.pth"
    
    # --- MODEL & DATA ---
    IMG_HEIGHT = 32
    IMG_WIDTH = 100
    
   
    ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    
    # --- TRAINING ---
    BATCH_SIZE = 512     
    LEARNING_RATE = 0.0005
    EPOCHS = 10            # 10 is usually enough for this dataset
    
    # --- HARDWARE ---
    DEVICE = "cuda"        # or "cpu"
    NUM_WORKERS = 4       # How many CPU cores to use for loading data