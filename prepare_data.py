import random
from pathlib import Path
from config import Config

def split_dataset():
    cfg = Config()
    
    # 1. Locate the file
    annotation_path = Path(cfg.DATA_ROOT) / cfg.ORIGINAL_ANNOTATION
    print(f"Looking for data at: {annotation_path}")
    
    if not annotation_path.exists():
        print(f"❌ ERROR: File not found!")
        print(f"Make sure your config.py DATA_ROOT is correct.")
        return

    # 2. Read the file
    print(f"Reading {annotation_path}...")
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} images. Shuffling...")

    # 3. Shuffle and Split
    random.seed(42)
    random.shuffle(lines)
    
    total = len(lines)
    train_end = int(total * cfg.TRAIN_RATIO)
    val_end = train_end + int(total * cfg.VAL_RATIO)
    
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]
    
    # 4. Save the new files
    base_dir = Path(cfg.DATA_ROOT)
    
    print("Writing new split files...")
    with open(base_dir / cfg.TRAIN_ANNOTATION, 'w') as f: f.writelines(train_lines)
    with open(base_dir / cfg.VAL_ANNOTATION, 'w') as f: f.writelines(val_lines)
    with open(base_dir / cfg.TEST_ANNOTATION, 'w') as f: f.writelines(test_lines)
    
    print(f"✅ Data split complete!")
    print(f"Train: {len(train_lines)}")
    print(f"Val:   {len(val_lines)}")
    print(f"Test:  {len(test_lines)}")

# ==========================================
# THIS IS THE PART THAT WAS MISSING
# ==========================================
if __name__ == "__main__":
    split_dataset()