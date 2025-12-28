import os
import random

# Path where your images are stored
# We use the path you confirmed works: ./mnt/ramdisk/max/90kDICT32px
DATA_ROOT = './mnt/ramdisk/max/90kDICT32px'
OUTPUT_TRAIN = 'my_train.txt'
OUTPUT_VAL = 'my_val.txt'

def create_annotation_files():
    valid_files = []
    
    print(f"🕵️‍♂️ Scanning {DATA_ROOT} for images...")
    
    # Walk through every folder to find images
    for root, dirs, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith('.jpg'):
                # filename format is: 112_highlights_36122.jpg
                parts = file.split('_')
                
                # We need the part in the middle (the word)
                if len(parts) >= 2:
                    label = parts[1]
                    
                    # Get the full valid path
                    full_path = os.path.join(root, file)
                    
                    # Check if label is a valid word (not empty)
                    if label:
                        valid_files.append(f"{full_path} {label}")

    if len(valid_files) == 0:
        print("❌ No images found! Please check the DATA_ROOT path in the script.")
        return

    # Shuffle to mix up the folders
    random.shuffle(valid_files)
    
    # Split: 90% for training, 10% for testing
    split_idx = int(len(valid_files) * 0.9)
    train_data = valid_files[:split_idx]
    val_data = valid_files[split_idx:]
    
    # Save the files
    with open(OUTPUT_TRAIN, 'w') as f:
        f.write('\n'.join(train_data))
        
    with open(OUTPUT_VAL, 'w') as f:
        f.write('\n'.join(val_data))
        
    print(f"✅ Success! Created correct label files.")
    print(f"   - Training images: {len(train_data)}")
    print(f"   - Validation images: {len(val_data)}")

if __name__ == "__main__":
    create_annotation_files()