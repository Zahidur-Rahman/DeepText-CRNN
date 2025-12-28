import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
from config import Config as cfg

class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        target_w, target_h = self.size
        
        # Keep aspect ratio
        new_w = int(target_h * w / h)
        img = img.resize((new_w, target_h), self.interpolation)
        
        # Pad with white if too small
        if new_w < target_w:
            padding_needed = target_w - new_w
            img = F.pad(img, (0, 0, padding_needed, 0), fill=255) 
        # Crop if too big
        elif new_w > target_w:
            img = img.crop((0, 0, target_w, target_h))

        img = F.to_tensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class OCRDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.alphabet = cfg.ALPHABET
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.alphabet)}
        
        # Limit max characters to fit in CNN output
        self.max_label_len = 20 
        
        print(f"🔎 Scanning {label_file}...")
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        skipped_long = 0
        skipped_char = 0
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            parts = line.split(' ')
            
            # --- PARSING LOGIC ---
            if len(parts) >= 2:
                # Type A: "./path/to/img.jpg Label"
                filename = parts[0]
                label_text = " ".join(parts[1:])
            else:
                # Type B: "32_annottred_7585.jpg" (Embedded Label)
                filename = line
                base_name = os.path.basename(filename)
                
                try:
                    # Split by underscore
                    name_parts = base_name.split('_')
                    
                    # Robust Fix: Take everything between first and last underscore
                    # This handles "32_annottred_7585.jpg" -> "annottred"
                    # And also "45_New_York_882.jpg" -> "New_York"
                    if len(name_parts) >= 3:
                        label_text = "_".join(name_parts[1:-1])
                    else:
                        # Fallback if filename is weird (e.g. "word.jpg")
                        label_text = os.path.splitext(base_name)[0]
                except:
                    continue

            # --- FILTER 1: Skip if label is too long for the model ---
            if len(label_text) > self.max_label_len:
                skipped_long += 1
                continue

            # --- FILTER 2: Skip if label has unknown characters ---
            valid_label = True
            for char in label_text:
                if char not in self.char2idx:
                    valid_label = False
                    break
            if not valid_label:
                skipped_char += 1
                continue

            # --- RESOLVE PATH ---
            if os.path.exists(filename):
                full_path = filename
            else:
                full_path = os.path.join(root_dir, filename)
                if not os.path.exists(full_path):
                    continue
            
            self.image_paths.append(full_path)
            self.labels.append(label_text)
        
        print(f"✅ Loaded {len(self.image_paths)} images.")
        print(f"⚠️ Skipped {skipped_long} labels (too long) and {skipped_char} labels (unknown chars).")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        try:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
                
            label_text = self.labels[index]
            label_indices = []
            for char in label_text:
                if char in self.char2idx:
                    label_indices.append(self.char2idx[char])
            
            return image, torch.IntTensor(label_indices), len(label_indices)
        except Exception:
            # Return dummy tensor if file is corrupt
            return torch.zeros((1, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)), torch.IntTensor([]), 0

def alignCollate(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.cat(labels)
    label_lengths = torch.IntTensor(lengths)
    
    # --- CRITICAL FIX: 24 ---
    # The CNN reduces 100px width to exactly 24 steps.
    # If we put 25 here, the Loss function crashes (NaN).
    input_lengths = torch.full(size=(len(batch),), fill_value=24, dtype=torch.int32)
    
    return images, labels, input_lengths, label_lengths

resize_normalize = ResizeNormalize((cfg.IMG_WIDTH, cfg.IMG_HEIGHT))