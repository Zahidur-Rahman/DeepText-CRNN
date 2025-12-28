import torch
import os
import argparse
from PIL import Image
from config import Config as cfg
from model import CRNN
from dataset import resize_normalize # Reuse the transform from training

def decode_prediction(preds, alphabet):
    """
    Decodes the raw numbers from the model back into text.
    1. Removes blanks (index 0).
    2. Merges duplicate consecutive characters.
    """
    # Get the index with highest probability at each time step
    preds = preds.argmax(dim=2) # Shape: [Time, Batch] -> [Time, 1]
    preds = preds.squeeze(1)    # Shape: [Time]
    
    decoded_text = []
    
    # CTC Decoding Logic
    for i in range(len(preds)):
        idx = preds[i].item()
        
        # 0 is the "Blank" token in CTC
        if idx != 0:
            # Only add if it's different from the previous character (ignoring blanks)
            if i == 0 or idx != preds[i-1].item():
                decoded_text.append(alphabet[idx - 1]) # -1 because 0 is blank
                
    return "".join(decoded_text)

def predict(image_path):
    # 1. Setup Device
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Model
    n_class = len(cfg.ALPHABET) + 1
    model = CRNN(img_channel=1, img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH, num_class=n_class)
    
    checkpoint_path = os.path.join(cfg.SAVE_DIR, cfg.MODEL_NAME)
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found at {checkpoint_path}")
        return

    print(f"⏳ Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Important: Switch to evaluation mode
    
    # 3. Load & Preprocess Image
    try:
        image = Image.open(image_path).convert('L')
        # Use the same resize/normalize logic as training
        image = resize_normalize(image) 
        image = image.unsqueeze(0) # Add batch dimension: [1, 1, H, W]
        image = image.to(device)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # 4. Run Inference
    with torch.no_grad():
        preds = model(image) # Output shape: [Time, Batch, Class]
    
    # 5. Decode
    predicted_text = decode_prediction(preds, cfg.ALPHABET)
    
    print("-" * 40)
    print(f"🖼️  Image:      {image_path}")
    print(f"🤖 Prediction: {predicted_text}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to the image to test')
    args = parser.parse_args()
    
    predict(args.image)