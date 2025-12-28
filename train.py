
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import your local files
from config import Config as cfg
from dataset import OCRDataset, alignCollate, resize_normalize
from model import CRNN

def train():
    # 1. Setup Device
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Device: {device}")
    
    # 2. Load Data (Train AND Val)
    print("⏳ Loading Datasets...")
    
    # --- TRAIN DATASET ---
    train_ds = OCRDataset(cfg.DATA_DIR, cfg.TRAIN_LABEL_FILE, transform=resize_normalize)
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,
        collate_fn=alignCollate,
        pin_memory=True
    )

    # --- VALIDATION DATASET (NEW) ---
    val_ds = OCRDataset(cfg.DATA_DIR, cfg.VAL_LABEL_FILE, transform=resize_normalize)
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False,  # Don't shuffle validation
        num_workers=cfg.NUM_WORKERS,
        collate_fn=alignCollate,
        pin_memory=True
    )
    
    # 3. Setup Model
    n_class = len(cfg.ALPHABET) + 1 
    print(f"🧠 Building Model for {len(cfg.ALPHABET)} characters...")
    
    model = CRNN(img_channel=1, img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH, num_class=n_class)
    model.to(device)
    
    # 4. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    if not os.path.exists(cfg.SAVE_DIR):
        os.makedirs(cfg.SAVE_DIR)
        
    print(f"🚀 Training started for {cfg.EPOCHS} epochs...")

    best_val_loss = float('inf')
    
    for epoch in range(cfg.EPOCHS):
        # ==========================
        # 1. TRAINING LOOP
        # ==========================
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]", unit="batch")
        
        for batch in pbar:
            images, labels, input_lengths, label_lengths = batch
            images = images.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            log_probs = preds.log_softmax(2)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) 
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'T-Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)

        # ==========================
        # 2. VALIDATION LOOP
        # ==========================
        model.eval()
        val_loss = 0
        # tqdm for validation is optional, but nice to have
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Val]  ", unit="batch", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                images, labels, input_lengths, label_lengths = batch
                images = images.to(device)
                labels = labels.to(device)
                input_lengths = input_lengths.to(device)
                label_lengths = label_lengths.to(device)

                preds = model(images)
                log_probs = preds.log_softmax(2)
                loss = criterion(log_probs, labels, input_lengths, label_lengths)
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # ==========================
        # 3. LOGGING & SAVING
        # ==========================
        print(f"📉 Epoch {epoch+1} Result: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save based on VAL loss (this prevents overfitting)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(cfg.SAVE_DIR, cfg.MODEL_NAME)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_path)
            print(f"💾 Saved New Best Model to {save_path}")
        print("-" * 50)

if __name__ == "__main__":
    train()