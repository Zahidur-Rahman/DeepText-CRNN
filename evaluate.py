import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import OCRDataset, alignCollate, resize_normalize
from model import CRNN
from config import Config as cfg

# --- A simple tool to calculate string similarity ---
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def tensor_to_text(preds, alphabet):
    _, max_index = preds.max(2)
    max_index = max_index.squeeze(1)
    result = []
    for i in range(len(max_index)):
        index = max_index[i].item()
        if index != 0:
            if i == 0 or index != max_index[i-1].item():
                result.append(alphabet[index - 1])
    return "".join(result)

def evaluate_cer(model, dataloader, device, alphabet):
    model.eval()
    
    total_char_distance = 0
    total_char_length = 0
    total_correct_words = 0
    total_words = 0
    
    pbar = tqdm(dataloader, desc="Calculating CER", unit="batch")
    
    with torch.no_grad():
        for batch in pbar:
            images, labels, input_lengths, label_lengths = batch
            images = images.to(device)
            preds = model(images)
            
            batch_size = images.size(0)
            for b in range(batch_size):
                # 1. Decode Prediction
                pred_text = tensor_to_text(preds[:, b:b+1, :], alphabet)
                
                # 2. Decode Label
                start = sum(label_lengths[:b])
                end = start + label_lengths[b]
                real_text = "".join([alphabet[i-1] for i in labels[start:end]])
                
                # 3. Calculate Errors
                dist = levenshtein_distance(pred_text.lower(), real_text.lower())
                total_char_distance += dist
                total_char_length += len(real_text)
                
                if dist == 0:
                    total_correct_words += 1
                
                total_words += 1
                
                # Show live stats in progress bar
                cer = (total_char_distance / total_char_length) * 100 if total_char_length > 0 else 0
                pbar.set_postfix({'CER': f'{cer:.2f}%'})

    final_cer = (total_char_distance / total_char_length) * 100
    accuracy = (total_correct_words / total_words) * 100
    
    return final_cer, accuracy

def main():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Device: {device}")

    # Load Alphabet from TRAIN to ensure consistency
    train_ds = OCRDataset(cfg.DATA_DIR, cfg.TRAIN_LABEL_FILE)
    my_alphabet = train_ds.alphabet
    
    # Load Validation Data
    test_ds = OCRDataset(cfg.DATA_DIR, cfg.VAL_LABEL_FILE, transform=resize_normalize)
    test_ds.alphabet = my_alphabet 
    
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, 
                             num_workers=cfg.NUM_WORKERS, collate_fn=alignCollate)

    # Load Model
    nclass = len(my_alphabet) + 1
    model = CRNN(img_channel=1, img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH, num_class=nclass)
    model.to(device)
    
    checkpoint = torch.load(f"{cfg.SAVE_DIR}/{cfg.MODEL_NAME}", map_location=device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()})
    
    print(f"🧪 Evaluation Started...")
    cer, acc = evaluate_cer(model, test_loader, device, my_alphabet)
    
    print("\n" + "="*30)
    print(f"🎯 Character Error Rate (CER): {cer:.2f}%")
    print(f"   (Lower is better. 10% is decent)")
    print("-" * 30)
    print(f"🏆 Strict Accuracy: {acc:.2f}%")
    print(f"   (Exact match only)")
    print("="*30)

if __name__ == "__main__":
    main()