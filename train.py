import os
import json
import argparse
import scipy.signal
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# Import local modules
from utils import get_device, set_seed, EarlyStopping, count_parameters
from model import SleepBiLSTM_Energy
from dataset import SleepDatasetEnergy
from loss import HybridEventLoss

# --- Configuration ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=True)

    for mel, energy, labels, masks, _ in pbar:
        mel, energy = mel.to(device), energy.to(device)
        labels, masks = labels.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(mel, energy)
            loss = criterion(outputs, labels, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return running_loss / len(loader)


def validate_metrics(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f"Validating Epoch {epoch}", leave=True)
    with torch.no_grad():
        for mel, energy, labels, masks, _ in pbar:
            mel, energy = mel.to(device), energy.to(device)
            labels, masks = labels.to(device), masks.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(mel, energy)
                loss = criterion(outputs, labels, masks)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            masks_np = masks.cpu().numpy().astype(bool)

            for i in range(len(labels_np)):
                valid_preds = preds[i][masks_np[i]]
                valid_targets = labels_np[i][masks_np[i]]
                all_preds.extend(valid_preds)
                all_targets.extend(valid_targets)

    avg_loss = running_loss / len(loader)

    if len(all_preds) > 0:
        all_preds = scipy.signal.medfilt(all_preds, kernel_size=5).astype(int)

    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    report = classification_report(all_targets, all_preds,
                                   target_names=['Normal', 'Hypopnea', 'Apnea'],
                                   digits=4, zero_division=0)

    print(f"\n[Validation Report - Epoch {epoch}]")
    print(report)
    print(f"Val Loss: {avg_loss:.4f} | Macro F1: {macro_f1:.4f}")
    return avg_loss, macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sleep Apnea Detection Training Script")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--mel_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--split_json', type=str, default='../dataset_split.json')
    parser.add_argument('--valid_ranges_json', type=str, default='../valid_ranges.json')
    parser.add_argument('--save_path', type=str, default='best_model.pth')

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | Seed: {args.seed}")

    # Load Splits
    if not os.path.exists(args.split_json):
        raise FileNotFoundError(f"Config not found: {args.split_json}")
    with open(args.split_json, 'r') as f:
        split_data = json.load(f)
        train_pids, val_pids = set(split_data['train']), set(split_data['val'])
    with open(args.valid_ranges_json, 'r') as f:
        valid_ranges = json.load(f)

    all_files = [f for f in os.listdir(args.mel_dir) if f.endswith('.npy')]
    train_files = [f for f in all_files if f.split('_')[0] in train_pids]
    val_files = [f for f in all_files if f.split('_')[0] in val_pids]
    print(f"Train size: {len(train_files)} | Val size: {len(val_files)}")

    # Datasets & Loaders
    train_dataset = SleepDatasetEnergy(train_files, args.mel_dir, args.label_dir, valid_ranges, augment=True)
    val_dataset = SleepDatasetEnergy(val_files, args.mel_dir, args.label_dir, valid_ranges, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # Model & Optimization
    model = SleepBiLSTM_Energy(num_classes=3).to(device)
    count_parameters(model)

    class_weights = torch.tensor([1.0, 4.0, 2.0]).to(device)
    criterion = HybridEventLoss(alpha=class_weights, gamma=2.0, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.save_path)

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_f1 = validate_metrics(model, val_loader, criterion, device, epoch)

        scheduler.step(val_f1)
        print(f"Epoch [{epoch}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break