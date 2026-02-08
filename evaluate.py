import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

# Import local modules
from utils import get_device, set_seed
from model import SleepBiLSTM_Energy
from dataset import SleepDatasetEnergy
from postprocess import apply_post_processing, calculate_ahi


def evaluate_pipeline(model_path, mel_dir, label_dir, valid_ranges_json, batch_size=1):
    device = get_device()

    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = SleepBiLSTM_Energy(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Prepare Data
    with open(valid_ranges_json, 'r') as f:
        valid_ranges = json.load(f)

    test_files = [f for f in os.listdir(mel_dir) if f.endswith('.npy')]
    # You might want to filter this list based on a 'test' split in dataset_split.json

    dataset = SleepDatasetEnergy(test_files, mel_dir, label_dir, valid_ranges, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Inference
    print("Running Inference...")
    all_preds_frame = []
    all_labels_frame = []

    patient_events = {}  # Store events per patient for AHI
    patient_durations = {}  # Store total duration per patient

    with torch.no_grad():
        for mel, energy, labels, masks, patient_ids in tqdm(loader):
            mel, energy = mel.to(device), energy.to(device)

            # Forward Pass
            outputs = model(mel, energy)  # [B, T, 3]
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()  # [B, T]

            # Collect Frame-level Data (Masked)
            labels_np = labels.cpu().numpy()
            masks_np = masks.cpu().numpy().astype(bool)

            for i in range(len(patient_ids)):
                pid = patient_ids[i]

                # Get valid frames only
                valid_p = preds[i][masks_np[i]]
                valid_l = labels_np[i][masks_np[i]]

                all_preds_frame.extend(valid_p)
                all_labels_frame.extend(valid_l)

                # --- Post-Processing per segment ---
                # Note: In a real scenario, you should stitch all segments for a patient
                # BEFORE post-processing. Here we process per-window for demonstration,
                # but ideally, you merge all windows for Patient X then run post-processing.

                # Assuming simple accumulation for AHI estimation (Approximate):
                # Calculate duration of this valid segment
                segment_duration_hours = (len(valid_l) * 0.08) / 3600.0  # 80ms per frame

                if pid not in patient_durations:
                    patient_durations[pid] = 0.0
                    patient_events[pid] = []

                patient_durations[pid] += segment_duration_hours

                # Apply Post-processing to get discrete events
                # Note: 'duration_sec' should match the actual valid length
                valid_duration = len(valid_p) * 0.08
                events = apply_post_processing(valid_p, None, duration_sec=valid_duration)
                patient_events[pid].extend(events)

    # 4. Frame-level Report
    print("\n=== Frame-level Classification Report ===")
    print(classification_report(all_labels_frame, all_preds_frame,
                                target_names=['Normal', 'Hypopnea', 'Apnea'], digits=4))

    # 5. Patient-level AHI Report
    print("\n=== Patient-level AHI Estimation ===")
    ahi_errors = []
    print(f"{'Patient ID':<15} | {'Est. AHI':<10}")
    print("-" * 30)

    for pid, duration in patient_durations.items():
        if duration < 0.1: continue  # Skip if too short

        # Calculate AHI
        events = patient_events[pid]
        ahi = calculate_ahi(events, duration)

        print(f"{pid:<15} | {ahi:.2f}")
        # If you have Ground Truth AHI, you can calculate MAE/Correlation here.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--mel_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--valid_ranges', type=str, default='../valid_ranges.json')
    args = parser.parse_args()

    set_seed(42)
    evaluate_pipeline(args.model_path, args.mel_dir, args.label_dir, args.valid_ranges)