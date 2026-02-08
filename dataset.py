import os
import json
import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

class SleepDatasetEnergy(Dataset):
    def __init__(self, file_list, mel_dir, label_dir, valid_ranges,
                 target_frames=750, input_time_dim=6000, augment=False):
        self.file_list = file_list
        self.mel_dir = mel_dir
        self.label_dir = label_dir
        self.valid_ranges = valid_ranges
        self.target_frames = target_frames
        self.input_time_dim = input_time_dim
        self.augment = augment

        if self.augment:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
            self.time_mask = T.TimeMasking(time_mask_param=40)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npy_file = self.file_list[idx]
        json_file = npy_file.replace('.npy', '.json')
        patient_id = npy_file.split('_')[0]

        try:
            time_suffix = npy_file.replace('.npy', '').split('_')[-1]
            slice_start_sec = float(time_suffix)
        except:
            slice_start_sec = 0.0

        # Load Mel
        mel_path = os.path.join(self.mel_dir, npy_file)
        mel_data = np.load(mel_path)
        if mel_data.shape[0] > mel_data.shape[1]:
            mel_data = mel_data.T

        if mel_data.shape[1] < self.input_time_dim:
            pad = self.input_time_dim - mel_data.shape[1]
            mel_data = np.pad(mel_data, ((0, 0), (0, pad)), mode='constant')
        else:
            mel_data = mel_data[:, :self.input_time_dim]

        # Energy Profile
        energy_curve = np.mean(mel_data, axis=0)
        e_mean = energy_curve.mean()
        e_std = energy_curve.std() + 1e-6
        energy_curve = (energy_curve - e_mean) / e_std

        mel_tensor = torch.FloatTensor(mel_data).unsqueeze(0)
        energy_tensor = torch.FloatTensor(energy_curve)

        mel_mean = mel_tensor.mean()
        mel_std = mel_tensor.std()
        mel_tensor = (mel_tensor - mel_mean) / (mel_std + 1e-6)

        if self.augment:
            gain = torch.empty(1).uniform_(0.8, 1.2)
            mel_tensor = mel_tensor * gain
            energy_tensor = energy_tensor * gain
            if torch.rand(1) < 0.5: mel_tensor = self.freq_mask(mel_tensor)
            if torch.rand(1) < 0.5: mel_tensor = self.time_mask(mel_tensor)

        # Labels
        label_tensor = torch.zeros(self.target_frames, dtype=torch.long)
        json_path = os.path.join(self.label_dir, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            duration = info.get('window_duration', 60.0)
            for event in info.get('events', []):
                evt_type = int(event['type'])
                start_sec = event['start']
                end_sec = event['end']
                s_idx = int((start_sec / duration) * self.target_frames)
                e_idx = int((end_sec / duration) * self.target_frames)
                s_idx = max(0, s_idx)
                e_idx = min(self.target_frames, e_idx)
                if e_idx > s_idx:
                    label_tensor[s_idx:e_idx] = evt_type

        # Mask
        mask_tensor = torch.ones(self.target_frames, dtype=torch.float32)
        roi_info = self.valid_ranges.get(patient_id, "FULL")
        if roi_info != "FULL":
            roi_start, roi_end = roi_info
            frame_dur = 60.0 / self.target_frames
            frame_times = torch.arange(self.target_frames) * frame_dur + slice_start_sec
            invalid_mask = (frame_times < roi_start) | (frame_times > roi_end)
            mask_tensor[invalid_mask] = 0.0

        return mel_tensor, energy_tensor, label_tensor, mask_tensor, patient_id