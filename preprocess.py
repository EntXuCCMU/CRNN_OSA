import os
import argparse
import numpy as np
import librosa
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def extract_features(audio_path, output_dir, sample_rate=16000,
                     n_mels=128, n_fft=1024, hop_length=160,
                     window_size=60, stride=30):
    """
    Implements Section 2.2.2: Feature Extraction
    - Resample to 16kHz
    - Log Mel-spectrogram (128 bands, 10ms hop)
    - Segmentation: 60s window, 30s stride (50% overlap)
    """
    filename = os.path.basename(audio_path).replace('.wav', '')

    # 1. Load Audio
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return

    # 2. Compute Mel-Spectrogram
    # hop_length=160 at 16kHz = 10ms resolution
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)

    # Convert to Log scale (dB)
    log_S = librosa.power_to_db(S, ref=np.max)

    # Transpose to [Time, Freq] -> [T, 128]
    log_S = log_S.T

    # 3. Slice into windows
    # Window length in frames: 60s / 0.01s = 6000 frames
    frames_per_window = int(window_size * (sample_rate / hop_length))
    step_size = int(stride * (sample_rate / hop_length))

    total_frames = log_S.shape[0]

    # Ensure output directory exists
    save_dir = os.path.join(output_dir, "mel_spectrograms")
    os.makedirs(save_dir, exist_ok=True)

    for start_idx in range(0, total_frames - frames_per_window + 1, step_size):
        end_idx = start_idx + frames_per_window
        segment = log_S[start_idx:end_idx, :]  # Shape: [6000, 128]

        # Calculate timestamp for file naming
        start_time_sec = start_idx * (hop_length / sample_rate)

        save_name = f"{filename}_{start_time_sec:.2f}.npy"
        np.save(os.path.join(save_dir, save_name), segment.astype(np.float32))


def process_dataset(data_dir, output_dir, workers=4):
    audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]

    print(f"Found {len(audio_files)} audio files. Starting extraction...")

    # Parallel processing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for wav_path in audio_files:
            futures.append(executor.submit(extract_features, wav_path, output_dir))

        for _ in tqdm(futures):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Audio to Log Mel Spectrograms")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to folder containing .wav files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder for .npy files')
    parser.add_argument('--workers', type=int, default=4, help='Number of CPU cores to use')

    args = parser.parse_args()

    process_dataset(args.data_dir, args.output_dir, args.workers)