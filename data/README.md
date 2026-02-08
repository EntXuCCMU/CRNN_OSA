# Data Directory
This directory is intended to store the dataset files used in the study.
**⚠️ IMPORTANT:** Due to patient privacy regulations (GDPR/Ethics) and GitHub file size limits, **the raw audio data (.wav) and processed features (.npy) are NOT included in this repository.**
## 📂 Directory Structure
To run the training and evaluation scripts, please organize your local data folder as follows:
```text
data/
├── raw_audio/                 # Place the downloaded .wav files here
│   ├── patient_1.wav
│   ├── patient_2.wav
│   └── ...
│
├── processed/                 # Output folder for preprocess.py
│   ├── mel_spectrograms/      # Contains .npy files
│   └── labels/                # Contains .json labels
│
├── dataset_split.json         # Train/Val/Test split configuration
└── valid_ranges.json          # ROI ranges (optional)
```

Data Sources & Access
1. Internal Dataset (Sismanoglio Cohort)
The internal dataset used in this study is the **PSG-Audio Dataset**, collected at Sismanoglio-Amalia Fleming General Hospital. It is publicly available.
* Download Link: [Science Data Bank (PSG-Audio)](https://www.scidb.cn/en/detail?dataSetId=778740145531650048) or [PhysioNet](https://www.google.com/search?q=https://physionet.org/content/psg-audio/)
* Description: Contains full-night PSG recordings with synchronized audio from 286 subjects.
* Citation: If you use this dataset, please cite the original authors:
> Korompili, G., Amfilochiou, A., Kokkalas, L. et al. PSG-Audio, a scored polysomnography dataset with simultaneous audio recordings for sleep apnea studies. *Sci Data* 8, 197 (2021). https://www.google.com/search?q=https://doi.org/10.1038/s41597-021-00977-w
2. External Dataset (Beijing Tongren Hospital)
The external validation dataset was collected from the Sleep Medicine Center of Beijing Tongren Hospital.
* Access Status: Restricted / Private.
* Note: Due to ethical restrictions and privacy protection policies approved by the IRB of Beijing Tongren Hospital (Approval No. TREC2023-KY015), this dataset is **not publicly available**.
* Collaboration: Researchers interested in this dataset for validation purposes may contact the corresponding author of the paper to discuss potential collaboration and data usage agreements.

Preprocessing Instructions
After downloading the public dataset:
1. Extract the `.wav` files into `data/raw_audio/`.
2. Run the preprocessing script to generate Log Mel-spectrograms and Energy Profiles:
```bash
cd ..
python preprocess.py --data_dir data/raw_audio --output_dir data/processed
```
