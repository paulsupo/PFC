import os
import numpy as np
import librosa

class LJSpeechDataset:
    def __init__(self, metadata_path, wavs_path):
        self.metadata_path = metadata_path
        self.wavs_path = wavs_path
        self.data = self._load_metadata()

    def _load_metadata(self):
        data = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                wav_file = os.path.join(self.wavs_path, f"{parts[0]}.wav")
                text = parts[1]
                data.append((wav_file, text))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_path, text = self.data[idx]
        audio, _ = librosa.load(wav_path, sr=16000)
        return audio, text