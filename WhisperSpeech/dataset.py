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

                # Validar que el texto sea un string
                if not isinstance(text, str):
                    raise ValueError(f"Expected a string for text in metadata, but got {type(text).__name__}")

                data.append((wav_file, text))
        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_path, text = self.data[idx]
        audio, _ = librosa.load(wav_path, sr=16000)

        # Rellenar o recortar el audio a una longitud fija
        max_length = 16000 
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)), mode="constant")
        else:
            audio = audio[:max_length]

        return audio, text
