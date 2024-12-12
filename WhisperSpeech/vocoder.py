import numpy as np
import soundfile as sf

class Vocoder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def synthesize(self, acoustic_tokens, output_file="output_audio.wav"):
        audio_signal = np.sin(2 * np.pi * acoustic_tokens)
        sf.write(output_file, audio_signal, self.sample_rate)
        print(f"Audio guardado en {output_file}")