import numpy as np

class TextTokenizer:
    def tokenize(self, text):
        # Convierte el texto en tokens
        return np.array([ord(char) for char in text], dtype=np.float32) / 255.0