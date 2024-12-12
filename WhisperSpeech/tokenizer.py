import numpy as np

class TextTokenizer:
    def tokenize(self, text):
        # Validar que el texto sea un string
        if not isinstance(text, str):
            raise ValueError(f"Expected a string for text, but got {type(text).__name__}")

        # Convierte el texto en tokens
        return np.array([ord(char) for char in text], dtype=np.float32) / 255.0