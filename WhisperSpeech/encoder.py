import numpy as np

class AcousticEncoder:
    def encode(self, semantic_tokens):
        # Se codifica los tokens semánticos en tokens acústicos
        return np.tanh(semantic_tokens)