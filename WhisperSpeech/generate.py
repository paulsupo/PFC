import torch
import os
from tokenizer import TextTokenizer
from encoder import AcousticEncoder
from vocoder import Vocoder
from model import TTSModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA disponible. Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA no está disponible. Usando CPU.")


def generate_audio_from_text(text, output_file="generated_output.wav"):
    # Inicializar modelo
    tokenizer = TextTokenizer()
    encoder = AcousticEncoder()
    vocoder = Vocoder()
    
    # 1. Tokenizar el texto
    semantic_tokens = tokenizer.tokenize(text)
    semantic_tokens = torch.tensor(semantic_tokens).to(device)

    # 2. Codificar a tokens acústicos
    acoustic_tokens = encoder.encode(semantic_tokens.cpu().numpy())

    # 3. Sintetizar y guardar el audio
    vocoder.synthesize(acoustic_tokens, output_file)
    print(f"Audio generado guardado en: {output_file}")

# Input
text_input = "Este es un ejemplo de síntesis de voz usando el modelo entrenado."

# Output
output_audio_file = "generated_output5_2.wav"

# Generar audio
generate_audio_from_text(text_input, output_audio_file)