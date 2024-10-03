# generate_audio.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import os

model_path = 'tacotron_model.h5' 
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"No se encontró el archivo de modelo: {model_path}. Asegúrate de que el archivo está en el directorio correcto.")

print(f"Cargando el modelo desde {model_path}...")
tacotron_model = load_model(model_path)
print("Modelo cargado correctamente.")

char_to_idx = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz '.,?!-")}
idx_to_char = {i: c for c, i in char_to_idx.items()}

def text_to_sequence(text, char_to_idx):
    """Convierte el texto en una secuencia de índices según el diccionario de caracteres."""
    sequence = [char_to_idx.get(c, char_to_idx[" "]) for c in text.lower() if c in char_to_idx]
    return np.array(sequence)

def generate_spectrogram(text, model, char_to_idx):
    """Genera un espectrograma a partir de un texto usando el modelo Tacotron."""
    input_sequence = text_to_sequence(text, char_to_idx)
    
    if len(input_sequence) < 10:
        print("El texto de entrada es muy corto, añadiendo padding para evitar errores en el modelo.")
        input_sequence = np.pad(input_sequence, (0, 10 - len(input_sequence)), 'constant')
    
    input_sequence = np.expand_dims(input_sequence, axis=0)
    
    print(f"Generando espectrograma para el texto: {text}")
    spectrogram = model.predict(input_sequence)
    print("Espectrograma generado.")
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram[0], sr=24000, x_axis='time', y_axis='mel')
    plt.title(f'Espectrograma generado a partir del texto: "{text}"')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Tiempo')
    plt.ylabel('Frecuencia')
    plt.show()
    
    return spectrogram

def spectrogram_to_waveform(spectrogram):
    """Convierte el espectrograma Mel en una forma de onda usando Griffin-Lim."""
    spectrogram = np.squeeze(spectrogram).T
    
    print(f"Dimensiones del espectrograma antes de la conversión: {spectrogram.shape}")
    
    mel_spectrogram = librosa.db_to_amplitude(spectrogram)
    
    print("Convirtiendo el espectrograma a una forma de onda usando Griffin-Lim...")
    waveform = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=24000, n_iter=100)
    print("Forma de onda generada.")
    return waveform


input_text = "hello world"  

spectrogram = generate_spectrogram(input_text, tacotron_model, char_to_idx)

waveform = spectrogram_to_waveform(spectrogram)

print(f"Longitud de la forma de onda generada: {len(waveform)} muestras")

output_audio_path = "output_audio.wav"
sf.write(output_audio_path, waveform, samplerate=24000) 
print(f"Archivo de audio guardado en: {output_audio_path}")

plt.figure(figsize=(10, 4))
librosa.display.waveshow(waveform, sr=24000)
plt.title(f'Forma de onda generada a partir del texto: "{input_text}"')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.show()