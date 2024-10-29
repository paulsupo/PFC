# generate_audio.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import write  


model_path = 'tacotron_model.keras' 
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"No se encontró el archivo de modelo: {model_path}. Asegúrate de que el archivo está en el directorio correcto.")

print(f"Cargando el modelo desde {model_path}...")
tacotron_model = load_model(model_path)
print("Modelo cargado correctamente.")

# Diccionario de caracteres utilizado para tokenizar el texto (basado en train.py)
char_to_idx = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz '.,?!-")}
idx_to_char = {i: c for c, i in char_to_idx.items()}


# Se convierte el texto en una secuencia de índices según el diccionario de caracteres
def text_to_sequence(text, char_to_idx, max_len=100):
    # Convertimos el texto a minúsculas y luego a índices según el diccionario
    sequence = [char_to_idx.get(c, char_to_idx[" "]) for c in text.lower() if c in char_to_idx]
    
    # Aplicar padding para asegurar que la secuencia tenga max_len
    sequence = pad_sequences([sequence], maxlen=max_len, padding='post')[0]
    return np.array(sequence)

# Generamos un espectrograma a partir de un texto
def generate_spectrogram(text, model, char_to_idx):
    
    # Convertir el texto a una secuencia de índices con padding
    input_sequence = text_to_sequence(text, char_to_idx, max_len=100)
    
    # Añadir dimensión extra para representar el lote (batch_size = 1)
    input_sequence = np.expand_dims(input_sequence, axis=0)
    
    # Predecir el espectrograma utilizando el modelo entrenado
    print(f"Generando espectrograma para el texto: {text}")
    spectrogram = model.predict(input_sequence)
    
    # Remover dimensiones adicionales si es necesario
    spectrogram = np.squeeze(spectrogram)
    print(f"Espectrograma generado con forma: {spectrogram.shape}")
    return spectrogram

# De espectrograma a forma de onda
def spectrogram_to_waveform(spectrogram):
    # Asegurarse de que el espectrograma tenga dos dimensiones
    if len(spectrogram.shape) == 1:
        spectrogram = np.expand_dims(spectrogram, axis=0)
    elif len(spectrogram.shape) == 3:
        spectrogram = np.squeeze(spectrogram, axis=-1)

    # Convertir el espectrograma Mel de decibelios a amplitud
    mel_spectrogram = librosa.db_to_amplitude(spectrogram)
    
    # Reconstruir el audio a partir del espectrograma Mel usando Griffin-Lim
    print("Convirtiendo el espectrograma a una forma de onda usando Griffin-Lim...")
    waveform = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=24000, n_iter=50)
    print("Forma de onda generada.")
    return waveform

# Definir el texto de entrada
input_text = "hello world"  

# Generar el espectrograma a partir del texto
spectrogram = generate_spectrogram(input_text, tacotron_model, char_to_idx)

# Convertir el espectrograma en una forma de onda
waveform = spectrogram_to_waveform(spectrogram)

# Guardar el audio
output_audio_path = "output_audio.wav"
write(output_audio_path, 24000, waveform) 
print(f"Archivo de audio guardado en: {output_audio_path}")

# Mostrar el waveform generado
plt.figure(figsize=(10, 4))
librosa.display.waveshow(waveform, sr=24000)
plt.title(f'Forma de onda generada a partir del texto: "{input_text}"')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.show()