import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, MaxPooling1D, GRU, Bidirectional, Add, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from tqdm import tqdm  
import matplotlib.pyplot as plt

# Ruta al archivo 'metadata.csv'
file_path = r"C:\Users\paul\Desktop\Tacotron\LJSpeech-1.1\metadata.csv"

# Verificar existencia de 'metadata.csv' y cargar el archivo
if not os.path.exists(file_path):
    raise FileNotFoundError(f"El archivo {file_path} no se encontró. Verifica la ruta y asegúrate de que esté en el directorio correcto.")
else:
    print(f"El archivo {file_path} se encontró correctamente.")
    metadata = pd.read_csv(file_path, delimiter='|', header=None, names=['ID', 'Transcription', 'Normalized'])

# Asegurarnos de tener las columnas esperadas en el CSV
if 'ID' not in metadata.columns or 'Transcription' not in metadata.columns:
    raise ValueError("El archivo 'metadata.csv' debe contener las columnas 'ID' y 'Transcription'.")

# Preprocesamiento de los datos
print("Preprocesando los datos...")
max_len = 100                   # Ajustar longitud máxima de las secuencias para que coincida con la salida del modelo
char_to_index = {char: index for index, char in enumerate(sorted(set("".join(metadata['Transcription'].values))))}
index_to_char = {index: char for char, index in char_to_index.items()}
vocab_size = len(char_to_index)

# Convertir las transcripciones en secuencias de enteros
def text_to_sequence(text):
    return [char_to_index[char] for char in text if char in char_to_index]

metadata['sequence'] = metadata['Transcription'].apply(text_to_sequence)

# Padding y truncamiento de las secuencias para que todas tengan la misma longitud
X = keras.preprocessing.sequence.pad_sequences(metadata['sequence'], maxlen=max_len, padding='post')
y = X   # La entrada y la salida son iguales (autoencoder)

# Asegurar que la salida tenga la forma adecuada (agregamos una dimensión extra)
y = np.expand_dims(y, axis=-1)

# División de los datos entre entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Definiendo el modelo...")
input_text = Input(shape=(max_len,), dtype='int32', name='input_text')  # Ajustar para que la entrada tenga la longitud deseada
x = Embedding(input_dim=vocab_size, output_dim=256)(input_text)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

# Bloque convolucional
for _ in range(10):  # Simplificamos y evitamos desbalance de dimensiones
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)

# Ajustar tamaño para mantener dimensiones
x = MaxPooling1D(pool_size=1)(x)

# Más capas convolucionales
x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
x = Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)

# Ajustar el residual para que coincida con las dimensiones de `x`
residual = Dense(256)(x)
residual = Add()([x, residual])

# Capas GRU bidireccionales
x = Bidirectional(GRU(128, return_sequences=True))(residual)
x = Bidirectional(GRU(128, return_sequences=True))(x)

# Capa de salida ajustada para coincidir con la dimensión de salida deseada
output = TimeDistributed(Dense(1, activation='linear'), name='output')(x)

# Definir y compilar el modelo
tacotron_model = Model(inputs=input_text, outputs=output, name="Tacotron")
tacotron_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Imprimir la estructura del modelo
tacotron_model.summary()

# Definir callbacks para el entrenamiento
callbacks = [
    ModelCheckpoint("tacotron_weights.keras", monitor='val_loss', save_best_only=True, mode='min', verbose=1), 
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, verbose=1)
]

# Entrenamiento del modelo
epocas=50
print("=== Entrenando Época 1/",epocas,"===")
history = tacotron_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epocas,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# Guardar el modelo
tacotron_model.save("tacotron_model.keras") 

# Graficar la pérdida
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento y la validación')
plt.show()