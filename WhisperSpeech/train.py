import torch
from dataset import LJSpeechDataset
from tokenizer import TextTokenizer
from encoder import AcousticEncoder
from vocoder import Vocoder
from model import TTSModel
from torch.utils.data import random_split

# Configuración de parámetros
epochs = 5

# Verificar si CUDA está disponible y configurar el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA disponible. Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA no está disponible. Usando CPU.")

# Inicialización de los componentes
metadata_path = "LJSpeech-1.1/metadata.csv"
wavs_path = "LJSpeech-1.1/wavs"
dataset = LJSpeechDataset(metadata_path, wavs_path)

# Dividir el conjunto de datos en entrenamiento y validación (por ejemplo, 80-20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Inicialización del tokenizador, codificador y vocoder
tokenizer = TextTokenizer()
encoder = AcousticEncoder()
vocoder = Vocoder()

# Inicializar el modelo y entrenarlo
model = TTSModel()
model.train_model(train_dataset, val_dataset, tokenizer, encoder, vocoder, epochs=epochs)
