import torch
from dataset import LJSpeechDataset
from tokenizer import TextTokenizer
from encoder import AcousticEncoder
from vocoder import Vocoder
from model import TTSModel
from torch.utils.data import DataLoader, random_split
import os

# Configuración de parámetros
epochs = 5
batch_size = 16
validation_split = 0.1
test_split = 0.1
model_save_path = "tts_model.pth"

def split_dataset(dataset, validation_split, test_split):
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    return random_split(dataset, [train_size, val_size, test_size])


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


# Dividir el conjunto de datos
dataset_train, dataset_val, dataset_test = split_dataset(dataset, validation_split, test_split)

def collate_fn(batch):
    audios, texts = zip(*batch)

    # Convertir a tensores y asegurarse de que las dimensiones sean consistentes
    audios = torch.tensor(audios)
    return audios, texts

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset_val, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate_fn)


# Inicializar componentes
tokenizer = TextTokenizer()
encoder = AcousticEncoder()
vocoder = Vocoder()
model = TTSModel()


# Entrenar el modelo
print("Iniciando el entrenamiento...")
model.train_model(dataset_train, dataset_val, tokenizer, encoder, vocoder, epochs)

# Guardar el modelo al finalizar
torch.save(model.state_dict(), model_save_path)
print(f"Modelo guardado en {model_save_path}")