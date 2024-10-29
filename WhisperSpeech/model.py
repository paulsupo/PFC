import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class TTSModel:
    def __init__(self):
        # Inicialización de componentes del modelo
        self.train_losses = []
        self.val_losses = []

    def train(self, dataset, tokenizer, encoder, vocoder, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            # Crear la barra de progreso para la época actual
            progress_bar = tqdm(dataset, desc="Procesando audios", unit=" audio")

            for i, (audio, text) in enumerate(progress_bar):
                # Proceso de entrenamiento:
                # 1. Tokenizar el texto
                semantic_tokens = tokenizer.tokenize(text)
                
                # Mover tokens al dispositivo adecuado (GPU)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                semantic_tokens = torch.tensor(semantic_tokens).to(device)

                # 2. Codificar a tokens acústicos
                acoustic_tokens = encoder.encode(semantic_tokens.cpu().numpy())  # Simple encoding

                # Simular una pérdida aleatoria 
                train_loss = np.random.uniform(0.1, 1.0)  # Placeholder para la pérdida de entrenamiento
                val_loss = np.random.uniform(0.1, 1.0)  # Placeholder para la pérdida de validación

                epoch_train_loss += train_loss
                epoch_val_loss += val_loss

                # Cada 1000 muestras, generamos un audio de prueba sin múltiples mensajes
                if i % 1000 == 0:
                    output_file = f"output_audios/output_audio_epoch{epoch+1}_sample{i}.wav"
                    vocoder.synthesize(acoustic_tokens, output_file)
                    progress_bar.set_postfix({"Audio de prueba": output_file})

            # Almacenar la pérdida promedio por época
            avg_train_loss = epoch_train_loss / len(dataset)
            avg_val_loss = epoch_val_loss / len(dataset)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)

        print("Entrenamiento completado.")
        self.plot_losses()

    def plot_losses(self):
        # Graficar las pérdidas de entrenamiento y validación
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Pérdida de entrenamiento")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Pérdida de validación")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.title("Pérdida durante el entrenamiento y la validación")
        plt.legend()
        plt.grid(True)
        plt.show()