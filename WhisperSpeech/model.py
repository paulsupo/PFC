import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class TTSModel(torch.nn.Module):
    def __init__(self):
        super(TTSModel, self).__init__()
        self.train_losses = []
        self.val_losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Modelo configurado para usar: {self.device}")

    def train_model(self, train_dataset, val_dataset, tokenizer, encoder, vocoder, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            # Entrenamiento
            for audio, text in tqdm(train_dataset, desc="Entrenando", leave=False):
                semantic_tokens = tokenizer.tokenize(text)
                semantic_tokens = torch.tensor(semantic_tokens).to(self.device)
                acoustic_tokens = encoder.encode(semantic_tokens.cpu().numpy())
                epoch_train_loss += torch.nn.functional.mse_loss(
                    torch.tensor(acoustic_tokens), torch.tensor(audio[:len(acoustic_tokens)])
                ).item()

            avg_train_loss = epoch_train_loss / len(train_dataset)
            self.train_losses.append(avg_train_loss)

            # Validación
            for audio, text in tqdm(val_dataset, desc="Validando", leave=False):
                semantic_tokens = tokenizer.tokenize(text)
                semantic_tokens = torch.tensor(semantic_tokens).to(self.device)
                acoustic_tokens = encoder.encode(semantic_tokens.cpu().numpy())
                epoch_val_loss += torch.nn.functional.mse_loss(
                    torch.tensor(acoustic_tokens), torch.tensor(audio[:len(acoustic_tokens)])
                ).item()

            avg_val_loss = epoch_val_loss / len(val_dataset)
            self.val_losses.append(avg_val_loss)

            print(f"Pérdida de entrenamiento: {avg_train_loss:.4f} | Pérdida de validación: {avg_val_loss:.4f}")

        print("Entrenamiento completado.")
        self.plot_losses()

    def plot_losses(self):
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Pérdida de entrenamiento")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Pérdida de validación")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.title("Pérdida durante el entrenamiento y la validación")
        plt.legend()
        plt.grid(True)
        plt.show()
