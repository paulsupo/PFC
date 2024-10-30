import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim

class TTSModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(TTSModel, self).__init__()
        # Definimos las capas con Dropout
        self.fc1 = nn.Linear(151, 100)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(100, 50)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(50, 50)
        
        self.train_losses = []
        self.val_losses = []
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Verificar y ajustar la dimensión de entrada si es necesario
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != 151:
            x = nn.functional.pad(x, (0, 151 - x.shape[1]), "constant", 0)
        
        # Paso hacia adelante con capas de Dropout
        x = self.fc1(x)
        x = self.dropout1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        x = nn.functional.relu(x)
        
        return self.output_layer(x)

    def train_model(self, dataset, val_dataset, tokenizer, encoder, vocoder, epochs, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_train_loss = 0.0

            with tqdm(total=len(dataset), desc=f"Procesando epoch {epoch + 1}/{epochs}", unit=" audio") as progress_bar:
                for i, (audio, text) in enumerate(dataset):
                    semantic_tokens = tokenizer.tokenize(text)
                    semantic_tokens = torch.tensor(semantic_tokens, dtype=torch.float32).to(device)

                    acoustic_tokens = encoder.encode(semantic_tokens.cpu().numpy())
                    acoustic_tokens = torch.tensor(acoustic_tokens, dtype=torch.float32).to(device)

                    output = self.forward(acoustic_tokens)

                    # Asegurarse de que target_tokens tenga el mismo tamaño que output
                    target_tokens = torch.tensor(audio[:output.shape[1]], dtype=torch.float32).to(device)
                    target_tokens = target_tokens.view(1, -1)  # Cambia el tamaño para que coincida con [1, 50]

                    optimizer.zero_grad()
                    train_loss = self.criterion(output, target_tokens)
                    epoch_train_loss += train_loss.item()

                    train_loss.backward()
                    optimizer.step()

                    # Actualizar la barra de progreso
                    progress_bar.update(1)

                    if i == len(dataset) - 1:
                        output_file = f"output_audios/output_audio_epoch{epoch + 1}_sample{i}.wav"
                        vocoder.synthesize(output.cpu().detach().numpy(), output_file)

            avg_train_loss = epoch_train_loss / len(dataset)
            self.train_losses.append(avg_train_loss)

            # Calcular y almacenar la pérdida de validación
            val_loss = self.validate_model(val_dataset, tokenizer, encoder, device)
            self.val_losses.append(val_loss)

        print("Entrenamiento completado.")
        self.plot_losses()

    def validate_model(self, val_dataset, tokenizer, encoder, device):
        val_loss = 0.0
        with torch.no_grad():
            for audio, text in val_dataset:
                semantic_tokens = tokenizer.tokenize(text)
                semantic_tokens = torch.tensor(semantic_tokens, dtype=torch.float32).to(device)

                acoustic_tokens = encoder.encode(semantic_tokens.cpu().numpy())
                acoustic_tokens = torch.tensor(acoustic_tokens, dtype=torch.float32).to(device)

                output = self.forward(acoustic_tokens)
                
                # Asegurarse de que target_tokens tenga el mismo tamaño que output
                target_tokens = torch.tensor(audio[:output.shape[1]], dtype=torch.float32).to(device)
                target_tokens = target_tokens.view(1, -1)  # Cambia el tamaño para que coincida con [1, 50]

                loss = self.criterion(output, target_tokens)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataset)
        print(f"Pérdida de validación: {avg_val_loss}")
        return avg_val_loss

    def plot_losses(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label="Pérdida de entrenamiento", color="blue")
        plt.plot(epochs, self.val_losses, label="Pérdida de validación", color="orange")

        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.title("Pérdida durante el entrenamiento y la validación")
        plt.legend()
        plt.grid(True)
        plt.show()