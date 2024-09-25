from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


def calculate_r2(output, target):
    r2 = r2_score(target, output, multioutput='variance_weighted')
    return r2


def calculate_MSE(output, target):
    mse = mean_squared_error(target, output)
    return mse


def carica_dataset():

    # Caricamento del dataset dai files CSV
    df = pd.read_csv('../Datasets/train.csv')
    vs = pd.read_csv('../Datasets/validation.csv')
    ts = pd.read_csv('../Datasets/test.csv')

    # Elimino le righe dove la colonna 'Current ball velocity y' vale 0
    df = df.query('`Current ball velocity y` != 0')
    vs = vs.query('`Current ball velocity y` != 0')
    ts = ts.query('`Current ball velocity y` != 0')

    # Estrazione delle feature e del target dal DataFrame pandas
    X = df[['Current ball position x', 'Current ball position y', 'Current ball position z']].values

    Y = df[['Current joint positions 0', 'Current joint positions 1',
            'Current joint positions 2', 'Current joint positions 3',
            'Current joint positions 4', 'Current joint positions 5',
            'Current joint positions 6', 'Current joint positions 7',
            'Current joint positions 8', 'Current joint positions 9',
            'Current joint positions 10']].values

    XV = vs[['Current ball position x', 'Current ball position y', 'Current ball position z']].values

    YV = vs[['Current joint positions 0', 'Current joint positions 1',
             'Current joint positions 2', 'Current joint positions 3',
             'Current joint positions 4', 'Current joint positions 5',
             'Current joint positions 6', 'Current joint positions 7',
             'Current joint positions 8', 'Current joint positions 9',
             'Current joint positions 10']].values

    XT = ts[['Current ball position x', 'Current ball position y', 'Current ball position z']].values

    YT = ts[['Current joint positions 0', 'Current joint positions 1',
             'Current joint positions 2', 'Current joint positions 3',
             'Current joint positions 4', 'Current joint positions 5',
             'Current joint positions 6', 'Current joint positions 7',
             'Current joint positions 8', 'Current joint positions 9',
             'Current joint positions 10']].values

    return X, Y, XV, YV, XT, YT


def addestra_MLP(early_stopping_patience):

    X, Y, XV, YV, XT, YT = carica_dataset()

    # Conversione dei dati in tensori PyTorch
    X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(XV, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(YV, dtype=torch.float32).to(device)

    # Creazione dei TensorDataset e DataLoader
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Parametri della rete:
    input_size = X_train_tensor.shape[1]  # Numero di caratteristiche
    print(input_size)
    hidden_size1 = 256
    output_size = Y_train_tensor.shape[1]  # Numero di output
    print(output_size)

    # Creazione del modello:
    model = MLP(input_size, hidden_size1, output_size, dropout_prob=0.5).to(device)

    # Se esiste un modello salvato e caricalo, sennò crealo
    if os.path.exists('MLP_Model.pth'):
        model.load_state_dict(torch.load('MLP_Model.pth', map_location=device))
        print("Modello caricato correttamente.")
    else:
        print("Modello non trovato, si parte da zero")

    # Definizione della loss function e dell'ottimizzatore con weight decay
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    ####################################################################################################################

    # LOOP di ADDESTRAMENTO:

    num_epochs = 100

    # Early stopping
    best_val_loss = np.inf
    patience = 0

    for epoch in range(num_epochs):
        model.train()  # Modalità di addestramento
        train_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)  # Sposta i tensori su GPU/CPU
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Calcolo della loss di validazione e dell'accuratezza
        model.eval()  # Modalità di valutazione
        val_loss = 0
        val_outputs = []

        with torch.no_grad():

            for data, target in val_loader:
                data, target = data.to(device), target.to(device)  # Sposta i tensori su GPU/CPU
                output = model(data)
                val_loss += criterion(output, target).item()
                val_outputs.append(output.cpu().numpy())  # Sposta output su CPU per R2

        val_loss /= len(val_loader)

        # Calcolo R2 sul set di VALIDAZIONE:
        val_outputs = np.concatenate(val_outputs, axis=0)
        r2_val = calculate_r2(val_outputs, YV)
        mse_val = calculate_MSE(val_outputs, YV)

        print(f'Epoch {epoch+1}/{num_epochs}, \nTraining Loss: {train_loss}, \nValidation Loss: {val_loss}, \nValidation R2: {r2_val}, MSE: {mse_val}')

        # Check per early stopping:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience > early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1} with validation loss: {val_loss}')
                break

    # Salvataggio del modello addestrato
    torch.save(model.state_dict(), 'MLP_Model.pth')


if __name__ == '__main__':
    addestra_MLP(early_stopping_patience=15)
