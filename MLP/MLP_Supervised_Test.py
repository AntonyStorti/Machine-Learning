from MLP_Supervised_Training import *
import numpy as np
import torch


def main():

    X, Y, XV, YV, XT, YT = carica_dataset()

    X_test_tensor = torch.tensor(XT, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(YT, dtype=torch.float32).to(device)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Parametri della rete:
    input_size = X_test_tensor.shape[1]  # Numero di caratteristiche
    hidden_size1 = 256
    output_size = Y_test_tensor.shape[1]  # Numero di output

    # Creazione del modello:
    model = MLP(input_size, hidden_size1, output_size, dropout_prob=0.5).to(device)

    # Se esiste un modello salvato e caricalo, sennò crealo
    if os.path.exists('MLP_Model.pth'):
        model.load_state_dict(torch.load('MLP_Model.pth', map_location=device))
        print("Modello caricato correttamente.")
    else:
        raise "Modello non trovato !!!"

    # Definizione della loss function e dell'ottimizzatore con weight decay
    criterion = nn.MSELoss()

    # Valutazione del modello sul set di TEST:
    model.eval()
    test_loss = 0
    test_outputs = []

    with torch.no_grad():

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Sposta i tensori su GPU/CPU
            output = model(data)
            test_loss += criterion(output, target).item()
            test_outputs.append(output.cpu().numpy())  # Sposta output su CPU per R2

    test_loss /= len(test_loader)

    # Calcolo R2 sul set di TEST
    test_outputs = np.concatenate(test_outputs, axis=0)
    r2_test = calculate_r2(test_outputs, YT)
    mse_test = calculate_MSE(test_outputs, YT)

    print(f'Test Loss: {test_loss}, Test R2: {r2_test}%, Test MSE: {mse_test}')


if __name__ == '__main__':
    main()
