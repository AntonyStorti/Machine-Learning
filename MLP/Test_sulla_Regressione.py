from MLP.MLP_Supervised_Training import MLP
import torch


def main():

    # Carico il modello
    model = MLP(3, 256, 11)
    model.load_state_dict(torch.load("MLP_Model.pth", map_location=torch.device('cpu')))
    model.eval()  # Metto il modello in modalità valutazione

    # Dati di input di test
    input_data = torch.tensor([[0.09600278493262648, 0.04373932662760982, 1.0408854139914077]], dtype=torch.float32)

    # Passo gli input attraverso il modello per ottenere le predizioni --> Disabilito calcolo dei gradienti
    with torch.no_grad():
        predictions = model(input_data)

    # Valori attesi:
    expected_values = torch.tensor([
        [-0.30000011103118973, 0.008276522350872503, 3.141634389609734, 0.08268948806696301,
         0.009762533878662383, 0.7666174464490809, 0.0014105674438350383, 0.7240330619023801,
         0.0003073685295914808, 0.9257157577691185, 1.5716772920715176]
    ], dtype=torch.float32)

    # Calcolo la differenza tra predizioni e valori attesi
    difference = predictions - expected_values

    # Stampo i risultati:
    print("Predictions:", predictions)
    print("Expected values:", expected_values)
    print("Difference:", difference)

    ####################################################################################################################

    # RISULTATI:

    # 128/50 --> 5 prese --> Margine +- 3%

    # 200/50 --> 8 prese --> Margine +- 3%

    # 256/50 --> 8 prese --> Margine +- 3% --> OK
    # 256/75 --> 7 prese --> Margine +- 3%

    # 300/50 --> 5 prese --> Margine +- 3%
    # 350/50 --> 8 prese --> Margine +- 3%

    # 450/50 --> 5 prese --> Margine +- 3%

    # 512/25 --> 7 prese --> Margine +- 3%
    # 512/50 --> 7 prese --> Margine +- 3%


if __name__ == "__main__":
    main()
