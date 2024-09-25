from client import Client, JOINTS, DEFAULT_PORT
from MLP.MLP_Supervised_Training import MLP
import DDPG.RL_Training
import DDPG.RL_Test
import numpy as np
import torch
import math
import sys


def run(cli, mode):

    j = np.zeros((JOINTS,))
    j[2] = math.pi
    j[10] = math.pi / 2
    j[5] = math.pi / 2
    j[9] = math.pi / 4

    # Carica il modello addestrato MLP
    model = MLP(3, 256, 11)
    model.load_state_dict(torch.load("MLP/MLP_Model.pth", map_location=torch.device('cpu')))
    model.eval()  # Metto il modello in modalità di sola valutazione


    if mode == '--train':
        replay_buffer = DDPG.RL_Training.ReplayBuffer(1500)
        noise, actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer = DDPG.RL_Training.start_RL_Training()

    elif mode == '--test':
        actor_model, critic_model = DDPG.RL_Test.start_RL_Test()


    while True:

        state = cli.get_state()

        # Input che si aspetta la rete MLP
        state_input = [state[17], state[18], state[19]]

        # Input convertito in tensore
        input_data = torch.tensor(state_input, dtype=torch.float32)

        # Passo gli input al modello per ottenere le predizioni
        with torch.no_grad():  # Disabilito il calcolo dei gradienti
            predictions = model(input_data)

        # Invio ai giunti le azioni corrette predette dal MLP
        cli.send_joints(predictions)

        # Quando la palla è stata presa, passo il controllo al RL per lanciarla correttamente
        if state[31] == 1:

            if mode == '--train':

                DDPG.RL_Training.main(state, cli, noise, actor_model, critic_model, target_actor, target_critic, actor_optimizer, critic_optimizer, replay_buffer)

            elif mode == '--test':

                DDPG.RL_Test.main(state, cli, actor_model, critic_model)


def main():

    if len(sys.argv) < 3 or sys.argv[-1] not in ['--train', '--test']:
        print("Utilizzo corretto: python client_example.py player_name [port] [host] --train/--test")
        return

    name = sys.argv[1]

    port = DEFAULT_PORT
    if len(sys.argv) > 3:
        try:
            port = int(sys.argv[2])
        except ValueError:
            print(f"Errore: {sys.argv[2]} non è un numero di porta valido.")
            return

    host = 'localhost'
    if len(sys.argv) > 3:
        host = sys.argv[3]

    mode = sys.argv[-1]

    cli = Client(name, host, port)
    run(cli, mode)


if __name__ == '__main__':

    '''
    Utilizzo:
    > python client_example.py player_name --train/--test [port] [host]
    Default parameters: 
     port: client.DEFAULT_PORT 
     host: 'localhost' 

    Esempi:
    > python client_example.py player_A --train
    > python client_example.py player_B --test 8080
    > python client_example.py player_C --train 9544 localhost
    '''

    main()
