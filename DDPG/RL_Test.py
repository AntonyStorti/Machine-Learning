from DDPG import RL_Training
import os
import torch


def start_RL_Test():

    hidden_size = 64
    input_size = 3
    output_size = 2

    actor_model = RL_Training.ActorNetwork(input_size, hidden_size, output_size)
    critic_model = RL_Training.CriticNetwork(input_size, hidden_size, output_size)
    target_actor = RL_Training.ActorNetwork(input_size, hidden_size, output_size)
    target_critic = RL_Training.CriticNetwork(input_size, hidden_size, output_size)

    # Carico i modelli salvati, se esistono:
    if os.path.exists('DDPG/actor_model.pth') and os.path.exists('DDPG/critic_model.pth'):
        actor_model.load_state_dict(torch.load('DDPG/actor_model.pth'))
        critic_model.load_state_dict(torch.load('DDPG/critic_model.pth'))
        target_actor.load_state_dict(torch.load('DDPG/actor_model.pth'))
        target_critic.load_state_dict(torch.load('DDPG/critic_model.pth'))
        print("Modelli caricati dai file salvati.")
    else:
        raise Exception("Modelli non trovati !!!")

    return actor_model, critic_model


def main(state, cli, actor_model, critic_model):

    actor_model.eval()
    critic_model.eval()

    # Input che si aspetta la rete DDPG
    state_input = [state[17], state[18], state[19]]

    # Input convertito in tensore
    input_data = torch.tensor(state_input, dtype=torch.float32)

    # Passo gli input al modello per ottenere le predizioni
    with torch.no_grad():  # Disabilito il calcolo dei gradienti
        action = actor_model(input_data)

    # Invio dei giunti --> (Da 0 a 8): MPL | (Da 9 a 10): DDPG
    joints = [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], action[0], action[1]]

    cli.send_joints(joints)
