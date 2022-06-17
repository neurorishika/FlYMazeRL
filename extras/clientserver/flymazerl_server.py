import socket
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = get_device()


# Create a class for the RNNAgent
class RNNAgent(nn.Module):
    def __init__(self, input_size, state_size, action_space_size, num_layers):
        super(RNNAgent, self).__init__()
        self.state_size = state_size
        self.input_size = input_size
        self.action_space_size = action_space_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, state_size, num_layers, batch_first=True)
        self.action_decoder = nn.Linear(state_size, action_space_size)

    def forward(self, inputs, hidden_state):
        output, hidden_state = self.rnn(inputs, hidden_state)
        output = self.action_decoder(output)
        return output, hidden_state

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.state_size, dtype=torch.float32, device=device)


# Create a class for the DRQN Experimenter Agent
class DRQNExperimenterAgent(nn.Module):
    def __init__(self, input_size, state_size, reward_space_size, num_layers):
        super(DRQNExperimenterAgent, self).__init__()
        self.state_size = state_size
        self.input_size = input_size
        self.reward_space_size = reward_space_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, state_size, num_layers, batch_first=True)
        self.q_decoder = nn.Linear(state_size, reward_space_size)

    def forward(self, inputs, hidden_state):
        output, hidden_state = self.rnn(inputs, hidden_state)
        output = self.q_decoder(output)
        return output, hidden_state

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.state_size).to(device)


HOST = "localhost"
PORT = 28567

verbose = False

if __name__ == "__main__":

    # looptype = input("Enter loop type (open/closed): ")

    # if looptype == "close":

    #     # Initialize the RNN Agent
    #     learning_model = RNNAgent(input_size=2, state_size=3, action_space_size=2, num_layers=1).to(device)

    #     # Load the learning model
    #     # learning_model.load_state_dict(torch.load("learning_model.pt"))

    #     print("Running closed loop. Learning model loaded.")

    # elif looptype == "open":
    #     print("Running open loop. learning model not loaded.")

    # else:
    #     print("Invalid loop type.")

    experimenter_model = DRQNExperimenterAgent(input_size=2, state_size=2048, reward_space_size=4, num_layers=1).to(
        device
    )

    # Load the experimenter model
    # experimenter_model.load_state_dict(torch.load("experimenter_model.pt"))

    print("Experimenter model loaded.")

    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind to the port
    s.bind((HOST, PORT))
    # Listen for 1 connections
    s.listen(1)
    # Accept the connection
    conn, addr = s.accept()
    print("Connected by", addr)
    # Receive the data in small chunks and retransmit it
    while True:
        data = conn.recv(1024)
        if not data:
            print("Connection closed. Exiting.")
            break
        # print(data)
        # conn.sendall(data)
        if data == b"new session":
            if verbose:
                print("New session Initiated.")

            # Reset History
            history = torch.tensor(np.array([[[0, 0]]], dtype=np.float32)).to(device)  # past_action, past_reward

            # Initialize the DRQN Experimenter Hidden State
            experimenter_hidden_state = experimenter_model.init_hidden(1).to(device)
            reward_logits, _ = experimenter_model(history.float(), experimenter_hidden_state)
            reward_probability = reward_logits.softmax(dim=2).squeeze(0)[-1]
            reward = torch.multinomial(reward_probability, 1).cpu().detach().numpy()[0]
            conn.sendall(str(reward).encode())
            if verbose:
                print("Reward sent.")

        elif data.decode("utf-8").split(":")[0] == "a,r":
            action = int(data.decode("utf-8").split(":")[1].split(",")[0])
            reward = int(data.decode("utf-8").split(":")[1].split(",")[1])
            if verbose:
                print("Received (action, reward) pair: (", action, ",", reward, ")")

            if "history" not in locals():
                print("Error: Session not initialized!")
                continue

            # Update the history
            history = torch.cat(
                (history, torch.tensor(np.array([[[action, reward]]], dtype=np.float32)).to(device)), dim=1
            ).to(device)

            if verbose:
                print("History size:", history.shape[1], "trials")

            # Update the hidden state
            experimenter_hidden_state = experimenter_model.init_hidden(1).to(device)

            # Get the reward logits
            reward_logits, _ = experimenter_model(history.float(), experimenter_hidden_state)
            reward_probability = reward_logits.softmax(dim=2).squeeze(0)[-1]
            reward = torch.multinomial(reward_probability, 1).cpu().detach().numpy()[0]
            conn.sendall(str(reward).encode())
            if verbose:
                print("Reward Code", reward, "sent.")

        else:
            print("Invalid data received.")
