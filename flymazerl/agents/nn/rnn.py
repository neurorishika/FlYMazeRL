import numpy as np
import os
import torch
import torch.nn as nn
from flymazerl.agents.base import FlYMazeAgent
import time

"""
+++++++++++++++++++++++++++++++
Recurrent Neural Network Agents
+++++++++++++++++++++++++++++++
"""


class VanillaRNN(nn.Module):
    """
    A class to create a typical Recurrent Neural Network (RNN) with the option to use an intermediate Linear encoder.
    """

    def __init__(
        self,
        input_size,
        state_size,
        output_size,
        num_layers,
        use_intermediate_encoder=False,
        encoder_size=None,
        allow_negative_values=True,
        symmetric=False,
        device="cpu",
        hardness=0.8141,
    ):
        """
        Initialize the RNN
        ==================

        Parameters:
        input_size: The number of inputs to the RNN (int)
        state_size: The number of hidden units in the RNN (int)
        output_size: The number of outputs from the RNN (int)
        num_layers: The number of layers in the RNN (int)
        use_intermediate_encoder: Whether or not to use an intermediate encoder (bool)
        encoder_size: The number of units in the intermediate encoder (int)
        allow_negative_values: Whether or not to allow negative values in the output (bool)
        symmetric: Whether or not to use a symmetric RNN (bool)
        device: The device to use for the RNN (str)
        hardness: The balance between sigmoid and hardsigmoid (float)
        """
        super(VanillaRNN, self).__init__()
        self.state_size = state_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        if use_intermediate_encoder:
            self.encoder = nn.Linear(input_size, encoder_size)
            self.encoder_size = encoder_size
            self.rnn = nn.RNN(encoder_size, state_size, num_layers, batch_first=True)
        else:
            self.encoder = None
            self.encoder_size = None
            self.rnn = nn.RNN(input_size, state_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(state_size, output_size)
        self.device = device
        self.allow_negative_values = allow_negative_values
        self.symmetric = symmetric
        self.hardness = hardness

        # perform xavier initialization
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, inputs, hidden_state):
        """
        Forward pass through the RNN
        =============================

        Parameters:
        inputs: The inputs to the RNN (torch.Tensor)
        hidden_state: The hidden state of the RNN (torch.Tensor)

        Returns:
        output: The output of the RNN (torch.Tensor)
        hidden_state: The new hidden state of the RNN (torch.Tensor)
        """
        x = inputs
        x_ = inputs * torch.tensor([[-1, 1]]).to(self.device)

        if self.encoder is not None:
            output, _ = self.rnn(self.encoder(x), hidden_state)
            output_, _ = self.rnn(self.encoder(x_), hidden_state)
        else:
            output, _ = self.rnn(x, hidden_state)
            output_, _ = self.rnn(x_, hidden_state)

        output = self.decoder(output)
        output_ = self.decoder(output_)

        if self.symmetric:
            output = (output + torch.flip(output_, dims=[2])) / 2
        else:
            output = output

        if self.allow_negative_values:
            output = output
        else:
            output = (1 - self.hardness) * torch.nn.Sigmoid()(output) + self.hardness * torch.nn.Hardsigmoid()(output)

        return output

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state of the RNN
        ======================================

        Parameters:
        batch_size: The batch size of the RNN (int)

        Returns:
        hidden_state: The hidden state of the RNN (torch.Tensor)
        """
        return torch.zeros(self.num_layers, batch_size, self.state_size).to(self.device)


class LSTMNN(nn.Module):
    """
    A class to create a Long Short-Term Memory (LSTM) RNN with the option to use an intermediate Linear encoder.
    """

    def __init__(
        self,
        input_size,
        state_size,
        output_size,
        num_layers,
        use_intermediate_encoder=False,
        encoder_size=None,
        allow_negative_values=True,
        symmetric=False,
        device="cpu",
        hardness=0.8141,
    ):
        """
        Initialize the LSTM
        ===================

        Parameters:
        input_size: The number of inputs to the LSTM (int)
        state_size: The number of hidden units in the LSTM (int)
        output_size: The number of outputs from the LSTM (int)
        num_layers: The number of layers in the LSTM (int)
        use_intermediate_encoder: Whether or not to use an intermediate encoder (bool)
        encoder_size: The number of units in the intermediate encoder (int)
        allow_negative_values: Whether or not to allow negative values in the output (bool)\
        symmetric: Whether or not to use a symmetric LSTM (bool)
        device: The device to use for the LSTM (str)
        hardness: The balance between sigmoid and hardsigmoid (float)
        """
        super(LSTMNN, self).__init__()
        self.state_size = state_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        if use_intermediate_encoder:
            self.encoder = nn.Linear(input_size, encoder_size)
            self.encoder_size = encoder_size
            self.rnn = nn.LSTM(encoder_size, state_size, num_layers, batch_first=True)
        else:
            self.encoder = None
            self.encoder_size = None
            self.rnn = nn.LSTM(input_size, state_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(state_size, output_size)
        self.device = device
        self.allow_negative_values = allow_negative_values
        self.symmetric = symmetric
        self.hardness = hardness

        # perform xavier initialization
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, inputs, hidden_state):
        """
        Forward pass through the LSTM
        =============================

        Parameters:
        inputs: The inputs to the LSTM (torch.Tensor)
        hidden_state: The hidden state of the LSTM (torch.Tensor)

        Returns:
        output: The output of the LSTM (torch.Tensor)
        hidden_state: The new hidden state of the LSTM (torch.Tensor)
        """
        x = inputs
        x_ = inputs * torch.tensor([[-1, 1]]).to(self.device)

        if self.encoder is not None:
            output, _ = self.rnn(self.encoder(x), hidden_state)
            output_, _ = self.rnn(self.encoder(x_), hidden_state)
        else:
            output, _ = self.rnn(x, hidden_state)
            output_, _ = self.rnn(x_, hidden_state)

        output = self.decoder(output)
        output_ = self.decoder(output_)

        if self.symmetric:
            output = (output + torch.flip(output_, dims=[2])) / 2
        else:
            output = output

        if self.allow_negative_values:
            output = output
        else:
            output = (1 - self.hardness) * torch.nn.Sigmoid()(output) + self.hardness * torch.nn.Hardsigmoid()(output)

        return output

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state of the LSTM
        =======================================

        Parameters:
        batch_size: The batch size of the LSTM (int)

        Returns:
        hidden_state: The hidden state of the LSTM (torch.Tensor)
        """
        return (
            torch.zeros(self.num_layers, batch_size, self.state_size).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.state_size).to(self.device),
        )


class GRUNN(nn.Module):
    """
    A class to create a Gated Recurrent Unit (GRU) RNN with the option to use an intermediate Linear encoder.
    """

    def __init__(
        self,
        input_size,
        state_size,
        output_size,
        num_layers,
        use_intermediate_encoder=False,
        encoder_size=None,
        allow_negative_values=True,
        symmetric=False,
        device="cpu",
        hardness=0.8141,
    ):
        """
        Initialize the GRU
        ==================

        Parameters:
        input_size: The number of inputs to the GRU (int)
        state_size: The number of hidden units in the GRU (int)
        output_size: The number of outputs from the GRU (int)
        num_layers: The number of layers in the GRU (int)
        use_intermediate_encoder: Whether or not to use an intermediate encoder (bool)
        encoder_size: The number of units in the intermediate encoder (int)
        allow_negative_values: Whether or not to allow negative values in the output (bool)
        symmetric: Whether or not to use a symmetric GRU (bool)
        device: The device to use for the GRU (str)
        hardness: The balance between sigmoid and hardsigmoid (float)
        """
        super(GRUNN, self).__init__()
        self.state_size = state_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        if use_intermediate_encoder:
            self.encoder = nn.Linear(input_size, encoder_size)
            self.encoder_size = encoder_size
            self.rnn = nn.GRU(encoder_size, state_size, num_layers, batch_first=True)
        else:
            self.encoder = None
            self.encoder_size = None
            self.rnn = nn.GRU(input_size, state_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(state_size, output_size)
        self.device = device
        self.allow_negative_values = allow_negative_values
        self.symmetric = symmetric
        self.hardness = hardness

        # perform xavier initialization
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, inputs, hidden_state):
        """
        Forward pass through the GRU
        ============================

        Parameters:
        inputs: The inputs to the GRU (torch.Tensor)
        hidden_state: The hidden state of the GRU (torch.Tensor)

        Returns:
        output: The output of the GRU (torch.Tensor)
        hidden_state: The new hidden state of the GRU (torch.Tensor)
        """
        x = inputs
        x_ = inputs * torch.tensor([[-1, 1]]).to(self.device)

        if self.encoder is not None:
            output, _ = self.rnn(self.encoder(x), hidden_state)
            output_, _ = self.rnn(self.encoder(x_), hidden_state)
        else:
            output, _ = self.rnn(x, hidden_state)
            output_, _ = self.rnn(x_, hidden_state)

        output = self.decoder(output)
        output_ = self.decoder(output_)

        if self.symmetric:
            output = (output + torch.flip(output_, dims=[2])) / 2
        else:
            output = output

        if self.allow_negative_values:
            output = output
        else:
            output = (1 - self.hardness) * torch.nn.Sigmoid()(output) + self.hardness * torch.nn.Hardsigmoid()(output)

        return output

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state of the GRU
        ======================================

        Parameters:
        batch_size: The batch size of the GRU (int)

        Returns:
        hidden_state: The hidden state of the GRU (torch.Tensor)
        """
        return torch.zeros(self.num_layers, batch_size, self.state_size).to(self.device)


class GRNNLearner(FlYMazeAgent):
    def init_variables(
        self,
        reservoir_size,
        num_layers,
        symmetric_q_function=True,
        allow_negative_values=True,
        omission_is_punishment=False,
        encoder_size=None,
        kind="RNN",
        policy_type="softmax",
        device="cpu",
        pre_trained=False,
        model_path=None,
        multi_agent=False,
        n_agents=1,
        hardness=0.8141,
    ):
        """
        Initialize the variables of a Generalized RNN Learner
        =====================================================

        Parameters:
        reservoir_size: The number of units in the RNN hidden layer (int)
        num_layers: The number of layers in the RNN (int)
        symmetric_q_function: Whether or not to use a symmetric decision network (bool)
        allow_negative_values: Whether or not to allow negative values in the RNN (bool)
        omission_is_punishment: Whether or not to punish omission (bool)
        encoder_size: The number of units in the intermediate encoder (int)
        kind: The type of RNN to use (str) (options: "RNN", "LSTM", "GRU")
        policy_type: The type of policy to use (str) (options: "softmax", "greedy")
        device: The device to use for the RNN (str) (options: "cpu", "cuda")
        pre_trained: Whether or not to load a pre-trained model (bool)
        model_path: The path to the pre-trained model (str)
        multi_agent: Whether or not to simulate multiple agents (bool)
        n_agents: The number of agents to simulate (int)
        hardness: The balance between sigmoid and hardsigmoid (float)
        """
        self.reservoir_size = reservoir_size
        self.num_layers = num_layers
        self.encoder_size = encoder_size
        self.device = device
        self.symmetric_q_function = symmetric_q_function
        self.allow_negative_values = allow_negative_values
        self.omission_is_punishment = omission_is_punishment
        self.kind = kind
        self.hardness = hardness

        if pre_trained:
            assert (
                model_path is not None
            ), "If you want to use a pre-trained model, you need to specify the path to the model."

        if kind == "RNN":
            self.agent = VanillaRNN(
                2,
                self.reservoir_size,
                self.action_space_size,
                self.num_layers,
                use_intermediate_encoder=self.encoder_size is not None,
                encoder_size=self.encoder_size,
                allow_negative_values=self.allow_negative_values,
                symmetric=self.symmetric_q_function,
                device=self.device,
                hardness=self.hardness,
            ).to(self.device)
        elif kind == "LSTM":
            self.agent = LSTMNN(
                2,
                self.reservoir_size,
                self.action_space_size,
                self.num_layers,
                use_intermediate_encoder=self.encoder_size is not None,
                encoder_size=self.encoder_size,
                allow_negative_values=self.allow_negative_values,
                symmetric=self.symmetric_q_function,
                device=self.device,
                hardness=self.hardness,
            ).to(self.device)
        elif kind == "GRU":
            self.agent = GRUNN(
                2,
                self.reservoir_size,
                self.action_space_size,
                self.num_layers,
                use_intermediate_encoder=self.encoder_size is not None,
                encoder_size=self.encoder_size,
                allow_negative_values=self.allow_negative_values,
                symmetric=self.symmetric_q_function,
                device=self.device,
                hardness=self.hardness,
            ).to(self.device)
        else:
            raise ValueError("Unknown RNN kind: {}".format(kind))

        if pre_trained:
            self.agent.load_state_dict(torch.load(model_path))

        self.policy_type = policy_type

        self.multi_agent = multi_agent
        self.n_agents = n_agents
        if multi_agent:
            self.history = torch.zeros((n_agents, 1, 2)).to(self.device)
            self.bias = np.zeros(n_agents)
        else:
            self.history = torch.zeros((1, 1, 2)).to(self.device)

    def trial_step(self, state):
        """
        Take a step in the environment given the current state
        ======================================================

        Parameters:
        state: The current state of the environment (np.array)

        Returns:
        action: The action to take in the environment (int)
        """
        hidden = self.agent.init_hidden(self.n_agents).to(self.device)
        action_logits = self.agent(self.history.float(), hidden)

        if self.multi_agent:
            if self.policy_type == "softmax":
                action_probabilities = action_logits.softmax(dim=2)[:, -1, :]
                action = torch.multinomial(action_probabilities, 1)
            elif self.policy_type == "greedy":
                action_probabilities = action_logits[:, -1, :]
                action = action_probabilities.argmax(dim=1)
            elif self.policy_type == "acceptreject":
                action_probabilities = torch.exp(
                    torch.stack(
                        [
                            torch.log(action_logits[:, :, 0])
                            + torch.log(3 - action_logits[:, :, 1])
                            - torch.log(
                                3 * action_logits[:, :, 1]
                                + 3 * action_logits[:, :, 0]
                                - 2 * action_logits[:, :, 0] * action_logits[:, :, 1]
                            ),
                            torch.log(action_logits[:, :, 1])
                            + torch.log(3 - action_logits[:, :, 0])
                            - torch.log(
                                3 * action_logits[:, :, 1]
                                + 3 * action_logits[:, :, 0]
                                - 2 * action_logits[:, :, 0] * action_logits[:, :, 1]
                            ),
                        ],
                        dim=2,
                    )
                )[:, -1, :]
                action = torch.multinomial(action_probabilities, 1)
            else:
                raise ValueError("Unknown policy type: {}".format(self.policy_type))

            new_state, reward, done, _ = self.env.step(action)

            reward = torch.tensor(reward)

            if self.omission_is_punishment:
                self.history = torch.cat(
                    [
                        self.history,
                        torch.concat([action.view(-1, 1) * 2 - 1, reward.view(-1, 1) * 2 - 1], axis=1).unsqueeze(1),
                    ],
                    axis=1,
                )
                self.action_history = (self.history.cpu().detach().numpy()[:, 1:, 0] + 1) / 2
                self.reward_history = (self.history.cpu().detach().numpy()[:, 1:, 1] + 1) / 2
            else:
                self.history = torch.cat(
                    [
                        self.history,
                        torch.concat([action.view(-1, 1) * 2 - 1, reward.view(-1, 1)], axis=1).unsqueeze(1),
                    ],
                    axis=1,
                )
                self.action_history = (self.history.cpu().detach().numpy()[:, 1:, 0] + 1) / 2
                self.reward_history = self.history.cpu().detach().numpy()[:, 1:, 1]

            self.bias += (action == self.biased_action).cpu().detach().numpy()  # update bias estimate

        else:
            if self.policy_type == "softmax":
                action_probabilities = action_logits.softmax(dim=2).squeeze(0)[-1]
                action = torch.multinomial(action_probabilities, 1).item()
            elif self.policy_type == "greedy":
                action_probabilities = action_logits.squeeze(0)[-1]
                action = action_probabilities.argmax().item()
            elif self.policy_type == "acceptreject":
                action_probabilities = torch.exp(
                    torch.stack(
                        [
                            torch.log(action_logits[:, :, 0])
                            + torch.log(3 - action_logits[:, :, 1])
                            - torch.log(
                                3 * action_logits[:, :, 1]
                                + 3 * action_logits[:, :, 0]
                                - 2 * action_logits[:, :, 0] * action_logits[:, :, 1]
                            ),
                            torch.log(action_logits[:, :, 1])
                            + torch.log(3 - action_logits[:, :, 0])
                            - torch.log(
                                3 * action_logits[:, :, 1]
                                + 3 * action_logits[:, :, 0]
                                - 2 * action_logits[:, :, 0] * action_logits[:, :, 1]
                            ),
                        ],
                        dim=2,
                    )
                ).squeeze(0)[-1]
                action = torch.multinomial(action_probabilities, 1).item()
            else:
                raise ValueError("Unknown policy type: {}".format(self.policy_type))

            new_state, reward, done, _ = self.env.step(action)

            if self.omission_is_punishment:
                self.history = torch.cat(
                    (self.history, torch.tensor([[[action * 2 - 1, reward * 2 - 1]]]).to(self.device)), dim=1
                )
                self.action_history = (self.history.squeeze(0).cpu().detach().numpy()[1:, 0] + 1) / 2
                self.reward_history = (self.history.squeeze(0).cpu().detach().numpy()[1:, 1] + 1) / 2
            else:
                self.history = torch.cat(
                    (self.history, torch.tensor([[[action * 2 - 1, reward]]]).to(self.device)), dim=1
                )
                self.action_history = (self.history.squeeze(0).cpu().detach().numpy()[1:, 0] + 1) / 2
                self.reward_history = self.history.squeeze(0).cpu().detach().numpy()[1:, 1]

            if action == self.biased_action:
                self.bias += 1  # update bias estimate

        return new_state, done

    def get_q_history(self):
        """
        Get the history of the values
        =================================

        Returns:
        q_history: The history of the values (np.array)
        """
        if self.multi_agent:
            return self.agent.forward(self.history.float())[:, 1:-1, :].cpu().detach().numpy()
        else:
            return self.agent.forward(self.history.float())[:, 1:-1, :].squeeze(0).cpu().detach().numpy()

    def run_episode(self):
        """
        Run an episode of the environment
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset the variables of the agent
        """
        self.history = torch.zeros((1, 1, 2)).to(self.device)

    def get_action_probabilities_from_data(self, actions_set, rewards_set):
        """
        Given a dataset, infer the action probabilities
        ===============================================

        actions_set: The set of actions taken in the dataset (np.array)
        rewards_set: The set of rewards obtained in the dataset (np.array)

        Returns:
        action_probabilities: The inferred action probabilities (np.array)
        """

        dataset = torch.tensor(np.array([actions_set, rewards_set]).transpose((1, 2, 0)), dtype=torch.int32).to(
            self.device
        )
        X = torch.clone(dataset[:, :-1, :])
        if self.omission_is_punishment:
            X = X * 2 - 1
        else:
            X[:, :, 0] = X[:, :, 0] * 2 - 1

        hidden = self.agent.init_hidden(X.shape[0]).to(self.device)
        logits = self.agent.forward(X.float(), hidden)
        if self.policy_type == "softmax":
            action_probabilities = logits.softmax(dim=2).cpu().detach().numpy()
        elif self.policy_type == "greedy":
            action_probabilities = logits.cpu().detach().numpy()
        elif self.policy_type == "acceptreject":
            action_probabilities = (
                torch.exp(
                    torch.stack(
                        [
                            torch.log(logits[:, :, 0])
                            + torch.log(3 - logits[:, :, 1])
                            - torch.log(
                                3 * logits[:, :, 1] + 3 * logits[:, :, 0] - 2 * logits[:, :, 0] * logits[:, :, 1]
                            ),
                            torch.log(logits[:, :, 1])
                            + torch.log(3 - logits[:, :, 0])
                            - torch.log(
                                3 * logits[:, :, 1] + 3 * logits[:, :, 0] - 2 * logits[:, :, 0] * logits[:, :, 1]
                            ),
                        ],
                        dim=2,
                    )
                )
                .cpu()
                .detach()
                .numpy()
            )
        else:
            raise ValueError("Unknown policy type: {}".format(self.policy_type))
        return action_probabilities

    def get_q_values_from_data(self, actions_set, rewards_set):
        """
        Given a dataset, infer the action probabilities
        ===============================================

        actions_set: The set of actions taken in the dataset (np.array)
        rewards_set: The set of rewards obtained in the dataset (np.array)

        Returns:
        action_probabilities: The inferred action probabilities (np.array)
        """

        dataset = torch.tensor(np.array([actions_set, rewards_set]).transpose((1, 2, 0)), dtype=torch.int32).to(
            self.device
        )
        X = torch.clone(dataset[:, :-1, :])
        if self.omission_is_punishment:
            X = X * 2 - 1
        else:
            X[:, :, 0] = X[:, :, 0] * 2 - 1

        hidden = self.agent.init_hidden(X.shape[0]).to(self.device)
        logits = self.agent.forward(X.float(), hidden)
        return logits[:, :, :].cpu().detach().numpy()

    def load_pre_trained_model(self, model_path):
        """
        Load a pre-trained model
        """
        assert os.path.exists(model_path), "The model path does not exist."
        self.agent.load_state_dict(torch.load(model_path))

    def fit(
        self,
        actions_set,
        rewards_set,
        train_test_split=0.8,
        n_replications=1,
        early_stopping=True,
        early_stopping_patience=50,
        max_epochs=10000,
        learning_rate=0.0005,
        print_every=500,
        weight_decay=1e-5,
        filter_best=False,
        uid=None,
        tolerance=1e-4,
        scheduler=False,
        minibatch_size=1,
        minibatch_seed=15403997,
    ):
        """
        Fit the agent to the data using early stopping
        ==============================================

        Parameters:
        actions_set: The set of actions taken in the environment (np.array)
        rewards_set: The set of rewards received in the environment (np.array)
        train_test_split: The percentage of the data to use for training (float)
        n_replications: The number of times to replicate the training (int)
        early_stopping: Whether or not to use early stopping (bool)
        early_stopping_patience: The number of epochs to wait before stopping (int)
        max_epochs: The maximum number of epochs to train for (int)
        learning_rate: The learning rate to use for training (float)
        print_every: The number of epochs to wait before printing the loss (int)
        weight_decay: The weight decay to use for training (float)
        filter_best: Whether or not to filter the best model (bool)
        uid: The unique identifier of the model (str)
        tolerance: The tolerance for the loss (float)
        scheduler: Whether or not to use a scheduler (bool)
        minibatch_size: The size of the minibatch (int)
        minibatch_seed: The seed to use for the minibatch (int)
        """

        dataset = torch.tensor(np.array([actions_set, rewards_set]).transpose((1, 2, 0)), dtype=torch.int32).to(
            self.device
        )
        X = torch.clone(dataset[:, :-1, :])
        y = torch.clone(dataset[:, 1:, 0])
        if self.omission_is_punishment:
            X = X * 2 - 1
        else:
            X[:, :, 0] = X[:, :, 0] * 2 - 1

        fitting_stats = []

        for i in range(n_replications):

            start_time = time.time()

            for layer in self.agent.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

            optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate, weight_decay=weight_decay)
            if scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5, verbose=True
                )
            loss_fn = nn.CrossEntropyLoss()

            train_indices = np.random.choice(len(X), int(len(X) * train_test_split), replace=False)
            val_indices = np.array(list(set(range(len(X))) - set(train_indices)))
            X_train, X_val = X[train_indices].to(self.device), X[val_indices].to(self.device)
            y_train, y_val = y[train_indices].to(self.device), y[val_indices].to(self.device)

            training_loss = []
            validation_loss = []

            patience = early_stopping_patience
            best_val_loss = float("inf")
            for epoch in range(max_epochs):
                # Shuffle the data
                torch.manual_seed(minibatch_seed + epoch)
                permutation = torch.randperm(X_train.shape[0])
                X_train = X_train[permutation]
                y_train = y_train[permutation]
                # Split the data into minibatches
                minibatches = torch.split(X_train, minibatch_size, dim=0)
                minibatches_y = torch.split(y_train, minibatch_size, dim=0)
                # Train the model
                self.agent.train()
                for minibatch, minibatch_y in zip(minibatches, minibatches_y):
                    optimizer.zero_grad()
                    hidden = self.agent.init_hidden(minibatch.shape[0]).to(self.device)
                    output = self.agent(minibatch.float(), hidden)
                    if self.policy_type == "softmax":
                        output = output.softmax(dim=2).view(-1, self.action_space_size)
                    elif self.policy_type == "greedy":
                        output = output.argmax(dim=2).view(-1)
                    elif self.policy_type == "acceptreject":
                        output = torch.exp(
                            torch.stack(
                                [
                                    torch.log(output[:, :, 0])
                                    + torch.log(3 - output[:, :, 1])
                                    - torch.log(
                                        3 * output[:, :, 1]
                                        + 3 * output[:, :, 0]
                                        - 2 * output[:, :, 0] * output[:, :, 1]
                                    ),
                                    torch.log(output[:, :, 1])
                                    + torch.log(3 - output[:, :, 0])
                                    - torch.log(
                                        3 * output[:, :, 1]
                                        + 3 * output[:, :, 0]
                                        - 2 * output[:, :, 0] * output[:, :, 1]
                                    ),
                                ],
                                dim=2,
                            )
                        ).view(-1, self.action_space_size)
                    else:
                        raise ValueError("Unknown policy type.")
                    loss = loss_fn(output, minibatch_y.view(-1).long())
                    training_loss.append(loss.item())

                    loss.backward()
                    optimizer.step()

                self.agent.eval()
                with torch.no_grad():
                    hidden = self.agent.init_hidden(X_val.shape[0]).to(self.device)
                    output = self.agent(X_val.float(), hidden)
                    if self.policy_type == "softmax":
                        output = output.softmax(dim=2).view(-1, self.action_space_size)
                    elif self.policy_type == "greedy":
                        output = output.argmax(dim=2).view(-1)
                    elif self.policy_type == "acceptreject":
                        output = torch.exp(
                            torch.stack(
                                [
                                    torch.log(output[:, :, 0])
                                    + torch.log(3 - output[:, :, 1])
                                    - torch.log(
                                        3 * output[:, :, 1]
                                        + 3 * output[:, :, 0]
                                        - 2 * output[:, :, 0] * output[:, :, 1]
                                    ),
                                    torch.log(output[:, :, 1])
                                    + torch.log(3 - output[:, :, 0])
                                    - torch.log(
                                        3 * output[:, :, 1]
                                        + 3 * output[:, :, 0]
                                        - 2 * output[:, :, 0] * output[:, :, 1]
                                    ),
                                ],
                                dim=2,
                            )
                        ).view(-1, self.action_space_size)
                    else:
                        raise ValueError("Unknown policy type.")
                    val_loss = loss_fn(output, y_val.view(-1).long())
                    validation_loss.append(val_loss.item())

                if scheduler:
                    scheduler.step(val_loss)

                if epoch % print_every == 0:
                    # get number of minibatches
                    n_minibatches = len(minibatches)
                    avg_train_loss = np.mean(training_loss[-n_minibatches:])
                    print(
                        "Epoch {}: \tTraining Loss: {:.4f}\tValidation Loss: {:.4f}".format(
                            epoch, avg_train_loss, val_loss
                        )
                    )
                if early_stopping:
                    if val_loss < best_val_loss - tolerance:
                        best_val_loss = val_loss
                        patience = early_stopping_patience
                        torch.save(
                            self.agent.state_dict(),
                            "model_{}.pt".format(i) if uid is None else "model_{}_{}.pt".format(uid, i),
                        )
                    else:
                        patience -= 1
                    if patience == 0:
                        print("Early stopping at epoch {}".format(epoch))
                        print("Best validation loss: {:.4f}".format(best_val_loss))
                        break

            fitting_stats.append(
                {
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "best_val_epoch": epoch - early_stopping_patience,
                    "training_time": time.time() - start_time,
                }
            )

        if filter_best:
            print("Finding best model replicate...")

            best_model_loss = float("inf")
            for i in range(n_replications):
                for layer in self.agent.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
                loss_fn = nn.CrossEntropyLoss()

                self.agent.load_state_dict(
                    torch.load("model_{}.pt".format(i) if uid is None else "model_{}_{}.pt".format(uid, i))
                )
                self.agent.eval()
                with torch.no_grad():
                    hidden = self.agent.init_hidden(X.shape[0]).to(self.device)
                    output = self.agent(X.float(), hidden)
                    if self.policy_type == "softmax":
                        output = output.softmax(dim=2).view(-1, self.action_space_size)
                    elif self.policy_type == "greedy":
                        output = output.argmax(dim=2).view(-1)
                    elif self.policy_type == "acceptreject":
                        output = torch.exp(
                            torch.stack(
                                [
                                    torch.log(output[:, :, 0])
                                    + torch.log(3 - output[:, :, 1])
                                    - torch.log(
                                        3 * output[:, :, 1]
                                        + 3 * output[:, :, 0]
                                        - 2 * output[:, :, 0] * output[:, :, 1]
                                    ),
                                    torch.log(output[:, :, 1])
                                    + torch.log(3 - output[:, :, 0])
                                    - torch.log(
                                        3 * output[:, :, 1]
                                        + 3 * output[:, :, 0]
                                        - 2 * output[:, :, 0] * output[:, :, 1]
                                    ),
                                ],
                                dim=2,
                            )
                        ).view(-1, self.action_space_size)
                    else:
                        raise ValueError("Unknown policy type.")
                    val_loss = loss_fn(output, y.view(-1).long())
                if val_loss < best_model_loss:
                    best_model_loss = val_loss
                    torch.save(
                        self.agent.state_dict(), "best_model.pt" if uid is None else "best_model_{}.pt".format(uid)
                    )
                os.remove("model_{}.pt".format(i) if uid is None else "model_{}_{}.pt".format(uid, i))
            print("Best model found!")

        return fitting_stats
