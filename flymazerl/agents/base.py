import numpy as np
import os
import arviz as az
import gc


class FlYMazeAgent:
    """
    Base class for all agents.
    """

    def __init__(self, env, params=None, policy_params=None, history=True):
        """
        Constructor for the FlYMazeAgent class.
        =======================================

        Parameters:
        env: ymaze_static environment object
        params: dictionary of parameters for the agent
        policy_params: dictionary of parameters for the policy
        history: whether to record history
        
        Returns:
        FlYMazeAgent object
        """
        np.random.seed()
        self.env = env
        self.action_space_size = env.action_space.n
        self.biased_action = 0
        self.state_space_size = env.state_space.n
        self.episode_counter = 0
        self.history = history
        if history:
            self.init_recorder()

        if params is not None:
            self.init_variables(**params)
        else:
            self.init_variables()

        if policy_params is not None:
            self.init_policy_variables(**policy_params)
        else:
            self.init_policy_variables()
        self.bias = 0

    def get_bias_estimate(self):
        return self.bias / self.env.trial_no

    def init_recorder(self):
        """
        Initialize recorder for the agent.
        """
        pass

    def reset_recorder(self):
        """
        Reset the recorder.
        """
        pass

    def init_variables(self):
        """
        Initialize variables for the agent.
        """
        pass

    def init_policy_variables(self):
        """
        Initialize variables for the policy.
        """
        pass

    def run_episode(self):
        """
        Describe a single episode of the agent.
        """
        pass

    def reset_variables(self):
        """
        Reset variables for the agent.
        """
        pass

    def next_episode(self):
        """
        Run a single episode of the agent.
        """
        self.run_episode()
        self.episode_counter += 1

    def reset_agent(self):
        """
        Reset the agent.
        """
        self.episode_counter = 0
        self.bias = 0
        self.reset_variables()
        if self.history:
            self.reset_recorder()

    def load(self, filename, sample_from_population=False):
        """
        Load agent parameters from a file

        Parameters:
        filename: name of the file to load parameters from (str)
        sample_from_population: whether to sample from the population or just use the mean (bool)
        """
        assert os.path.isfile(filename), "File {} does not exist".format(filename)
        assert (
            filename.endswith(".nc") or filename.endswith(".npz") or filename.endswith(".pt")
        ), "File {} is not a NetCDF/Numpy/PyTorch file".format(filename)

        if filename.endswith(".nc"):
            inferenceData = az.from_netcdf(filename)
            if sample_from_population:
                space_size = inferenceData.posterior.dims["chain"] * inferenceData.posterior.dims["draw"]
                sample = np.random.choice(space_size)
                for i in list(inferenceData.posterior.data_vars):
                    exec(f"self.{i} = inferenceData.posterior.{i}.values.flatten()[{sample}]")
            else:
                for i in list(inferenceData.posterior.data_vars):
                    exec(f"self.{i} = np.mean(inferenceData.posterior.{i}.values.flatten())")
            del inferenceData
        elif filename.endswith(".npz"):
            data = np.load(filename)
            for i in list(data.keys()):
                exec(f"self.{i} = data['{i}']")
            del data
        else:
            exec(f'self.agent.load_state_dict(torch.load("{filename}"))')

        gc.collect()


class RandomAgent(FlYMazeAgent):
    """
    Random agent.
    """

    def init_policy_variables(self, bias_vector=None):
        """
        Initialize variables for the policy.
        ====================================

        Parameters:
        bias_vector: vector of probabilities for each action
        """
        if bias_vector is not None:
            self.bias_vector = bias_vector
        else:
            self.bias_vector = np.ones(self.action_space_size) / self.action_space_size

    def init_recorder(self):
        """
        Initialize recorder for the agent.
        """
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single random step.
        =============================

        Parameters:
        state: current state of the environment (int)

        Returns:
        next_state: next state of the environment (int)
        done: whether the episode is done (bool)
        """
        action = np.random.choice(self.action_space_size, p=self.bias_vector)

        new_state, reward, done, _ = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1 / self.env.n_trials_per_session  # update bias estimate

        return new_state, done

    def run_episode(self):
        """
        Describe a single episode of the agent.
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_recorder(self):
        """
        Reset the recorder.
        """
        self.reward_history = []
        self.action_history = []

