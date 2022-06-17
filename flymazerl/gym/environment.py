import gym
from flymazerl.utils.generators import generate_random_schedule
import numpy as np
import torch


class ymaze_static(gym.Env):
    """
    A class for creating an OpenAI gym environment for the yMaze task. Can be generalized for any k-armed bandit task.
    """

    def __init__(self, n_trials_per_session, reward_fraction=None, random_seed=None, schedule=None):
        """
        A constructor for a static Y-MAZE environment
        =============================================
        Parameters:
        n_trials_per_session: number of trials per episode (int)
        reward_fraction: fraction of rewarded trials in each alternative (float)
        random_seed: random seed (int) (default: None)
        schedule: schedule to be used (np.array) (shape: n_trials_per_session x 2) (default: None)
        """
        self.n_trials_per_session = n_trials_per_session
        self.reward_fraction = reward_fraction
        self.random_seed = random_seed
        self.action_space = gym.spaces.Discrete(2)  # Left or Right
        self.state_space = gym.spaces.Discrete(2)  # Active or Complete
        if schedule is None and reward_fraction is not None:  # generate random reward schedule
            if random_seed is None:
                np.random.seed()
            else:
                np.random.seed(random_seed)
            self.schedule = generate_random_schedule(n_trials_per_session, reward_fraction)
        else:  # use predefined reward schedule
            self.schedule = schedule
        self.trial_no = 0

    def step(self, action):
        """
        Perform a single trial in the Y-Maze task
        =========================================
        Parameters:
        action: action to be performed (int)

        Returns:
        state: state of the environment (int)
        reward: reward received (float)
        done: whether the episode is finished (bool)
        info: auxiliary information (dict)
        """
        self.trial_no += 1  # iterate over trials
        if self.trial_no >= self.n_trials_per_session:  # end episode
            state = 1
            done = True
        else:
            state = 0
            done = False
        reward = self.schedule[self.trial_no - 1, action]
        info = {}
        return state, reward, done, info

    def reset(self):
        """
        Reset the environment to the initial state
        ==========================================
        Returns:
        state: initial state (int)
        """
        state = 0
        self.trial_no = 0
        return state


class ymaze_dynamic(gym.Env):
    """
    A class for creating an OpenAI gym environment for a dynamic yMaze task.
    """

    def __init__(self, n_trials_per_session):
        """
        A constructor for a Dynamic Y-maze environment
        ==============================================
        Parameters:
        n_trials_per_session: number of trials per episode (int)
        reward_fraction: fraction of rewarded trials in each alternative (float)
        random_seed: random seed (int) (default: None)
        """
        self.n_trials_per_session = n_trials_per_session
        self.action_space = gym.spaces.Discrete(2)  # Left or Right
        self.state_space = gym.spaces.Discrete(2)  # Active or Complete
        self.trial_no = 0

    def step(self, action):
        """
        Perform a single trial in the Y-Maze task
        =========================================
        Parameters:
        action: action to be performed (int)

        Returns:
        state: state of the environment (int)
        reward: reward received (float)
        done: whether the episode is finished (bool)
        info: auxiliary information (dict)
        """
        self.trial_no += 1  # iterate over trials
        if self.trial_no >= self.n_trials_per_session:  # end episode
            state = 1
            done = True
        else:
            state = 0
            done = False
        reward = self.generator(action)
        info = {}
        return state, reward, done, info

    def generator(self, action):
        """
        Generates no reward
        ===================

        Parameters:
        action: action to be performed (int)
        state_vector: state vector (np.array)
        """
        return 0

    def reset(self):
        """
        Reset the environment to the initial state
        ==========================================
        Returns:
        state: initial state (int)
        """
        state = 0
        self.trial_no = 0
        return state

class ymaze_fixedreward(ymaze_dynamic):
    """
    A class for creating an OpenAI gym environment for the yMaze task.
    """

    def __init__(self, n_trials_per_session, reward_fraction=None, random_seed=None):
        """
        A constructor for a Dynamic Y-maze environment
        ==============================================
        Parameters:
        n_trials_per_session: number of trials per episode (int)
        reward_fraction: fraction of rewarded trials in each alternative (float)
        random_seed: random seed (int) (default: None)
        """
        super().__init__(n_trials_per_session)
        self.reward_fraction = reward_fraction
        self.random_seed = random_seed
        self.rewards_used = [0, 0]
        if self.reward_fraction is not None:
            self.n_rewarded_trials = [
                int(self.n_trials_per_session * self.reward_fraction),
                int(self.n_trials_per_session * self.reward_fraction),
            ]
        else:
            self.n_rewarded_trials = [np.inf, np.inf]

    def generator(self, action):
        """
        Generates a random reward
        =========================

        Parameters:
        action: action to be performed (int)
        state_vector: state vector (np.array)
        """

        p_0 = (self.n_rewarded_trials[0] - self.rewards_used[0]) / (self.n_trials_per_session - self.trial_no - 1)
        p_1 = (self.n_rewarded_trials[1] - self.rewards_used[1]) / (self.n_trials_per_session - self.trial_no - 1)

        rewards = [np.random.choice([0, 1], p=[1 - p_0, p_0]), np.random.choice([0, 1], p=[1 - p_1, p_1])]

        self.rewards_used = [self.rewards_used[0] + rewards[0], self.rewards_used[1] + rewards[1]]

        return rewards[action]


class ymaze_gammatest(ymaze_fixedreward):
    """
    A class for creating an OpenAI gym environment for a future bait Y-maze task.
    """

    def __init__(self, n_trials_per_session, reward_fraction=None, random_seed=None):
        """
        A constructor for a future bait Y-maze environment
        ==================================================
        Parameters:
        n_trials_per_session: number of trials per episode (int)
        reward_fraction: fraction of rewarded trials in each alternative (float)
        random_seed: random seed (int) (default: None)
        """
        super().__init__(n_trials_per_session, reward_fraction, random_seed)
        self.reward_next = False
        self.punish_next = False
        self.rewards = np.zeros((n_trials_per_session, 2))

    def generator(self, action):
        """
        Generates a random reward
        =========================

        Parameters:
        action: action to be performed (int)
        state_vector: state vector (np.array)
        """

        p_0 = np.clip(
            (self.n_rewarded_trials[0] - self.rewards_used[0]) / (self.n_trials_per_session - self.trial_no + 1), 0, 1
        )
        p_1 = np.clip(
            (self.n_rewarded_trials[1] - self.rewards_used[1]) / (self.n_trials_per_session - self.trial_no + 1), 0, 1
        )

        if self.reward_next:
            if p_0 > 0.0:
                self.rewards[self.trial_no - 1, 0] = 1
            else:
                self.rewards[self.trial_no - 1, 0] = 0
            self.rewards[self.trial_no - 1, 1] = np.random.choice([0, 1], p=[1 - p_1, p_1])
            self.reward_next = False
        elif self.punish_next:
            self.rewards[self.trial_no - 1, 0] = np.random.choice([0, 1], p=[1 - p_0, p_0])
            if p_1 < 1.0:
                self.rewards[self.trial_no - 1, 1] = 0
            else:
                self.rewards[self.trial_no - 1, 1] = 1
            self.punish_next = False
        else:
            self.rewards[self.trial_no - 1, 0] = np.random.choice([0, 1], p=[1 - p_0, p_0])
            self.rewards[self.trial_no - 1, 1] = np.random.choice([0, 1], p=[1 - p_1, p_1])

        self.rewards_used = [
            self.rewards_used[0] + self.rewards[self.trial_no - 1, 0],
            self.rewards_used[1] + self.rewards[self.trial_no - 1, 1],
        ]

        reward = self.rewards[self.trial_no - 1, action]

        if action == 0 and reward == 1:
            self.reward_next = True
        elif action == 1 and reward == 1:
            self.punish_next = True

        return reward


class ymaze_RNN(ymaze_fixedreward):
    def __init__(self, n_trials_per_session, reward_fraction=None, random_seed=None, reward_network=None):
        super().__init__(n_trials_per_session, reward_fraction, random_seed)
        self.reward_network = reward_network
        self.action_history = [0]
        self.reward_presentation_history = []
        self.reward_history = [0]
        self.schedule = []

    def generator(self, action):
        state_vector = torch.tensor(
            np.array([np.arange(len(self.action_history)), self.action_history, self.reward_history]).T,
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            reward_presentation = (
                self.reward_network(state_vector)[:, -1, :].squeeze().softmax(dim=0).multinomial(1).item()
            )
        self.reward_presentation_history.append(reward_presentation)
        p_0 = (self.n_rewarded_trials[0] - self.rewards_used[0]) / (self.n_trials_per_session - self.trial_no + 1)
        p_1 = (self.n_rewarded_trials[1] - self.rewards_used[1]) / (self.n_trials_per_session - self.trial_no + 1)

        if p_0 >= 1 and p_1 >= 1 and self.reward_fraction is not None:
            reward = 1
            self.schedule.append([1, 1])
            # self.reward_presentation_history.append(3)
            self.rewards_used[0] += 1
            self.rewards_used[1] += 1
        elif p_0 >= 1 and self.reward_fraction is not None:
            reward = 1 - action
            self.schedule.append([1, 0])
            # self.reward_presentation_history.append(1)
            self.rewards_used[0] += 1
        elif p_1 >= 1 and self.reward_fraction is not None:
            reward = action
            self.schedule.append([0, 1])
            # self.reward_presentation_history.append(2)
            self.rewards_used[1] += 1
        else:
            if reward_presentation == 0:
                reward = 0
                self.schedule.append([0, 0])
                # self.reward_presentation_history.append(0)
            elif reward_presentation == 1:
                if self.n_rewarded_trials[0] - self.rewards_used[0] > 0:
                    reward = 1 - action
                    self.rewards_used[0] += 1
                    self.schedule.append([1, 0])
                    # self.reward_presentation_history.append(1)
                else:
                    reward = 0
                    self.schedule.append([0, 0])
                    # self.reward_presentation_history.append(0)
            elif reward_presentation == 2:
                if self.n_rewarded_trials[1] - self.rewards_used[1] > 0:
                    reward = action
                    self.rewards_used[1] += 1
                    self.schedule.append([0, 1])
                    # self.reward_presentation_history.append(2)
                else:
                    reward = 0
                    self.schedule.append([0, 0])
                    # self.reward_presentation_history.append(0)
            else:
                if (
                    self.n_rewarded_trials[0] - self.rewards_used[0] > 0
                    and self.n_rewarded_trials[1] - self.rewards_used[1] > 0
                ):
                    reward = 1
                    self.rewards_used[0] += 1
                    self.rewards_used[1] += 1
                    self.schedule.append([1, 1])
                    # self.reward_presentation_history.append(3)
                elif self.n_rewarded_trials[0] - self.rewards_used[0] > 0:
                    reward = 1 - action
                    self.rewards_used[0] += 1
                    self.schedule.append([1, 0])
                    # self.reward_presentation_history.append(1)
                elif self.n_rewarded_trials[1] - self.rewards_used[1] > 0:
                    reward = action
                    self.rewards_used[1] += 1
                    self.schedule.append([0, 1])
                    # self.reward_presentation_history.append(2)
                else:
                    reward = 0
                    self.schedule.append([0, 0])
                    # self.reward_presentation_history.append(0)
        self.reward_history.append(reward)
        self.action_history.append(action)
        return reward

    def reset(self):
        super().reset()
        self.action_history = [0]
        self.reward_presentation_history = []
        self.reward_history = [0]
        self.schedule = []
        return 0

    def get_state_history(self):
        return torch.tensor(
            np.array([np.arange(len(self.action_history)), self.action_history, self.reward_history]).T,
            dtype=torch.float32,
        ).unsqueeze(0)

    def get_reward_presentation_history(self):
        return torch.tensor(np.array(self.reward_presentation_history), dtype=torch.float32).unsqueeze(0).unsqueeze(2)


class ymaze_baiting(ymaze_dynamic):
    """
    A class for creating and OpenAI gym environment for a "baited" foraging task.
    """
    def __init__(self, n_trials_per_session, n_trial_per_blocks, baiting_probabilities):
        """
        A constructor for a Dynamic Y-maze environment
        ==============================================
        Parameters:
        n_trials_per_session: number of trials per episode (int)
        n_trial_per_blocks: number of trials per block (list of ints) 
        baiting_probabilities: probabilities of baiting (np.array) (shape: n_blocks x 2)
        """
        super().__init__(n_trials_per_session)
        
        assert len(baiting_probabilities) == len(n_trial_per_blocks), "Number of blocks and baiting probabilities must be equal"
        assert sum(n_trial_per_blocks) == n_trials_per_session, "Sum of number of trials per block must be equal to number of trials per session"
        self.odor1_probabilities = np.concatenate([np.array([baiting_probabilities[i][0]]*n_trial_per_blocks[i]) for i in range(len(n_trial_per_blocks))])
        self.odor2_probabilities = np.concatenate([np.array([baiting_probabilities[i][1]]*n_trial_per_blocks[i]) for i in range(len(n_trial_per_blocks))])
        self.rewardstate  = [0,0]
    
    def generator(self, action):
        
        if self.rewardstate[0]!=1: # Not Baited condition
            p = self.odor1_probabilities[self.trial_no-1]
            self.rewardstate[0] = np.random.choice([0,1], p=[1-p,p])
    
        if self.rewardstate[1] !=1: # Not Baited condition
            p = self.odor2_probabilities[self.trial_no-1]
            self.rewardstate[1] = np.random.choice([0,1], p=[1-p,p])

        reward = self.rewardstate[action]
        self.rewardstate[action] = 0 # Reset reward state

        return reward