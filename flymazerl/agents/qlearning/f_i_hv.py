import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
import arviz as az

from flymazerl.agents.base import FlYMazeAgent

class FHQLearner(FlYMazeAgent):
    def init_variables(self, alpha_r=0.1, alpha_h=0.1, weight_r=0.1, weight_h=0.1, weight_b=0.0):
        """
        Initialize variables for the Forgetting Habitual Q-Learning agent
        ======================================================

        Parameters:
        alpha_r: learning rate for reward value (float)
        alpha_h: learning rate for habit value (float)
        weight_r: weight for reward value (float)
        weight_h: weight for habit value (float)
        weight_b: bias (float)
        """
        self.alpha_r = alpha_r  # Reward learning rate
        self.alpha_h = alpha_h  # Habit learning rate
        self.weight_r = weight_r  # Arbiter weight for reward
        self.weight_h = weight_h  # Arbiter weight for habits
        self.weight_b = weight_b  # Arbiter bias
        self.w = 0.5  # Controller weight
        self.r_table = np.zeros((self.state_space_size, self.action_space_size))  # Reward table
        self.h_table = np.zeros((self.state_space_size, self.action_space_size))  # Habit table

    def init_recorder(self):
        """
        Initialize recorder for the Forgetting Habitual Q-Learning agent
        """
        self.r_history = []
        self.h_history = []
        self.w_history = []
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Forgetting Habitual Q-Learning step in the environment
        - Make a choice based on the current state
        - Get reward and new state
        - Update according to the equation:
            R[state, action] = (1 - alpha_r) * R[state, action] + alpha_r * reward
            R[state, other action] = (1 - alpha_r) * R[state, other action]
            H[state, action] = (1- alpha_h) * H[state, action] + alpha_h * habit
            H[state, other actions] = (1- alpha_h) * H[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)[0]  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update values

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        action_probabilities: action probabilities (list)
        """
        action_probabilities = np.ones(self.action_space_size) / self.action_space_size  # uniform distribution
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # random choice
        return action, action_probabilities

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update value-table of the agent according to the Forgetting Habitual Q-Learning equation:
            R[state, action] = (1 - alpha_r) * R[state, action] + alpha_r * reward
            R[state, other action] = (1 - alpha_r) * R[state, other action]
            H[state, action] = (1- alpha_h) * H[state, action] + alpha_h
            H[state, other actions] = (1- alpha_h) * H[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.r_table[state, :] = (1 - self.alpha_r) * self.r_table[state, :]
        self.r_table[state, action] += self.alpha_r * reward  # update reward table

        self.h_table[state, :] = (1 - self.alpha_h) * self.h_table[state, :]  # update habit table
        self.h_table[state, action] += self.alpha_h * 1  # update habit table

        expected_reward = np.dot(self.r_table[state, :], self.make_choice(state)[1])
        reward_variance = np.abs((self.r_table[state, :] - expected_reward))
        g = np.dot(reward_variance, self.make_choice(state)[1])
        expected_habit = np.mean(self.h_table[state, :])
        habit_variance = np.abs(self.h_table[state, :] - expected_habit)
        h = np.mean(habit_variance)
        self.w = 1 / (1 + np.exp(self.weight_r * g - self.weight_h * h + self.weight_b))

        if self.history:
            self.r_history.append([self.r_table[0, 0], self.r_table[0, 1]])
            self.h_history.append([self.h_table[0, 0], self.h_table[0, 1]])
            self.w_history.append(self.w)
            self.q_history.append(
                [
                    self.w * self.theta_r * self.r_table[0, 0] + (1 - self.w) * self.theta_h * self.h_table[0, 0],
                    self.w * self.theta_r * self.r_table[0, 1] + (1 - self.w) * self.theta_h * self.h_table[0, 1],
                ]
            )

    def run_episode(self):
        """
        Describe a single episode of the Forgetting Habitual Q-Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Forgetting Habitual Q-Learning agent
        """
        self.r_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset reward table
        self.h_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset habit table

    def reset_recorder(self):
        """
        Reset the recorder for the Forgetting Habitual Q-Learning agent
        """
        self.r_history = []
        self.h_history = []
        self.w_history = []
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class FHQLearner_egreedy(FHQLearner):
    def init_policy_variables(self, theta_r=1, theta_h=1, epsilon=0.1):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        theta_r: reward scaling factor (float) (default: 0.1)
        theta_h: habit scaling factor (float) (default: 0.1)
        epsilon: exploration rate (float) (default: 0.1)
        """
        self.theta_r = theta_r  # Reward scaling parameter
        self.theta_h = theta_h  # Habit scaling factor
        self.epsilon = epsilon  # Exploration rate

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        action_probabilities: action probabilities (numpy array)
        """
        action_probabilities = np.ones(self.action_space_size) * (self.epsilon / self.action_space_size)
        action_probabilities[
            np.argmax(
                self.w * self.theta_r * self.r_table[state, :] + (1 - self.w) * self.theta_h * self.h_table[state, :]
            )
        ] += (1 - self.epsilon)
        action_probabilities /= np.sum(action_probabilities)
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action, action_probabilities

    def vectorizedUpdate(
        self, action, reward, tables, alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon
    ):
        """
        Vectorized update of value-table using Forgetting Habitual Q-learning Algorithm
        ====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: learning rate for reward value (Theano tensor)
        alpha_h: learning rate for habit value (Theano tensor)
        weight_r: reward weight (Theano tensor)
        weight_h: habit weight (Theano tensor)
        weight_b: bias (Theano tensor)
        theta_r: reward scaling factor (Theano tensor)
        theta_h: habit scaling factor (Theano tensor)
        epsilon: exploration rate (Theano tensor)

        Returns:
        tables: updated value tables (Theano tensor)
        """
        tables = tt.set_subtensor(tables[: self.action_space_size], (1 - alpha_r) * tables[: self.action_space_size],)
        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * reward,)
        tables = tt.set_subtensor(
            tables[self.action_space_size : 2 * self.action_space_size],
            (1 - alpha_h) * tables[self.action_space_size : 2 * self.action_space_size],
        )
        tables = tt.set_subtensor(
            tables[self.action_space_size + action], tables[self.action_space_size + action] + alpha_h
        )

        Ds = (
            tables[2 * self.action_space_size] * theta_r * tables[: self.action_space_size]
            + (1 - tables[2 * self.action_space_size])
            * theta_h
            * tables[self.action_space_size : 2 * self.action_space_size]
        )
        Ds_ = tt.cast(tt.eq(Ds, tt.max(Ds, keepdims=True)), "float64")
        pDs = (1 - epsilon) * Ds_ + epsilon / self.action_space_size

        g = tt.dot(pDs, tt.abs_((tables[: self.action_space_size] - tt.dot(pDs, tables[: self.action_space_size]))))
        h = tt.sum(
            tt.abs_(
                (
                    tables[self.action_space_size : 2 * self.action_space_size]
                    - tt.mean(tables[self.action_space_size : 2 * self.action_space_size])
                )
            )
        )
        w = 1 / (1 + tt.exp(weight_r * g - weight_h * h + weight_b))
        return tt.concatenate([tables[: 2 * self.action_space_size], tt.reshape(w, (1,))])

    def vectorizedActionProbabilities(
        self, alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon, actions_set, rewards_set
    ):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha_r: learning rate for reward value (Theano tensor)
        alpha_h: learning rate for habit value (Theano tensor)
        weight_r: reward weight (Theano tensor)
        weight_h: habit weight (Theano tensor)
        weight_b: bias (Theano tensor)
        theta_r: reward scaling factor (Theano tensor)
        theta_h: habit scaling factor (Theano tensor)
        epsilon: exploration rate (float)
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)

        Returns:
        action_probabilities: action probabilities (Theano tensor)
        """
        probabilities = []
        for (actions_, rewards_) in zip(actions_set, rewards_set):  # for each session
            rewards = theano.shared(np.asarray(rewards_, dtype="int16"))
            actions = theano.shared(np.asarray(actions_, dtype="int16"))

            # Compute the Qs values
            tables = tt.zeros((2 * self.action_space_size + 1), dtype="float64")
            tables, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards],
                outputs_info=[tables],
                non_sequences=[alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon],
            )

            Qs = tables[:, : self.action_space_size]
            Hs = tables[:, self.action_space_size : 2 * self.action_space_size]
            w = tt.extra_ops.repeat(tables[:, 2 * self.action_space_size :], self.action_space_size, axis=1)
            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((self.action_space_size), dtype="float64")], Qs], axis=0)
            Qs_ = Qs_[:-1]
            Hs_ = tt.concatenate([[tt.zeros((self.action_space_size), dtype="float64")], Hs], axis=0)
            Hs_ = Hs_[:-1]
            w_ = tt.concatenate([[0.5 * tt.ones((self.action_space_size), dtype="float64")], w], axis=0)
            w_ = w_[:-1]
            Ds_ = w_ * theta_r * Qs_ + (1 - w_) * theta_h * Hs_
            prob_actions = tt.cast(tt.eq(Ds_, tt.max(Ds_, axis=1, keepdims=True)), "float64")

            # Adjust and store probabilities according to epsilon-greedy policy
            probabilities.append((1 - epsilon) * prob_actions[:, 1] + epsilon / self.action_space_size)
        action_probabilities = tt.concatenate(probabilities, axis=0)
        return action_probabilities

    def fit(
        self,
        actions_set,
        rewards_set,
        alpha_r_params=(1, 1),
        alpha_h_params=(1, 1),
        weight_r_params=(0, 1),
        weight_h_params=(0, 1),
        weight_b_params=(0, 1),
        theta_r_params=(1, 1),
        theta_h_params=(1, 1),
        epsilon_params=(1, 1),
        niters=1000,
        ntune=1000,
        nchains=2,
        nparallel=1,
        target_accept=0.8,
        plot_trace=False,
        plot_posterior=False,
        print_summary=True,
    ):
        """
        Fit the epsilon-greedy Forgetting Habitual Q-Learning model
        ================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_r_params: Reward learning rate beta prior parameters (default: (1,1))
        alpha_h_params: Habit learning rate beta prior parameters (default: (1,1))
        weight_r_params: Reward weight normal prior parameters (default: (0,1))
        weight_h_params: Habit weight normal prior parameters (default: (0,1))
        weight_b_params: Bias beta prior parameters (default: (0,1))
        theta_r_params: Reward scaling beta prior parameters (default: (1,1))
        theta_h_params: Habit scaling beta prior parameters (default: (1,1))
        epsilon_params: Exploration rate beta prior parameters (default: (1,1))
        niters: number of iterations in the MC-MC sampling (default: 1000)
        ntune: number of tuning iterations in the MC-MC sampling (default: 1000)
        nchains: number of parallel chains (default: 2)
        nparallel: number of parallel sampling processes (default: 1)
        target_accept: target acceptance rate (default: 0.8)
        plot_trace: plot trace of posterior distributions (default: True)
        plot_posterior: plot posterior distributions (default: True)
        print_summary: print summary of posterior distributions (default: False)

        Returns:
        model: Bayesian Model (PyMC3 object)
        trace: trace of posterior distributions (Arviz InferenceData object)
        """
        with pm.Model() as model:  # define model
            alpha_r = pm.Beta("alpha_r", alpha_r_params[0], alpha_r_params[1])  # reward learning rate
            alpha_h = pm.Beta("alpha_h", alpha_h_params[0], alpha_h_params[1])  # habit learning rate
            weight_r = pm.Normal("weight_r", weight_r_params[0], weight_r_params[1])  # reward weight
            weight_h = pm.Normal("weight_h", weight_h_params[0], weight_h_params[1])  # habit weight
            weight_b = pm.Normal("weight_b", weight_b_params[0], weight_b_params[1])  # bias
            theta_r = pm.Beta("theta_r", theta_r_params[0], theta_r_params[1])  # reward scaling
            theta_h = pm.Beta("theta_h", theta_h_params[0], theta_h_params[1])  # habit scaling
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon, actions_set, rewards_set
            )  # action probabilities
            like = pm.Bernoulli(
                "like", p=action_probabilities, observed=np.concatenate([a for a in actions_set])
            )  # Bernoulli likelihood
            trace = pm.sample(
                tune=ntune,
                draws=niters,
                chains=nchains,
                cores=nparallel,
                return_inferencedata=True,
                target_accept=target_accept,
            )  # sample posterior

        if plot_trace:  # plot trace
            az.plot_trace(trace)

        if plot_posterior:  # plot posterior
            az.plot_posterior(trace, hdi_prob=0.95)

        summary = az.summary(trace, hdi_prob=0.95)  # compute summary

        if print_summary:  # print summary
            print(summary)

        return model, trace


class FHQLearner_softmax(FHQLearner):
    def init_policy_variables(self, theta_r=1, theta_h=1, beta=0.2):
        """
        Initialize variables for the softmax policy
        ===========================================

        Parameters:
        theta_r: reward scaling (default: 1)
        theta_h: habit scaling (default: 1)
        beta: temperature for softmax (default: 0.2)
        """
        self.theta_r = theta_r  # Reward scaling parameter
        self.theta_h = theta_h  # Habit scaling factor
        self.beta = beta  # Temperature for softmax

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        action_probabilities: action probabilities (numpy array)
        """
        action_probabilities = np.exp(
            (self.w * self.theta_r * self.r_table[state, :] + (1 - self.w) * self.theta_h * self.h_table[state, :])
            / self.beta
        )
        action_probabilities /= np.sum(action_probabilities)
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action, action_probabilities

    def vectorizedUpdate(
        self, action, reward, tables, alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta
    ):
        """
        Vectorized update of value-table using Forgetting Habitual Q-learning Algorithm
        ====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: learning rate for reward value (Theano tensor)
        alpha_h: learning rate for habit value (Theano tensor)
        weight_r: reward weight (Theano tensor)
        weight_h: habit weight (Theano tensor)
        weight_b: bias (Theano tensor)
        theta_r: reward scaling factor (Theano tensor)
        theta_h: habit scaling factor (Theano tensor)
        beta: temperature for softmax (Theano tensor)

        Returns:
        tables: updated value tables (Theano tensor)
        """
        tables = tt.set_subtensor(tables[: self.action_space_size], (1 - alpha_r) * tables[: self.action_space_size],)
        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * reward,)
        tables = tt.set_subtensor(
            tables[self.action_space_size : 2 * self.action_space_size],
            (1 - alpha_h) * tables[self.action_space_size : 2 * self.action_space_size],
        )
        tables = tt.set_subtensor(
            tables[self.action_space_size + action], tables[self.action_space_size + action] + alpha_h
        )

        Ds = (
            tables[2 * self.action_space_size] * theta_r * tables[: self.action_space_size]
            + (1 - tables[2 * self.action_space_size])
            * theta_h
            * tables[self.action_space_size : 2 * self.action_space_size]
        )
        Ds_ = Ds / beta
        pDs = tt.exp(Ds_ - pm.math.logsumexp(Ds_))

        g = tt.dot(pDs, tt.abs_((tables[: self.action_space_size] - tt.dot(pDs, tables[: self.action_space_size]))))
        h = tt.sum(
            tt.abs_(
                (
                    tables[self.action_space_size : 2 * self.action_space_size]
                    - tt.mean(tables[self.action_space_size : 2 * self.action_space_size])
                )
            )
        )
        w = 1 / (1 + tt.exp(weight_r * g - weight_h * h + weight_b))
        return tt.concatenate([tables[: 2 * self.action_space_size], tt.reshape(w, (1,))])

    def vectorizedActionProbabilities(
        self, alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, actions_set, rewards_set
    ):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha_r: reward learning rate (Theano tensor)
        alpha_h: habit learning rate (Theano tensor)
        weight_r: reward weight (Theano tensor)
        weight_h: habit weight (Theano tensor)
        weight_b: bias (Theano tensor)
        theta_r: reward scaling (Theano tensor)
        theta_h: habit scaling (Theano tensor)
        beta: temperature for softmax (Theano tensor)
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)

        Returns:
        action_probabilities: action probabilities (Theano tensor)
        """
        probabilities = []
        for (actions_, rewards_) in zip(actions_set, rewards_set):  # for each session
            rewards = theano.shared(np.asarray(rewards_, dtype="int16"))
            actions = theano.shared(np.asarray(actions_, dtype="int16"))

            # Compute the Qs values
            tables = tt.zeros((2 * self.action_space_size + 1), dtype="float64")
            tables, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards],
                outputs_info=[tables],
                non_sequences=[alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta],
            )

            Qs = tables[:, : self.action_space_size]
            Hs = tables[:, self.action_space_size : 2 * self.action_space_size]
            w = tt.extra_ops.repeat(tables[:, 2 * self.action_space_size :], self.action_space_size, axis=1)
            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((self.action_space_size), dtype="float64")], Qs], axis=0)
            Qs_ = Qs_[:-1]
            Hs_ = tt.concatenate([[tt.zeros((self.action_space_size), dtype="float64")], Hs], axis=0)
            Hs_ = Hs_[:-1]
            w_ = tt.concatenate([[0.5 * tt.ones((self.action_space_size), dtype="float64")], w], axis=0)
            w_ = w_[:-1]
            Ds_ = (w_ * theta_r * Qs_ + (1 - w_) * theta_h * Hs_) / beta
            log_prob_actions = Ds_ - pm.math.logsumexp(Ds_, axis=1)

            # Store probabilities
            probabilities.append(tt.exp(log_prob_actions[:, 1]))
        action_probabilities = tt.concatenate(probabilities, axis=0)
        return action_probabilities

    def fit(
        self,
        actions_set,
        rewards_set,
        alpha_r_params=(1, 1),
        alpha_h_params=(1, 1),
        weight_r_params=(0, 1),
        weight_h_params=(0, 1),
        weight_b_params=(0, 1),
        theta_r_params=(1, 1),
        theta_h_params=(1, 1),
        beta_params=(1,),
        niters=1000,
        ntune=1000,
        nchains=2,
        nparallel=1,
        target_accept=0.8,
        plot_trace=False,
        plot_posterior=False,
        print_summary=True,
    ):
        """
        Fit the softmax Forgetting Habitual Q-Learning model
        =========================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_r_params: Reward learning rate beta prior parameters (default: (1,1))
        alpha_h_params: Habit learning rate beta prior parameters (default: (1,1))
        weight_r_params: Reward weight normal prior parameters (default: (0,1))
        weight_h_params: Habit weight normal prior parameters (default: (0,1))
        weight_b_params: Bias normal prior parameters (default: (0,1))
        theta_r_params: Reward scaling beta prior parameters (default: (1,1))
        theta_h_params: Habit scaling beta prior parameters (default: (1,1))
        beta_params: Temperature beta prior parameters (default: (1,))
        niters: number of iterations in the MC-MC sampling (default: 1000)
        ntune: number of tuning iterations in the MC-MC sampling (default: 1000)
        nchains: number of parallel chains (default: 2)
        nparallel: number of parallel sampling processes (default: 1)
        target_accept: target acceptance rate (default: 0.8)
        plot_trace: plot trace of posterior distributions (default: True)
        plot_posterior: plot posterior distributions (default: True)
        print_summary: print summary of posterior distributions (default: False)

        Returns:
        model: Bayesian Model (PyMC3 object)
        trace: trace of posterior distributions (Arviz InferenceData object)
        """
        with pm.Model() as model:  # define model
            alpha_r = pm.Beta("alpha_r", alpha_r_params[0], alpha_r_params[1])  # reward learning rate
            alpha_h = pm.Beta("alpha_h", alpha_h_params[0], alpha_h_params[1])  # habit learning rate
            weight_r = pm.Normal("weight_r", weight_r_params[0], weight_r_params[1])  # reward weight
            weight_h = pm.Normal("weight_h", weight_h_params[0], weight_h_params[1])  # habit weight
            weight_b = pm.Normal("weight_b", weight_b_params[0], weight_b_params[1])  # bias
            theta_r = pm.Beta("theta_r", theta_r_params[0], theta_r_params[1])  # reward scaling
            theta_h = pm.Beta("theta_h", theta_h_params[0], theta_h_params[1])  # habit scaling
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax

            action_probabilities = self.vectorizedActionProbabilities(
                alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, actions_set, rewards_set
            )  # action probabilities
            like = pm.Bernoulli(
                "like", p=action_probabilities, observed=np.concatenate([a for a in actions_set])
            )  # Bernoulli likelihood
            trace = pm.sample(
                tune=ntune,
                draws=niters,
                chains=nchains,
                cores=nparallel,
                return_inferencedata=True,
                target_accept=target_accept,
            )  # sample posterior

        if plot_trace:  # plot trace
            az.plot_trace(trace)

        if plot_posterior:  # plot posterior
            az.plot_posterior(trace, hdi_prob=0.95)

        summary = az.summary(trace, hdi_prob=0.95)  # compute summary

        if print_summary:  # print summary
            print(summary)

        return model, trace


class FHQLearner_esoftmax(FHQLearner):
    def init_policy_variables(self, theta_r=1, theta_h=1, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy.
        ====================================================

        Parameters:
        theta_r: reward scaling (default: 1)
        theta_h: habit scaling (default: 1)
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.theta_r = theta_r  # Reward scaling parameter
        self.theta_h = theta_h  # Habit scaling factor
        self.epsilon = epsilon  # Exploration rate
        self.beta = beta  # Temperature for softmax

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy.
        ====================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        action_probabilities: action probabilities (numpy array)
        """
        action_probabilities = np.exp(
            (self.w * self.theta_r * self.r_table[state, :] + (1 - self.w) * self.theta_h * self.h_table[state, :])
            / self.beta
        )
        action_probabilities /= np.sum(action_probabilities)
        action_probabilities = action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action, action_probabilities

    def vectorizedUpdate(
        self, action, reward, tables, alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, epsilon,
    ):
        """
        Vectorized update of Q-table using Forgetting Habitual Q-Learning model
        ============================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: reward learning rate (Theano tensor)
        alpha_h: habit learning rate (Theano tensor)
        weight_r: reward weight (Theano tensor)
        weight_h: habit weight (Theano tensor)
        weight_b: bias (Theano tensor)
        theta_r: reward scaling (Theano tensor)
        theta_h: habit scaling (Theano tensor)
        beta: temperature for softmax (Theano tensor)

        Returns:
        tables: updated value tables (Theano tensor)
        """

        tables = tt.set_subtensor(tables[: self.action_space_size], (1 - alpha_r) * tables[: self.action_space_size],)
        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * reward,)
        tables = tt.set_subtensor(
            tables[self.action_space_size : 2 * self.action_space_size],
            (1 - alpha_h) * tables[self.action_space_size : 2 * self.action_space_size],
        )
        tables = tt.set_subtensor(
            tables[self.action_space_size + action], tables[self.action_space_size + action] + alpha_h
        )

        Ds = (
            tables[2 * self.action_space_size] * theta_r * tables[: self.action_space_size]
            + (1 - tables[2 * self.action_space_size])
            * theta_h
            * tables[self.action_space_size : 2 * self.action_space_size]
        )
        Ds_ = Ds / beta
        pDs = (1 - epsilon) * tt.exp(Ds_ - pm.math.logsumexp(Ds_)) + epsilon / self.action_space_size

        g = tt.dot(pDs, tt.abs_((tables[: self.action_space_size] - tt.dot(pDs, tables[: self.action_space_size]))))
        h = tt.sum(
            tt.abs_(
                (
                    tables[self.action_space_size : 2 * self.action_space_size]
                    - tt.mean(tables[self.action_space_size : 2 * self.action_space_size])
                )
            )
        )
        w = 1 / (1 + tt.exp(weight_r * g - weight_h * h + weight_b))
        return tt.concatenate([tables[: 2 * self.action_space_size], tt.reshape(w, (1,))])

    def vectorizedActionProbabilities(
        self, alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, epsilon, actions_set, rewards_set,
    ):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha_r: reward learning rate (float)
        alpha_h: habit learning rate (float)
        weight_r: reward weight (float)
        weight_h: habit weight (float)
        weight_b: bias (float)
        theta_r: reward scaling (float)
        theta_h: habit scaling (float)
        beta: temperature for softmax (float)
        epsilon: exploration rate (float)
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)

        Returns:
        action_probabilities: action probabilities (Theano tensor)
        """
        probabilities = []
        ws = []
        for (actions_, rewards_) in zip(actions_set, rewards_set):  # for each session
            rewards = theano.shared(np.asarray(rewards_, dtype="int16"))
            actions = theano.shared(np.asarray(actions_, dtype="int16"))

            # Compute the Qs values
            tables = tt.zeros((2 * self.action_space_size + 1), dtype="float64")
            tables, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards],
                outputs_info=[tables],
                non_sequences=[alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, epsilon],
            )

            Qs = tables[:, : self.action_space_size]
            Hs = tables[:, self.action_space_size : 2 * self.action_space_size]
            w = tt.extra_ops.repeat(tables[:, 2 * self.action_space_size :], self.action_space_size, axis=1)
            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((self.action_space_size), dtype="float64")], Qs], axis=0)
            Qs_ = Qs_[:-1]
            Hs_ = tt.concatenate([[tt.zeros((self.action_space_size), dtype="float64")], Hs], axis=0)
            Hs_ = Hs_[:-1]
            w_ = tt.concatenate([[0.5 * tt.ones((self.action_space_size), dtype="float64")], w], axis=0)
            w_ = w_[:-1]
            # Ds_ = w_ * Qs_ + (1 - w_) * Hs_
            Ds_ = (w_ * theta_r * Qs_ + (1 - w_) * theta_h * Hs_) / beta
            log_prob_actions = Ds_ - pm.math.logsumexp(Ds_, axis=1)

            # Store probabilities
            probabilities.append((1 - epsilon) * tt.exp(log_prob_actions[:, 1]) + epsilon / self.action_space_size)
            ws.append(w_)
        action_probabilities = tt.concatenate(probabilities, axis=0)
        ws = tt.concatenate(ws, axis=0)
        return tt.clip(action_probabilities, 0.001, 0.999), ws

    def fit(
        self,
        actions_set,
        rewards_set,
        alpha_r_params=(1, 1),
        alpha_h_params=(1, 1),
        weight_r_params=(0, 1),
        weight_h_params=(0, 1),
        weight_b_params=(0, 1),
        theta_r_params=(1, 1),
        theta_h_params=(1, 1),
        beta_params=(0.1,),
        epsilon_params=(1, 1),
        niters=1000,
        ntune=1000,
        nchains=2,
        nparallel=1,
        target_accept=0.8,
        plot_trace=False,
        plot_posterior=False,
        print_summary=True,
    ):
        """
        Fit the epsilon-softmax Forgetting Habitual Q-learning model
        =================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_r_params: Reward learning rate beta prior parameters (default: (1,1))
        alpha_h_params: Habit learning rate beta prior parameters (default: (1,1))
        weight_r_params: Reward weight normal prior parameters (default: (0,1))
        weight_h_params: Habit weight normal prior parameters (default: (0,1))
        weight_b_params: Bias normal prior parameters (default: (0,1))
        theta_r_params: Reward scaling beta prior parameters (default: (1,1))
        theta_h_params: Habit scaling beta prior parameters (default: (1,1))
        beta_params: Temperature beta prior parameters (default: (1,))
        epsilon_params: Exploration rate beta prior parameters (default: (1,1))
        niters: number of iterations in the MC-MC sampling (default: 1000)
        ntune: number of tuning iterations in the MC-MC sampling (default: 1000)
        nchains: number of parallel chains (default: 2)
        nparallel: number of parallel sampling processes (default: 1)
        target_accept: target acceptance rate (default: 0.8)
        plot_trace: plot trace of posterior distributions (default: True)
        plot_posterior: plot posterior distributions (default: True)
        print_summary: print summary of posterior distributions (default: False)

        Returns:
        model: Bayesian Model (PyMC3 object)
        trace: trace of posterior distributions (Arviz InferenceData object)
        """
        with pm.Model() as model:  # define model
            alpha_r = pm.Beta("alpha_r", alpha_r_params[0], alpha_r_params[1])  # reward learning rate
            alpha_h = pm.Beta("alpha_h", alpha_h_params[0], alpha_h_params[1])  # habit learning rate
            weight_r = pm.Normal("weight_r", weight_r_params[0], weight_r_params[1])  # reward weight
            weight_h = pm.Normal("weight_h", weight_h_params[0], weight_h_params[1])  # habit weight
            weight_b = pm.Normal("weight_b", weight_b_params[0], weight_b_params[1])  # bias weight
            theta_r = pm.Beta("theta_r", theta_r_params[0], theta_r_params[1])  # reward scaling
            theta_h = pm.Beta("theta_h", theta_h_params[0], theta_h_params[1])  # habit scaling
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate

            action_probabilities = self.vectorizedActionProbabilities(
                alpha_r,
                alpha_h,
                weight_r,
                weight_h,
                weight_b,
                theta_r,
                theta_h,
                beta,
                epsilon,
                actions_set,
                rewards_set,
            )[
                0
            ]  # action probabilities
            like = pm.Bernoulli(
                "like", p=action_probabilities, observed=np.concatenate([a for a in actions_set])
            )  # Bernoulli likelihood
            trace = pm.sample(
                tune=ntune,
                draws=niters,
                chains=nchains,
                cores=nparallel,
                return_inferencedata=True,
                target_accept=target_accept,
                init="adapt_diag",
            )  # sample posterior

        if plot_trace:  # plot trace
            az.plot_trace(trace)

        if plot_posterior:  # plot posterior
            az.plot_posterior(trace, hdi_prob=0.95)

        summary = az.summary(trace, hdi_prob=0.95)  # compute summary

        if print_summary:  # print summary
            print(summary)

        return model, trace


class FHQLearner_acceptreject(FHQLearner):
    def init_policy_variables(self, theta_r=1, theta_h=1, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy.
        ====================================================

        Parameters:
        theta_r: reward scaling (default: 1)
        theta_h: habit scaling (default: 1)
        weight: q-value weight (default: 1.0)
        intercept: q-value intercept (default: 0.0)
        """
        self.theta_r = theta_r  # Reward scaling parameter
        self.theta_h = theta_h  # Habit scaling factor
        self.weight = weight  # Q-value weight
        self.intercept = intercept  # Q-value intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy.
        ====================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        action_probabilities: action probabilities (numpy array)
        """

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        q_table = self.w * self.theta_r * self.r_table[state, :] + (1 - self.w) * self.theta_h * self.h_table[state, :]
        accept_probabilities = sigmoid(self.weight * q_table + self.intercept)
        log_prob_action_1 = (
            np.log(accept_probabilities[1])
            + np.log(3 - accept_probabilities[0])
            - np.log(
                3 * accept_probabilities[0]
                + 3 * accept_probabilities[1]
                - 2 * accept_probabilities[0] * accept_probabilities[1]
            )
        )
        action_1_probability = np.exp(log_prob_action_1)
        action_probabilities = np.array([1 - action_1_probability, action_1_probability])
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action, action_probabilities

    def vectorizedUpdate(
        self,
        action,
        reward,
        tables,
        alpha_r,
        alpha_h,
        weight_r,
        weight_h,
        weight_b,
        theta_r,
        theta_h,
        weight,
        intercept,
    ):
        """
        Vectorized update of Q-table using Forgetting Habitual Q-Learning model
        ============================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: reward learning rate (Theano tensor)
        alpha_h: habit learning rate (Theano tensor)
        weight_r: reward weight (Theano tensor)
        weight_h: habit weight (Theano tensor)
        weight_b: bias (Theano tensor)
        theta_r: reward scaling (Theano tensor)
        theta_h: habit scaling (Theano tensor)
        weight: q-value weight (Theano tensor)
        intercept: q-value intercept (Theano tensor)

        Returns:
        tables: updated value tables (Theano tensor)
        """

        tables = tt.set_subtensor(tables[: self.action_space_size], (1 - alpha_r) * tables[: self.action_space_size],)
        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * reward,)
        tables = tt.set_subtensor(
            tables[self.action_space_size : 2 * self.action_space_size],
            (1 - alpha_h) * tables[self.action_space_size : 2 * self.action_space_size],
        )
        tables = tt.set_subtensor(
            tables[self.action_space_size + action], tables[self.action_space_size + action] + alpha_h
        )

        Ds = (
            tables[2 * self.action_space_size] * theta_r * tables[: self.action_space_size]
            + (1 - tables[2 * self.action_space_size])
            * theta_h
            * tables[self.action_space_size : 2 * self.action_space_size]
        )
        Ds_ = 1 / (1 + tt.exp(-(weight * Ds + intercept)))
        pDs = tt.stack(
            [
                Ds_[0] * (3 - Ds_[1]) / (3 * Ds_[0] + 3 * Ds_[1] - 2 * Ds_[0] * Ds_[1]),
                Ds_[1] * (3 - Ds_[0]) / (3 * Ds_[0] + 3 * Ds_[1] - 2 * Ds_[0] * Ds_[1]),
            ]
        )

        g = tt.dot(pDs, tt.abs_((tables[: self.action_space_size] - tt.dot(pDs, tables[: self.action_space_size]))))
        h = tt.sum(
            tt.abs_(
                (
                    tables[self.action_space_size : 2 * self.action_space_size]
                    - tt.mean(tables[self.action_space_size : 2 * self.action_space_size])
                )
            )
        )
        w = 1 / (1 + tt.exp(weight_r * g - weight_h * h + weight_b))
        return tt.concatenate([tables[: 2 * self.action_space_size], tt.reshape(w, (1,))])

    def vectorizedActionProbabilities(
        self,
        alpha_r,
        alpha_h,
        weight_r,
        weight_h,
        weight_b,
        theta_r,
        theta_h,
        weight,
        intercept,
        actions_set,
        rewards_set,
    ):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha_r: reward learning rate (float)
        alpha_h: habit learning rate (float)
        weight_r: reward weight (float)
        weight_h: habit weight (float)
        weight_b: bias (float)
        theta_r: reward scaling (float)
        theta_h: habit scaling (float)
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)

        Returns:
        action_probabilities: action probabilities (Theano tensor)
        """
        probabilities = []
        ws = []
        for (actions_, rewards_) in zip(actions_set, rewards_set):  # for each session
            rewards = theano.shared(np.asarray(rewards_, dtype="int16"))
            actions = theano.shared(np.asarray(actions_, dtype="int16"))

            # Compute the Qs values
            tables = tt.zeros((2 * self.action_space_size + 1), dtype="float64")
            tables, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards],
                outputs_info=[tables],
                non_sequences=[alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, weight, intercept],
            )

            Qs = tables[:, : self.action_space_size]
            Hs = tables[:, self.action_space_size : 2 * self.action_space_size]
            w = tt.extra_ops.repeat(tables[:, 2 * self.action_space_size :], self.action_space_size, axis=1)
            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((self.action_space_size), dtype="float64")], Qs], axis=0)
            Qs_ = Qs_[:-1]
            Hs_ = tt.concatenate([[tt.zeros((self.action_space_size), dtype="float64")], Hs], axis=0)
            Hs_ = Hs_[:-1]
            w_ = tt.concatenate([[0.5 * tt.ones((self.action_space_size), dtype="float64")], w], axis=0)
            w_ = w_[:-1]
            Ds_ = w_ * theta_r * Qs_ + (1 - w_) * theta_h * Hs_
            Ds_ = 1 / (1 + tt.exp(-(weight * Ds_ + intercept)))
            log_prob_action1 = (
                tt.log(Ds_[:, 1])
                + tt.log(3 - Ds_[:, 0])
                - tt.log(3 * Ds_[:, 1] + 3 * Ds_[:, 0] - 2 * Ds_[:, 0] * Ds_[:, 1])
            )

            # Store probabilities
            probabilities.append(tt.exp(log_prob_action1))
            ws.append(w_)
        action_probabilities = tt.concatenate(probabilities, axis=0)
        ws = tt.concatenate(ws, axis=0)
        return tt.clip(action_probabilities, 0.001, 0.999), ws

    def fit(
        self,
        actions_set,
        rewards_set,
        alpha_r_params=(1, 1),
        alpha_h_params=(1, 1),
        weight_r_params=(0, 1),
        weight_h_params=(0, 1),
        weight_b_params=(0, 1),
        theta_r_params=(1, 1),
        theta_h_params=(1, 1),
        weight_params=(0, 1),
        intercept_params=(0, 1),
        niters=1000,
        ntune=1000,
        nchains=2,
        nparallel=1,
        target_accept=0.8,
        plot_trace=False,
        plot_posterior=False,
        print_summary=True,
    ):
        """
        Fit the accept-reject Forgetting Habitual Q-learning model
        =================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_r_params: Reward learning rate beta prior parameters (default: (1,1))
        alpha_h_params: Habit learning rate beta prior parameters (default: (1,1))
        weight_r_params: Reward weight normal prior parameters (default: (0,1))
        weight_h_params: Habit weight normal prior parameters (default: (0,1))
        weight_b_params: Bias normal prior parameters (default: (0,1))
        theta_r_params: Reward scaling beta prior parameters (default: (1,1))
        theta_h_params: Habit scaling beta prior parameters (default: (1,1))
        weight_params: Q-value weight normal prior parameters (default: (0,1))
        intercept_params: Q-value intercept normal prior parameters (default: (0,1))
        niters: number of iterations in the MC-MC sampling (default: 1000)
        ntune: number of tuning iterations in the MC-MC sampling (default: 1000)
        nchains: number of parallel chains (default: 2)
        nparallel: number of parallel sampling processes (default: 1)
        target_accept: target acceptance rate (default: 0.8)
        plot_trace: plot trace of posterior distributions (default: True)
        plot_posterior: plot posterior distributions (default: True)
        print_summary: print summary of posterior distributions (default: False)

        Returns:
        model: Bayesian Model (PyMC3 object)
        trace: trace of posterior distributions (Arviz InferenceData object)
        """
        with pm.Model() as model:  # define model
            alpha_r = pm.Beta("alpha_r", alpha_r_params[0], alpha_r_params[1])  # reward learning rate
            alpha_h = pm.Beta("alpha_h", alpha_h_params[0], alpha_h_params[1])  # habit learning rate
            weight_r = pm.Normal("weight_r", weight_r_params[0], weight_r_params[1])  # reward weight
            weight_h = pm.Normal("weight_h", weight_h_params[0], weight_h_params[1])  # habit weight
            weight_b = pm.Normal("weight_b", weight_b_params[0], weight_b_params[1])  # bias weight
            theta_r = pm.Beta("theta_r", theta_r_params[0], theta_r_params[1])  # reward scaling
            theta_h = pm.Beta("theta_h", theta_h_params[0], theta_h_params[1])  # habit scaling
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # Q-value weight
            intercept = pm.Normal("intercept", intercept_params[0], intercept_params[1])  # Q-value intercept

            action_probabilities = self.vectorizedActionProbabilities(
                alpha_r,
                alpha_h,
                weight_r,
                weight_h,
                weight_b,
                theta_r,
                theta_h,
                weight,
                intercept,
                actions_set,
                rewards_set,
            )[
                0
            ]  # action probabilities
            like = pm.Bernoulli(
                "like", p=action_probabilities, observed=np.concatenate([a for a in actions_set])
            )  # Bernoulli likelihood
            trace = pm.sample(
                tune=ntune,
                draws=niters,
                chains=nchains,
                cores=nparallel,
                return_inferencedata=True,
                target_accept=target_accept,
                init="adapt_diag",
            )  # sample posterior

        if plot_trace:  # plot trace
            az.plot_trace(trace)

        if plot_posterior:  # plot posterior
            az.plot_posterior(trace, hdi_prob=0.95)

        summary = az.summary(trace, hdi_prob=0.95)  # compute summary

        if print_summary:  # print summary
            print(summary)

        return model, trace


