import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
import arviz as az

from flymazerl.agents.base import FlYMazeAgent


class ESARSALearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, gamma=0.5):
        """
        Initialize variables for the Expected SARSA agent.
        ==================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        gamma: discount rate for the agent (float) (default: 0.5)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))

    def init_recorder(self):
        """
        Initialize recorder for the Expected SARSA agent.
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Expected SARSA step in the environment.
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * E(Q[new_state,:]))
        ==============================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action, _ = self.make_choice(state)  # make a choice
        if self.history:
            self.action_history.append(action)
        if action == self.biased_action:
            self.bias += 1  # update bias estimate
        new_state, reward, done, info = self.env.step(action)  # get reward and new state
        if self.history:
            self.reward_history.append(reward)
        self.update_knowledge(state, action, new_state, reward)  # update knowledge

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy.
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # sample action
        action_probabilities = np.ones(self.action_space_size) * (
            1 / self.action_space_size
        )  # uniform action probabilities
        return action, action_probabilities

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Q-learning update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * E(Q[new_state,:]))
        ==============================================================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        _, action_probabilities = self.make_choice(new_state)  # get action probabilities
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (
            reward + self.gamma * np.dot(self.q_table[new_state, :], action_probabilities)
        )  # update Q-table
        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Expected SARSA agent.
        """
        state = self.env.reset()
        done = False
        while not done:
            state, done = self.trial_step(state)

    def reset_variables(self):
        """
        Reset variables for the Expected SARSA agent.
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))

    def reset_recorder(self):
        """
        Reset the recorder for the Expected SARSA agent.
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class ESARSALearner_egreedy(ESARSALearner):
    def init_policy_variables(self, epsilon=0.5):
        """
        Initialize variables for the epsilon-greedy policy.
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy.
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (self.epsilon / self.action_space_size)
        action_probabilities[np.argmax(self.q_table[state, :])] += 1 - self.epsilon
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action, action_probabilities

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma, epsilon):
        """
        Vectorized update of Q-table using Expected SARSA Algorithm.
        ============================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)
        epsilon: exploration rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs_ = tt.cast(tt.eq(Qs, tt.max(Qs, keepdims=True)), "float64")
        pQs = (1 - epsilon) * Qs_ + epsilon / self.action_space_size
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * tt.dot(pQs, Qs) - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, gamma, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        gamma: discount rate (float)
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
            Qs = tt.zeros((2), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma, epsilon],
            )

            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((2), dtype="float64")], Qs], axis=0)
            Qs_ = Qs_[:-1]
            prob_actions = tt.cast(tt.eq(Qs_, tt.max(Qs_, axis=1, keepdims=True)), "float64")

            # Adjust and store probabilities according to epsilon-greedy policy
            probabilities.append((1 - epsilon) * prob_actions[:, 1] + epsilon / self.action_space_size)
        action_probabilities = tt.concatenate(probabilities, axis=0)
        return action_probabilities

    def fit(
        self,
        actions_set,
        rewards_set,
        alpha_params=(1, 1),
        gamma_params=(1, 1),
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
        Fit the epsilon-greedy Expected SARSA agent to a given dataset.
        ==============================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        gamma_params: discount rate beta prior parameters (default: (1,1))
        epsilon_params: exploration rate beta prior parameters (default: (1,1))
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
            alpha = pm.Beta("alpha", alpha_params[0], alpha_params[1])  # learning rate (beta distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, gamma, epsilon, actions_set, rewards_set
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


class ESARSALearner_softmax(ESARSALearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy.
        ============================================

        Parameters:
        beta: temperature for softmax (default: 0.2)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy.
        ============================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.exp(self.q_table[state, :] / self.beta)  # compute action probabilities
        action_probabilities /= np.sum(action_probabilities)  # normalize
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # make choice
        return action, action_probabilities

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma, beta):
        """
        Vectorized update of Q-table using Expected SARSA Algorithm.
        ============================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)
        beta: temperature for softmax (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs_ = Qs / beta
        pQs = tt.exp(Qs_ - pm.math.logsumexp(Qs_))
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * tt.dot(pQs, Qs) - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, gamma, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        gamma: discount rate (float)
        beta: temperature for softmax (float)
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
            Qs = tt.zeros((2), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma, beta],
            )

            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((2), dtype="float64")], Qs], axis=0)
            Qs_ = Qs_[:-1] / beta
            log_prob_actions = Qs_ - pm.math.logsumexp(Qs_, axis=1)

            # Store probabilities
            probabilities.append(tt.exp(log_prob_actions[:, 1]))
        action_probabilities = tt.concatenate(probabilities, axis=0)
        return action_probabilities

    def fit(
        self,
        actions_set,
        rewards_set,
        alpha_params=(1, 1),
        gamma_params=(1, 1),
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
        Fit the softmax Expected SARSA agent to a given dataset.
        ========================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        gamma_params: discount rate beta prior parameters (default: (1,1))
        beta_params: temperature for softmax halfnormal prior parameters (default: (1,))
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
            alpha = pm.Beta("alpha", alpha_params[0], alpha_params[1])  # learning rate (beta distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, gamma, beta, actions_set, rewards_set
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


class ESARSALearner_esoftmax(ESARSALearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy.
        ====================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

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
        action_probabilities = np.exp(self.q_table[state, :] / self.beta)  # action probabilities
        action_probabilities /= np.sum(action_probabilities)  # normalize
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # make choice
        return action, action_probabilities

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma, beta, epsilon):
        """
        Vectorized update of Q-table using Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)
        beta: temperature for softmax (float)
        epsilon: exploration rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs_ = Qs / beta
        log_prob_actions = Qs_ - pm.math.logsumexp(Qs_)
        pQs = (1 - epsilon) * tt.exp(log_prob_actions) + epsilon / self.action_space_size
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * tt.dot(pQs, Qs) - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, gamma, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        gamma: discount rate (float)
        beta: temperature for softmax (float)
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
            Qs = tt.zeros((2), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma, beta, epsilon],
            )

            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((2), dtype="float64")], Qs], axis=0)
            Qs_ = Qs_[:-1] / beta
            log_prob_actions = Qs_ - pm.math.logsumexp(Qs_, axis=1)

            # Store probabilities
            probabilities.append((1 - epsilon) * tt.exp(log_prob_actions[:, 1]) + epsilon / self.action_space_size)
        action_probabilities = tt.concatenate(probabilities, axis=0)
        return action_probabilities

    def fit(
        self,
        actions_set,
        rewards_set,
        alpha_params=(1, 1),
        gamma_params=(1, 1),
        beta_params=(1,),
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
        Fit the epsilon-softmax Expected SARSA agent to a given dataset.
        ================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        gamma_params: discount rate beta prior parameters (default: (1,1))
        beta_params: temperature for softmax halfnormal prior parameters (default: (1,))
        epsilon_params: exploration rate beta prior parameters (default: (1,1))
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
            alpha = pm.Beta("alpha", alpha_params[0], alpha_params[1])  # learning rate (beta distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, gamma, beta, epsilon, actions_set, rewards_set
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


class ESARSALearner_acceptreject(ESARSALearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy.
        ====================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

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

        accept_probabilities = sigmoid(self.weight * self.q_table[state, :] + self.intercept)
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
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # make choice
        return action, action_probabilities

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma, weight, intercept):
        """
        Vectorized update of Q-table using Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)
        weight: q-value weight (float)
        intercept: q-value intercept (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """

        Qs_ = 1 / (1 + tt.exp(-(weight * Qs + intercept)))
        pQs = tt.stack(
            [
                Qs_[0] * (3 - Qs_[1]) / (3 * Qs_[0] + 3 * Qs_[1] - 2 * Qs_[0] * Qs_[1]),
                Qs_[1] * (3 - Qs_[0]) / (3 * Qs_[0] + 3 * Qs_[1] - 2 * Qs_[0] * Qs_[1]),
            ]
        )
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * tt.dot(pQs, Qs) - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, gamma, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        gamma: discount rate (float)
        weight: q-value weight (float)
        intercept: q-value intercept (float)
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
            Qs = tt.zeros((2), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma, weight, intercept],
            )

            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((2), dtype="float64")], Qs], axis=0)
            Qs_ = 1 / (1 + tt.exp(-(weight * Qs_[:-1] + intercept)))
            log_prob_action_1 = (
                tt.log(Qs_[:, 1])
                + tt.log(3 - Qs_[:, 0])
                - tt.log(3 * Qs_[:, 0] + 3 * Qs_[:, 1] - 2 * Qs_[:, 0] * Qs_[:, 1])
            )

            # Store probabilities
            probabilities.append(tt.exp(log_prob_action_1))
        action_probabilities = tt.concatenate(probabilities, axis=0)
        return action_probabilities

    def fit(
        self,
        actions_set,
        rewards_set,
        alpha_params=(1, 1),
        gamma_params=(1, 1),
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
        Fit the accept-reject Expected SARSA agent to a given dataset.
        ================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        gamma_params: discount rate beta prior parameters (default: (1,1))
        weight_params: q-value weight normal prior parameters (default: (0,1))
        intercept_params: q-value intercept normal prior parameters (default: (0,1))
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
            alpha = pm.Beta("alpha", alpha_params[0], alpha_params[1])  # learning rate (beta distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, gamma, weight, intercept, actions_set, rewards_set
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


class ESARSALearnerHM(FlYMazeAgent):
    def init_variables(self, alpha=[0.1, 0.1], init_q=[0.5, 0.5], gamma=0.5):
        """
        Initialize variables for the Expected SARSA agent.
        ==================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        gamma: discount rate for the agent (float) (default: 0.5)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        self.q_table[0, :] = init_q

    def init_recorder(self):
        """
        Initialize recorder for the Expected SARSA agent.
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Expected SARSA step in the environment.
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * E(Q[new_state,:]))
        ==============================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action, _ = self.make_choice(state)  # make a choice
        if self.history:
            self.action_history.append(action)
        if action == self.biased_action:
            self.bias += 1  # update bias estimate
        new_state, reward, done, info = self.env.step(action)  # get reward and new state
        if self.history:
            self.reward_history.append(reward)
        self.update_knowledge(state, action, new_state, reward)  # update knowledge

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy.
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # sample action
        action_probabilities = np.ones(self.action_space_size) * (
            1 / self.action_space_size
        )  # uniform action probabilities
        return action, action_probabilities

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Q-learning update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * E(Q[new_state,:]))
        ==============================================================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        _, action_probabilities = self.make_choice(new_state)  # get action probabilities
        self.q_table[state, action] = (1 - self.alpha[action]) * self.q_table[state, action] + self.alpha[action] * (
            reward + self.gamma * np.dot(self.q_table[new_state, :], action_probabilities)
        )  # update Q-table
        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Expected SARSA agent.
        """
        state = self.env.reset()
        done = False
        while not done:
            state, done = self.trial_step(state)

    def reset_variables(self):
        """
        Reset variables for the Expected SARSA agent.
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))

    def reset_recorder(self):
        """
        Reset the recorder for the Expected SARSA agent.
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class ESARSALearnerHM_acceptreject(ESARSALearnerHM):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy.
        ====================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

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

        accept_probabilities = sigmoid(self.weight * self.q_table[state, :] + self.intercept)
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
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # make choice
        return action, action_probabilities

    def vectorizedUpdate(self, action, reward, Qs, alphas, gamma, weight, intercept):
        """
        Vectorized update of Q-table using Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)
        weight: q-value weight (float)
        intercept: q-value intercept (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """

        Qs_ = 1 / (1 + tt.exp(-(weight * Qs + intercept)))
        pQs = tt.stack(
            [
                Qs_[0] * (3 - Qs_[1]) / (3 * Qs_[0] + 3 * Qs_[1] - 2 * Qs_[0] * Qs_[1]),
                Qs_[1] * (3 - Qs_[0]) / (3 * Qs_[0] + 3 * Qs_[1] - 2 * Qs_[0] * Qs_[1]),
            ]
        )
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alphas[action] * (reward + gamma * tt.dot(pQs, Qs) - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alphas, init_q, gamma, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        gamma: discount rate (float)
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)

        Returns:
        action_probabilities: action probabilities (Theano tensor)
        """
        probabilities = []
        for n, (actions_, rewards_) in enumerate(zip(actions_set, rewards_set)):  # for each session
            rewards = theano.shared(np.asarray(rewards_, dtype="int16"))
            actions = theano.shared(np.asarray(actions_, dtype="int16"))

            # Compute the Qs values
            Qs = init_q[n]
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards],
                outputs_info=[Qs],
                non_sequences=[alphas[n], gamma[n], weight, intercept],
            )

            # Compute the action probabilities
            Qs_ = tt.concatenate([[init_q[n]], Qs], axis=0)
            Qs_ = 1 / (1 + tt.exp(-(weight * Qs_[:-1] + intercept)))
            log_prob_action_1 = (
                tt.log(Qs_[:, 1])
                + tt.log(3 - Qs_[:, 0])
                - tt.log(3 * Qs_[:, 0] + 3 * Qs_[:, 1] - 2 * Qs_[:, 0] * Qs_[:, 1])
            )

            # Store probabilities
            probabilities.append(tt.exp(log_prob_action_1))
        action_probabilities = tt.concatenate(probabilities, axis=0)
        return action_probabilities

    def fit(
        self,
        actions_set,
        rewards_set,
        alpha_params=(1, 1),
        gamma_params=(1, 1),
        init_q_params=(0, 1),
        weight_params=(0, 1),
        intercept_params=(0, 1),
        niters=1000,
        ntune=1000,
        nchains=2,
        nparallel=1,
        target_accept=0.8,
        plot_trace=False,
        plot_posterior=False,
        return_predictives=False,
        print_summary=True,
    ):
        """
        Fit the accept-reject Expected SARSA agent to a given dataset.
        ================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        gamma_params: discount rate beta prior parameters (default: (1,1))
        weight_params: q-value weight normal prior parameters (default: (0,1))
        intercept_params: q-value intercept normal prior parameters (default: (0,1))
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
        n_flies = len(actions_set)
        n_actions = self.action_space_size

        with pm.Model() as model:  # define model
            # learning rate hyperpriors
            alpha_a_mean = pm.Gamma("alpha_beta_a", alpha_params[0], alpha_params[1], shape=n_actions)
            alpha_b_mean = pm.Gamma("alpha_beta_b", alpha_params[0], alpha_params[1], shape=n_actions)
            # learning rate priors samples (for each individual flies and action)
            alphas = []
            for n in range(n_actions):
                alphas.append(pm.Beta("alpha_{}".format(n), alpha_a_mean[n], alpha_b_mean[n], shape=n_flies))
            # combine learning rates
            alphas = pm.Deterministic("alpha", tt.stack(alphas, axis=1))

            # inital q-value hyperpriors
            init_q_mean = pm.Normal("init_q_mean", init_q_params[0], init_q_params[1], shape=n_actions)
            init_q_std = pm.Exponential("init_q_std", init_q_params[1], shape=n_actions)
            # initial q-value prior samples (for each individual flies and action)
            init_q = []
            for n in range(n_actions):
                init_q.append(pm.Normal("init_q_{}".format(n), init_q_mean[n], init_q_std[n], shape=n_flies))
            # combine initial q-values
            init_q = pm.Deterministic("init_q", tt.stack(init_q, axis=1))

            # discount rate hyperpriors
            gamma_a_mean = pm.Gamma("gamma_beta_a", gamma_params[0], gamma_params[1])
            gamma_b_mean = pm.Gamma("gamma_beta_b", gamma_params[0], gamma_params[1])
            # discount rate priors samples (for for each individual flies and action)
            gamma = pm.Beta("gamma", gamma_a_mean, gamma_b_mean, shape=n_flies)

            # policy priors
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alphas, init_q, gamma, weight, intercept, actions_set, rewards_set
            )  # action probabilities
            likelihood = pm.Bernoulli(
                "choice", p=action_probabilities, observed=np.concatenate([a for a in actions_set])
            )  # Bernoulli likelihood
            trace = pm.sample(
                tune=ntune,
                draws=niters,
                chains=nchains,
                cores=nparallel,
                return_inferencedata=True,
                target_accept=target_accept,
            )  # sample posterior

            if return_predictives:
                # sample posterior predictive
                posterior_predictive = pm.sample_posterior_predictive(trace)

        if plot_trace:  # plot trace
            az.plot_trace(trace)

        if plot_posterior:  # plot posterior
            az.plot_posterior(trace, hdi_prob=0.95)

        summary = az.summary(trace, hdi_prob=0.95)  # compute summary

        if print_summary:  # print summary
            print(summary)

        if return_predictives:  # return predictives
            return model, trace, posterior_predictive
        else:
            return model, trace
