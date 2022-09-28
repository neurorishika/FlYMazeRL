import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
import arviz as az

from flymazerl.agents.base import FlYMazeAgent

"""
This package contains classical Reinforcement Learning agents for the flymazerl environment.

    - IQLearner: an agent that learns using an Instantaneous Q-Learning algorithm
        - IQLearner_egreedy: uses e-greedy policy to select actions 
            Parameters: alpha, epsilon
        - IQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, beta
        - IQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, beta, epsilon
        - IQLearner_acceptreject: uses accept-rejection policy to select actions
            Parameters: alpha, weight, intercept

    - FQLearner: an agent that learns using a Forgetting Q-Learning algorithm
        - FQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, epsilon
        - FQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, beta
        - FQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, beta, epsilon
        - FQLearner_acceptreject: uses accept-rejection policy to select actions
            Parameters: alpha, weight, intercept

    -DFQLearner: an agent that learns using a Differential Forgetting Q-Learning algorithm
        - DFQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, kappa, epsilon
        - DFQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, kappa, beta
        - DFQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, kappa, beta, epsilon
    
    -DEQLearner: an agent that learns using a Differential Extinction Q-Learning algorithm
        - DEQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, tau, epsilon
        - DEQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, tau, beta
        - DEQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, tau, beta, epsilon

    - CQLearner: an agent that learns using a Long-Term Q-Learning algorithm
        - CQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, gamma, epsilon
        - CQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, gamma, beta
        - CQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, gamma, beta, epsilon

    - FCQLearner: an agent that learns using a Forgetting Long-Term Q-Learning algorithm
        - FCQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, gamma, kappa, epsilon
        - FCQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, gamma, kappa, beta
        - FCQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, gamma, kappa, beta, epsilon

    - DFCQLearner: an agent that learns using a Differential Forgetting Long-Term Q-Learning algorithm
        - DFCQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, gamma, kappa, epsilon
        - DFCQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, gamma, kappa, beta
        - DFCQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, gamma, kappa, beta, epsilon
    
    - DECQLearner: an agent that learns using a Differential Extinction Long-Term Q-Learning algorithm
        - DECQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, gamma, tau, epsilon
        - DECQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, gamma, tau, beta
        - DECQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, gamma, tau, beta, epsilon

    - SARSALearner: an agent that learns using a SARSA algorithm
        - SARSALearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, gamma, epsilon
        - SARSALearner_softmax: uses softmax policy to select actions
            Parameters: alpha, gamma, beta
        - SARSALearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, gamma, beta, epsilon

    - ESARSALearner: an agent that learns using an Expected SARSA algorithm
        - ESARSALearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, gamma, epsilon
        - ESARSALearner_softmax: uses softmax policy to select actions
            Parameters: alpha, gamma, beta
        - ESARSALearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, gamma, beta, epsilon

    - DQLearner: an agent that learns using a Double Q-Learning algorithm
        - DQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, gamma, epsilon
        - DQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, gamma, beta
        - DQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, gamma, beta, epsilon

    - HQLearner: an agent that learns using a Habitual Q-Learning algorithm
        - HQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon
        - HQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta
        - HQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, epsilon
    
    - HCQLearner: an agent that learns using a Habitual Long-Term Q-Learning algorithm
        - HCQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, gamma, epsilon
        - HCQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, gamma, beta
        - HCQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha_r, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, gamma, beta, epsilon

    - OSQLearner: an agent that learns using an Omission Sensitive Q-Learning algorithm
        - OSQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, theta, epsilon
        - OSQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, theta, beta
        - OSQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, theta, beta, epsilon
    
    - OSCQLearner: an agent that learns using an Omission Sensitive Long-Term Q-Learning algorithm
        - OSCQLearner_egreedy: uses e-greedy policy to select actions
            Parameters: alpha, theta, gamma, epsilon
        - OSCQLearner_softmax: uses softmax policy to select actions
            Parameters: alpha, theta, gamma, beta
        - OSCQLearner_esoftmax: uses epsilon-softmax policy to select actions
            Parameters: alpha, theta, gamma, beta, epsilon
"""


class RewardLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1):
        """
        Initialize variables for the Reward Learning agent
        =============================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        """
        self.alpha = alpha  # learning rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Reward Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Reward Learning step in the environment
        - Make a choice based on the current state
        - Get reward and new state
        - Update according to the equation:
            Q[state, action] = Q[state, action] + alpha * reward
        ======================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Reward learning update rule:
            Q[state, action] = Q[state, action] + alpha * reward
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * reward  # update Q-table

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Reward Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Reward Learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Reward Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class RewardLearner_egreedy(RewardLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Reward Learning Algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        return Qs

    def vectorizedActionProbabilities(self, alpha, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the epsilon-greedy Reward Learning agent to a given dataset
        =======================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, epsilon, actions_set, rewards_set
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


class RewardLearner_softmax(RewardLearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy
        ===========================================

        Parameters:
        beta: temperature for softmax (default: 0.2)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Reward learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        return Qs

    def vectorizedActionProbabilities(self, alpha, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the softmax Reward Learning agent to a given dataset
        ================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, beta, actions_set, rewards_set
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


class RewardLearner_esoftmax(RewardLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Reward learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        return Qs

    def vectorizedActionProbabilities(self, alpha, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the epsilon-softmax Reward Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, beta, epsilon, actions_set, rewards_set
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


class RewardLearner_acceptreject(RewardLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Reward learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        return Qs

    def vectorizedActionProbabilities(self, alpha, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the accept-reject Reward Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, weight, intercept, actions_set, rewards_set
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


class ForgettingRewardLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1):
        """
        Initialize variables for the Forgetting Reward Learning agent
        =============================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        """
        self.alpha = alpha  # learning rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Reward Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Forgetting Reward Learning step in the environment
        - Make a choice based on the current state
        - Get reward and new state
        - Update according to the equation:
            Q[state, action] = Q[state, action] + alpha * reward
            Q[state, other actions] = (1-alpha)*Q[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Reward learning update rule:
            Q[state, action] = Q[state, action] + alpha * reward
            Q[state, other actions] = (1-alpha)*Q[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.q_table[state, :action] = (1 - self.alpha) * self.q_table[state, :action]
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * reward  # update Q-table
        self.q_table[new_state, action + 1 :] = (1 - self.alpha) * self.q_table[new_state, action + 1 :]

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Reward Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Reward Learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Reward Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class ForgettingRewardLearner_egreedy(ForgettingRewardLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Reward Learning Algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - alpha) * Qs[:action])
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - alpha) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the epsilon-greedy Reward Learning agent to a given dataset
        =======================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, epsilon, actions_set, rewards_set
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


class ForgettingRewardLearner_softmax(ForgettingRewardLearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy
        ===========================================

        Parameters:
        beta: temperature for softmax (default: 0.2)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Reward learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - alpha) * Qs[:action])
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - alpha) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the softmax Reward Learning agent to a given dataset
        ================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, beta, actions_set, rewards_set
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


class ForgettingRewardLearner_esoftmax(ForgettingRewardLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Reward learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - alpha) * Qs[:action])
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - alpha) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the epsilon-softmax Reward Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, beta, epsilon, actions_set, rewards_set
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


class ForgettingRewardLearner_acceptreject(ForgettingRewardLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Reward learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - alpha) * Qs[:action])
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - alpha) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the accept-reject Reward Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, weight, intercept, actions_set, rewards_set
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


class IQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1):
        """
        Initialize variables for the Instantaneous Q-Learning agent
        =============================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        """
        self.alpha = alpha  # learning rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Instantaneous Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Instantaneous Q-Learning step in the environment
        - Make a choice based on the current state
        - Get reward and new state
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
        ======================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Instantaneous Q-learning update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[
            state, action
        ] + self.alpha * reward  # update Q-table

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Instantaneous Q-Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Instantaneous Q-Learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Instantaneous Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class IQLearner_egreedy(IQLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Instantaneous Q-Learning Algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the epsilon-greedy Instantaneous QLearning agent to a given dataset
        =======================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, epsilon, actions_set, rewards_set
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


class IQLearner_softmax(IQLearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy
        ===========================================

        Parameters:
        beta: temperature for softmax (default: 0.2)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Instantaneous Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the softmax Instantaneous QLearning agent to a given dataset
        ================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, beta, actions_set, rewards_set
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


class IQLearner_esoftmax(IQLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Instantaneous Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the epsilon-softmax Instantaneous Q-Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, beta, epsilon, actions_set, rewards_set
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


class IQLearner_acceptreject(IQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using Instantaneous Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the accept-reject Instantaneous Q-Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
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
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, weight, intercept, actions_set, rewards_set
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


class FQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1):
        """
        Initialize variables for the Forgetting Q-Learning agent
        ========================================================

        Parameters:
        alpha: learning and forgetting rate for the agent (float) (default: 0.1)
        """
        self.alpha = alpha  # learning rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Forgetting Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Forgetting Q-Learning step in the environment
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
            and
            Q[state, other actions] = (1 - alpha) * Q[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Forgetting Q-learning update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
            and
            Q[state, other actions] = (1 - alpha) * Q[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.q_table[state, :] = (1 - self.alpha) * self.q_table[state, :]
        self.q_table[state, action] += self.alpha * reward  # update Q-table

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Forgetting Q-Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Forgetting Q-Learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Forgetting Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class FQLearner_egreedy(FQLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using the Forgetting Q-Learning Algorithm
        ======================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning and forgetting rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        return Qs

    def vectorizedActionProbabilities(self, alpha, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning and forgetting rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the epsilon-greedy Forgetting QLearning agent to a given dataset
        ====================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning and forgetting rate beta prior parameters (default: (1,1))
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
            alpha = pm.Beta(
                "alpha", alpha_params[0], alpha_params[1]
            )  # learning and forgetting rate (beta distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, epsilon, actions_set, rewards_set
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


class FQLearner_softmax(FQLearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy
        ===========================================

        Parameters:
        beta: temperature for softmax (default: 0.2)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using the Forgetting Q-learning algorithm
        ======================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning and forgetting rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        return Qs

    def vectorizedActionProbabilities(self, alpha, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning and forgetting rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the softmax Forgetting QLearning agent to a given dataset
        =============================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning and forgetting rate beta prior parameters (default: (1,1))
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
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, beta, actions_set, rewards_set
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


class FQLearner_esoftmax(FQLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using the Forgetting Q-learning algorithm
        ======================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning and forgetting rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        return Qs

    def vectorizedActionProbabilities(self, alpha, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning and forgetting rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the epsilon-softmax Forgetting Q-Learning agent to a given dataset
        ======================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning and forgetting rate beta prior parameters (default: (1,1))
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
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, beta, epsilon, actions_set, rewards_set
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


class FQLearner_acceptreject(FQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha):
        """
        Vectorized update of Q-table using the Forgetting Q-learning algorithm
        ======================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning and forgetting rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * reward)
        return Qs

    def vectorizedActionProbabilities(self, alpha, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning and forgetting rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
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
        Fit the accept-reject Forgetting Q-Learning agent to a given dataset
        ======================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning and forgetting rate beta prior parameters (default: (1,1))
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
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, weight, intercept, actions_set, rewards_set
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


class DFQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, kappa=0.1):
        """
        Initialize variables for the Differential Forgetting Q-Learning agent
        =====================================================================

        Parameters:
        alpha: learning and forgetting rate for the agent (float) (default: 0.1)
        """
        self.alpha = alpha  # learning rate
        self.kappa = kappa  # forgetting rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Differential Forgetting Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Differential Forgetting Q-Learning step in the environment
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
            and
            Q[state, other actions] = (1 - kappa) * Q[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Differential Forgetting Q-learning update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward
            and
            Q[state, other actions] = (1 - kappa) * Q[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.q_table[state, :action] = (1 - self.kappa) * self.q_table[state, :action]
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * reward
        self.q_table[state, action + 1 :] = (1 - self.kappa) * self.q_table[state, action + 1 :]

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Differential Forgetting Q-Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Differential Forgetting Q-Learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class DFQLearner_egreedy(DFQLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa):
        """
        Vectorized update of Q-table using the Differential Forgetting Q-Learning Algorithm
        ===================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * reward)
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, kappa]
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
        kappa_params=(1, 1),
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
        Fit the epsilon-greedy Differential Forgetting QLearning agent to a given dataset
        =================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, epsilon, actions_set, rewards_set
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


class DFQLearner_softmax(DFQLearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy
        ===========================================

        Parameters:
        beta: temperature for softmax (default: 0.2)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa):
        """
        Vectorized update of Q-table using the Differential Forgetting Q-learning algorithm
        ===================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * reward)
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, kappa]
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
        kappa_params=(1, 1),
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
        Fit the softmax Differential Forgetting QLearning agent to a given dataset
        ==========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, beta, actions_set, rewards_set
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


class DFQLearner_esoftmax(DFQLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        action_probabilities: action probabilities (numpy array)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa):
        """
        Vectorized update of Q-table using the Differential Forgetting Q-learning algorithm
        ===================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * reward)
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, kappa]
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
        kappa_params=(1, 1),
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
        Fit the epsilon-softmax Differential Forgetting Q-Learning agent to a given dataset
        ===================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, beta, epsilon, actions_set, rewards_set
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


class DFQLearner_acceptreject(DFQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa):
        """
        Vectorized update of Q-table using the Differential Forgetting Q-learning algorithm
        ===================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * reward)
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, kappa]
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
        kappa_params=(1, 1),
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
        Fit the accept-reject Differential Forgetting Q-Learning agent to a given dataset
        ===================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, weight, intercept, actions_set, rewards_set
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


class DEQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, tau=0.1):
        """
        Initialize variables for the Differential Extinction Q-Learning agent
        =====================================================================

        Parameters:
        alpha: learning and forgetting rate for the agent (float) (default: 0.1)
        """
        self.alpha = alpha  # learning rate
        self.tau = tau  # extinction rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Differential Extinction Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Differential Forgetting Q-Learning step in the environment
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = Q[state, action] + alpha * (reward - Q[state, action]) if rewarded
            and
            Q[state, action] = (1-tau) Q[state, action] if not rewarded
        ======================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Differential Extinction Q-Learning equation
            Q[state, action] = Q[state, action] + alpha * (reward - Q[state, action]) if rewarded
            and
            Q[state, action] = (1-tau) * Q[state, action] if not rewarded
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        if reward != 0:
            self.q_table[state, action] += self.alpha * (reward - self.q_table[state, action])
        else:
            self.q_table[state, action] *= 1 - self.tau

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Differential Extinction Q-Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Differential Extinction Q-Learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class DEQLearner_egreedy(DEQLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, tau):
        """
        Vectorized update of Q-table using the Differential Extinction Q-Learning equation
        ===================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        tau: extinction rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(
            Qs[action], (1 - tau * tt.eq(reward, 0) - alpha * tt.neq(reward, 0)) * Qs[action] + alpha * reward
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, tau, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        tau: extinction rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, tau]
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
        tau_params=(1, 1),
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
        Fit the epsilon-greedy Differential Extinction QLearning agent to a given dataset
        =================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        tau_params: extinction rate beta prior parameters (default: (1,1))
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
            tau = pm.Beta("tau", tau_params[0], tau_params[1])  # forgetting rate (beta distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, tau, epsilon, actions_set, rewards_set
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


class DEQLearner_softmax(DEQLearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy
        ===========================================

        Parameters:
        beta: temperature for softmax (default: 0.2)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, tau):
        """
        Vectorized update of Q-table using the Differential Extinction Q-learning algorithm
        ===================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        tau: extinction rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(
            Qs[action], (1 - tau * tt.eq(reward, 0) - alpha * tt.neq(reward, 0)) * Qs[action] + alpha * reward
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, tau, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        tau: extinction rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, tau]
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
        tau_params=(1, 1),
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
        Fit the softmax Differential Extinction Q-learning model to a given agent
        ==========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        tau_params: extinction rate beta prior parameters (default: (1,1))
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
            tau = pm.Beta("tau", tau_params[0], tau_params[1])  # extinction rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, tau, beta, actions_set, rewards_set
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


class DEQLearner_esoftmax(DEQLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        action_probabilities: action probabilities (numpy array)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, tau):
        """
        Vectorized update of Q-table using the Differential Extinction Q-learning algorithm
        ===================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        tau: extinction rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(
            Qs[action], (1 - tau * tt.eq(reward, 0) - alpha * tt.neq(reward, 0)) * Qs[action] + alpha * reward
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, tau, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        tau: extinction rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, tau]
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
        tau_params=(1, 1),
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
        Fit the epsilon-softmax Differential Extinction Q-Learning agent to a given dataset
        ===================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        tau_params: extinction rate beta prior parameters (default: (1,1))
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
            tau = pm.Beta("tau", tau_params[0], tau_params[1])  # extinction rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, tau, beta, epsilon, actions_set, rewards_set
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


class DEQLearner_acceptreject(DEQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, tau):
        """
        Vectorized update of Q-table using the Differential Extinction Q-learning algorithm
        ===================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        tau: extinction rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(
            Qs[action], (1 - tau * tt.eq(reward, 0) - alpha * tt.neq(reward, 0)) * Qs[action] + alpha * reward
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, tau, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        tau: extinction rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, tau]
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
        tau_params=(1, 1),
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
        Fit the accept-reject Differential Extinction Q-Learning agent to a given dataset
        ===================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        tau_params: extinction rate beta prior parameters (default: (1,1))
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
            tau = pm.Beta("tau", tau_params[0], tau_params[1])  # extinction rate (beta distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, tau, weight, intercept, actions_set, rewards_set
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


class HQLearner(FlYMazeAgent):
    def init_variables(self, alpha_r=0.1, alpha_h=0.1, weight_r=0.1, weight_h=0.1, weight_b=0.0):
        """
        Initialize variables for the Habitual Q-Learning agent
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
        Initialize recorder for the Habitual Q-Learning agent
        """
        self.r_history = []
        self.h_history = []
        self.w_history = []
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Habitual Q-Learning step in the environment
        - Make a choice based on the current state
        - Get reward and new state
        - Update according to the equation:
            R[state, action] = (1 - alpha_r) * R[state, action] + alpha_r * reward
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
        Update value-table of the agent according to the Habitual Q-Learning equation:
            R[state, action] = (1 - alpha_r) * R[state, action] + alpha_r * reward
            H[state, action] = (1- alpha_h) * H[state, action] + alpha_h * habit
            H[state, other actions] = (1- alpha_h) * H[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.r_table[state, action] = (1 - self.alpha_r) * self.r_table[
            state, action
        ] + self.alpha_r * reward  # update reward table

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
        Describe a single episode of the Habitual Q-Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Habitual Q-Learning agent
        """
        self.r_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset reward table
        self.h_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset habit table

    def reset_recorder(self):
        """
        Reset the recorder for the Habitual Q-Learning agent
        """
        self.r_history = []
        self.h_history = []
        self.w_history = []
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class HQLearner_egreedy(HQLearner):
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
        Vectorized update of value-table using Habitual Q-learning Algorithm
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
        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * (reward - tables[action]))
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
        Fit the epsilon-greedy Habitual Q-Learning model
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


class HQLearner_softmax(HQLearner):
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
        Vectorized update of value-table using Habitual Q-learning Algorithm
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
        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * (reward - tables[action]))
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
        Fit the softmax Habitual Q-Learning model
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


class HQLearner_esoftmax(HQLearner):
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
        Vectorized update of Q-table using Habitual Q-Learning model
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

        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * (reward - tables[action]))
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
        Fit the epsilon-softmax Habitual Q-learning model
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


class HQLearner_acceptreject(HQLearner):
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
        Vectorized update of Q-table using Habitual Q-Learning model
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

        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * (reward - tables[action]))
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
        Fit the accept-reject Habitual Q-learning model
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


class OSQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, theta=-1):
        """
        Initialize variables for the Omission Sensitive Q-learning agent
        ================================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        theta: reward omission sensitivity (float) (default: -1)
        """
        self.alpha = alpha  # learning rate
        self.theta = theta  # reward omission sensitivity
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Omission Sensitive Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Omission Sensitive Q-Learning step in the environment
        - Make a choice based on the current state
        - Get reward and new state
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward, if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * theta, if not rewarded
        ======================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Omission Sensitive Q-learning equation
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward, if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * theta, if not rewarded
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.q_table[state, action] = (
            (1 - self.alpha) * self.q_table[state, action]
            + self.alpha * reward
            + self.alpha * self.theta * (reward == 0)
        )  # update Q-table

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Omission Sensitive Q-learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Omission Sensitive Q-learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Omision Sensitive Q-learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class OSQLearner_egreedy(OSQLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta):
        """
        Vectorized update of Q-table using Omission Sensitive Q-Learning Algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: reward omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, theta]
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
        theta_params=(0, 1),
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
        Fit the epsilon-greedy Omission Sensitive QLearning agent to a given dataset
        =======================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, epsilon, actions_set, rewards_set
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


class OSQLearner_softmax(OSQLearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy
        ===========================================

        Parameters:
        beta: temperature for softmax (default: 0.2)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta):
        """
        Vectorized update of Q-table using Omission Sensitive Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        theta: omission sensitivity (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, theta]
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
        theta_params=(0, 1),
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
        Fit the softmax Omission Sensitive QLearning agent to a given dataset
        ================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, beta, actions_set, rewards_set
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


class OSQLearner_esoftmax(OSQLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta):
        """
        Vectorized update of Q-table using Omission Sensitive Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        theta: omission sensitivity (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, theta]
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
        theta_params=(0, 1),
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
        Fit the epsilon-softmax Omission Sensitive Q-Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, beta, epsilon, actions_set, rewards_set
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


class OSQLearner_acceptreject(OSQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta):
        """
        Vectorized update of Q-table using Omission Sensitive Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        theta: omission sensitivity (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, theta]
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
        theta_params=(0, 1),
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
        Fit the accept-reject Omission Sensitive Q-Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, weight, intercept, actions_set, rewards_set
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


class FOSQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, theta=-1):
        """
        Initialize variables for the Forgetting Omission Sensitive Q-learning agent
        ================================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        theta: reward omission sensitivity (float) (default: -1)
        """
        self.alpha = alpha  # learning rate
        self.theta = theta  # reward omission sensitivity
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Omission Sensitive Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Omission Sensitive Q-Learning step in the environment
        - Make a choice based on the current state
        - Get reward and new state
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward, if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * theta, if not rewarded
            Q[state, other_action] = (1 - alpha) * Q[state, other_action]
        ======================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Forgetting Omission Sensitive Q-learning equation
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward, if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * theta, if not rewarded
            Q[state, other_action] = (1 - alpha) * Q[state, other_action]
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.q_table[state, :] = (1 - self.alpha) * self.q_table[state, :]
        self.q_table[state, action] += self.alpha * (reward + self.theta * (reward == 0))  # update Q-table

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Forgetting Omission Sensitive Q-learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Forgetting Omission Sensitive Q-learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Forgetting Omission Sensitive Q-learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class FOSQLearner_egreedy(FOSQLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta):
        """
        Vectorized update of Q-table using Forgetting Omission Sensitive Q-Learning Algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: reward omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0)))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, theta]
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
        theta_params=(0, 1),
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
        Fit the epsilon-greedy Forgetting Omission Sensitive QLearning agent to a given dataset
        =======================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, epsilon, actions_set, rewards_set
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


class FOSQLearner_softmax(FOSQLearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy
        ===========================================

        Parameters:
        beta: temperature for softmax (default: 0.2)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta):
        """
        Vectorized update of Q-table using Forgetting Omission Sensitive Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0)))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        theta: omission sensitivity (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, theta]
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
        theta_params=(0, 1),
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
        Fit the softmax Forgetting Omission Sensitive QLearning agent to a given dataset
        ================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, beta, actions_set, rewards_set
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


class FOSQLearner_esoftmax(FOSQLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta):
        """
        Vectorized update of Q-table using Forgetting Omission Sensitive Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0)))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        theta: omission sensitivity (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, theta]
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
        theta_params=(0, 1),
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
        Fit the epsilon-softmax Forgetting Omission Sensitive Q-Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, beta, epsilon, actions_set, rewards_set
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


class FOSQLearner_acceptreject(FOSQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta):
        """
        Vectorized update of Q-table using Forgetting Omission Sensitive Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0)))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        theta: omission sensitivity (float)
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, theta]
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
        theta_params=(0, 1),
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
        Fit the accept-reject Forgetting Omission Sensitive Q-Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, weight, intercept, actions_set, rewards_set
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


class DFOSQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, kappa=0.1, theta=-1):
        """
        Initialize variables for the Differential Forgetting Omission Sensitive Q-learning agent
        ================================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        kappa: forgetting rate for the agent (float) (default: 0.1)
        theta: reward omission sensitivity (float) (default: -1)
        """
        self.alpha = alpha  # learning rate
        self.kappa = kappa  # forgetting rate
        self.theta = theta  # reward omission sensitivity
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Differential Omission Sensitive Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Differential Omission Sensitive Q-Learning step in the environment
        - Make a choice based on the current state
        - Get reward and new state
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward, if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * theta, if not rewarded
            Q[state, other_action] = (1 - kappa) * Q[state, other_action]
        ======================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Differential Forgetting Omission Sensitive Q-learning equation
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * reward, if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * theta, if not rewarded
            Q[state, other_action] = (1 - kappa) * Q[state, other_action]
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.q_table[state, :action] = (1 - self.kappa) * self.q_table[state, :action]
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (
            reward + self.theta * (reward == 0)
        )  # update Q-table
        self.q_table[new_state, action + 1 :] = (1 - self.kappa) * self.q_table[new_state, action + 1 :]

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Differential Forgetting Omission Sensitive Q-learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Differential Forgetting Omission Sensitive Q-learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Differential Forgetting Omission Sensitive Q-learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class DFOSQLearner_egreedy(DFOSQLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, theta):
        """
        Vectorized update of Q-table using Differential Forgetting Omission Sensitive Q-Learning Algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        theta: reward omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + theta * tt.eq(reward, 0)))
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, theta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                non_sequences=[alpha, kappa, theta],
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
        kappa_params=(1, 1),
        theta_params=(0, 1),
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
        Fit the epsilon-greedy Differential Forgetting Omission Sensitive QLearning agent to a given dataset
        =======================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, theta, epsilon, actions_set, rewards_set
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


class DFOSQLearner_softmax(DFOSQLearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy
        ===========================================

        Parameters:
        beta: temperature for softmax (default: 0.2)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and softmax policy
        ===========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, theta):
        """
        Vectorized update of Q-table using Differential Forgetting Omission Sensitive Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        theta: omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + theta * tt.eq(reward, 0)))
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, theta, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        theta: omission sensitivity (float)
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
                non_sequences=[alpha, kappa, theta],
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
        kappa_params=(1, 1),
        theta_params=(0, 1),
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
        Fit the softmax Differential Forgetting Omission Sensitive QLearning agent to a given dataset
        ================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, theta, beta, actions_set, rewards_set
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


class DFOSQLearner_esoftmax(DFOSQLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, theta):
        """
        Vectorized update of Q-table using Differential Forgetting Omission Sensitive Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        theta: omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + theta * tt.eq(reward, 0)))
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, theta, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        theta: omission sensitivity (float)
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
                non_sequences=[alpha, kappa, theta],
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
        kappa_params=(1, 1),
        theta_params=(0, 1),
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
        Fit the epsilon-softmax Differential Forgetting Omission Sensitive Q-Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, theta, beta, epsilon, actions_set, rewards_set
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


class DFOSQLearner_acceptreject(DFOSQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, theta):
        """
        Vectorized update of Q-table using Differential Forgetting Omission Sensitive Q-learning algorithm
        =====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        theta: omission sensitivity (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + theta * tt.eq(reward, 0)))
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, theta, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        theta: omission sensitivity (float)
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
                non_sequences=[alpha, kappa, theta],
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
        kappa_params=(1, 1),
        theta_params=(0, 1),
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
        Fit the accept-reject Differential Forgetting Omission Sensitive Q-Learning agent to a given dataset
        =========================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, theta, weight, intercept, actions_set, rewards_set
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


class CQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, gamma=0.5):
        """
        Initialize variables for the Long-Term Q-Learning agent
        =======================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        gamma: discount rate for the agent (float) (default: 0.5)
        """
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Long-Term Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Long-Term Q-Learning step in the environment
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:]))
        ==============================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Long-Term Q-learning update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:]))
        ==============================================================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (
            reward + self.gamma * np.max(self.q_table[new_state, :])
        )  # update Q-table

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Long-Term Q-Learning agent.
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Long-Term Q-Learning agent.
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Long-Term Q-Learning agent.
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class CQLearner_egreedy(CQLearner):
    def init_policy_variables(self, epsilon=0.2):
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
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Q-Learning Algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * tt.max(Qs) - Qs[action]))
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, gamma]
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
        Fit the epsilon-greedy QLearning agent to a given dataset.
        ==========================================================

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


class CQLearner_softmax(CQLearner):
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
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * tt.max(Qs) - Qs[action]))
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, gamma]
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
        Fit the softmax QLearning agent to a given dataset.
        ===================================================

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


class CQLearner_esoftmax(CQLearner):
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
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * tt.max(Qs) - Qs[action]))
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, gamma]
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
        Fit the epsilon-softmax agent to a given dataset.
        =================================================

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


class CQLearner_acceptreject(CQLearner):
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * tt.max(Qs) - Qs[action]))
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, gamma]
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
        Fit the accept-reject agent to a given dataset.
        =================================================

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


class FCQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, gamma=0.5):
        """
        Initialize variables for the Forgetting Long-Term Q-Learning agent
        ==================================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        gamma: discount rate for the agent (float) (default: 0.5)
        """
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Forgetting Long-Term Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Forgetting Long-Term Q-Learning step in the environment
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:]))
            and
            Q[state, other_action] = (1 - alpha) * Q[state, other_action]
        ==============================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Forgetting Long-Term Q-learning update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:]))
            and
            Q[state, other_action] = (1 - alpha) * Q[state, other_action]
        ==============================================================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        next_best = np.max(self.q_table[new_state, :])  # best value in the new state
        self.q_table[state, :] = (1 - self.alpha) * self.q_table[state, :]
        self.q_table[state, action] += self.alpha * (reward + self.gamma * next_best)

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Forgetting Long-Term Q-Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Forgetting Long-Term Q-Learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Forgetting Long-Term Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class FCQLearner_egreedy(FCQLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Forgetting Long-Term Q-Learning Algorithm
        ============================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * next_best))
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, gamma]
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
        Fit the epsilon-greedy Forgetting Long-Term QLearning agent to a given dataset
        ==============================================================================

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


class FCQLearner_softmax(FCQLearner):
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
        Make a choice based on the current state and softmax policy
        ============================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Forgetting Long-Term Q-learning algorithm
        ============================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * next_best))
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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, gamma]
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
        Fit the softmax Forgetting Long-Term QLearning agent to a given dataset
        =======================================================================

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


class FCQLearner_esoftmax(FCQLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Forgetting Long-Term Q-learning algorithm
        ============================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * next_best))
        return Qs

    def vectorizedActionProbabilities(self, alpha, gamma, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, gamma]
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
        Fit the epsilon-softmax Forgetting Long-Term Q Learning agent to a given dataset
        ================================================================================

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


class FCQLearner_acceptreject(FCQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Forgetting Long-Term Q-learning algorithm
        ============================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * next_best))
        return Qs

    def vectorizedActionProbabilities(self, alpha, gamma, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
                fn=self.vectorizedUpdate, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha, gamma]
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
        Fit the accept-reject Forgetting Long-Term Q Learning agent to a given dataset
        ================================================================================

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


class DFCQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, kappa=0.1, gamma=0.5):
        """
        Initialize variables for the Differential Forgetting Long-Term Q-Learning agent
        ===============================================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        kappa: forgetting rate for the agent (float) (default: 0.1)
        gamma: discount rate for the agent (float) (default: 0.5)
        """
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.kappa = kappa  # forgetting rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Differential Forgetting Long-Term Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Differential Forgetting Long-Term Q-Learning step in the environment
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - kappa) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:]))
            and
            Q[state, other_action] = (1 - kappa) * Q[state, other_action]
        ==============================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Differential Forgetting Long-Term Q-learning update rule:
            Q[state, action] = (1 - kappa) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:]))
            and
            Q[state, other_action] = (1 - kappa) * Q[state, other_action]
        ==============================================================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        next_best = np.max(self.q_table[new_state, :])
        self.q_table[state, :action] = (1 - self.kappa) * self.q_table[state, :action]
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (
            reward + self.gamma * next_best
        )
        self.q_table[state, action + 1 :] = (1 - self.kappa) * self.q_table[state, action + 1 :]

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Differential Forgetting Long-Term Q-Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Differential Forgetting Long-Term Q-Learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Differential Forgetting Long-Term Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class DFCQLearner_egreedy(DFCQLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, gamma):
        """
        Vectorized update of Q-table using Forgetting Long-Term Q-Learning Algorithm
        ============================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + gamma * next_best))
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, gamma, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                non_sequences=[alpha, kappa, gamma],
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
        kappa_params=(1, 1),
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
        Fit the epsilon-greedy Differential Forgetting Long-Term QLearning agent to a given dataset
        ===========================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, gamma, epsilon, actions_set, rewards_set
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


class DFCQLearner_softmax(DFCQLearner):
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
        Make a choice based on the current state and softmax policy
        ============================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, gamma):
        """
        Vectorized update of Q-table using Differential Forgetting Long-Term Q-learning algorithm
        =========================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + gamma * next_best))
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, gamma, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                non_sequences=[alpha, kappa, gamma],
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
        kappa_params=(1, 1),
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
        Fit the softmax Differential Forgetting Long-Term QLearning agent to a given dataset
        ====================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, gamma, beta, actions_set, rewards_set
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


class DFCQLearner_esoftmax(DFCQLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, gamma):
        """
        Vectorized update of Q-table using Differential Forgetting Long-Term Q-learning algorithm
        =========================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + gamma * next_best))
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, gamma, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                non_sequences=[alpha, kappa, gamma],
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
        kappa_params=(1, 1),
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
        Fit the epsilon-softmax Differential Forgetting Long-Term Q Learning agent to a given dataset
        =============================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, gamma, beta, epsilon, actions_set, rewards_set
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


class DFCQLearner_acceptreject(DFCQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, gamma):
        """
        Vectorized update of Q-table using Differential Forgetting Long-Term Q-learning algorithm
        =========================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + gamma * next_best))
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, gamma, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                non_sequences=[alpha, kappa, gamma],
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
        kappa_params=(1, 1),
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
        Fit the accept-reject Differential Forgetting Long-Term Q Learning agent to a given dataset
        =============================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, gamma, weight, intercept, actions_set, rewards_set
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


class DECQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, tau=0.1, gamma=0.5):
        """
        Initialize variables for the Differential Extinction Long-Term Q-Learning agent
        ===============================================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        tau: extinction rate for the agent (float) (default: 0.1)
        gamma: discount rate for the agent (float) (default: 0.5)
        """
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.tau = tau  # extinction rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Differential Extinction Long-Term Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Differential Extinction Long-Term Q-Learning step in the environment
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - tau) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:]))
            and
            Q[state, other_action] = (1 - tau) * Q[state, other_action]
        ==============================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Differential Extinction Long-Term Q-learning update rule:
            Q[state, action] = (1 - tau) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:]))
            and
            Q[state, other_action] = (1 - tau) * Q[state, other_action]
        ==============================================================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        next_best = np.max(self.q_table[new_state, :])
        self.q_table[state, :] = (1 - self.tau) * self.q_table[state, :]
        self.q_table[state, action] += self.alpha * (reward + self.gamma * next_best)

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Differential Extinction Long-Term Q-Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Differential Extinction Long-Term Q-Learning agent
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Differential Extinction Long-Term Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class DECQLearner_egreedy(DECQLearner):
    def init_policy_variables(self, epsilon=0.2):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate (default: 0.2)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-greedy policy
        ==================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, tau, gamma):
        """
        Vectorized update of Q-table using Extinction Long-Term Q-Learning Algorithm
        ============================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        tau: extinction rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(
            Qs[action],
            (1 - tau * tt.eq(reward, 0) - alpha * tt.neq(reward, 0)) * Qs[action]
            + alpha * (reward + gamma * next_best),
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, tau, gamma, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        tau: extinction rate (float)
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
                non_sequences=[alpha, tau, gamma],
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
        tau_params=(1, 1),
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
        Fit the epsilon-greedy Differential Extinction Long-Term QLearning agent to a given dataset
        ===========================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        tau_params: extinction rate beta prior parameters (default: (1,1))
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
            tau = pm.Beta("tau", tau_params[0], tau_params[1])  # extinction rate (beta distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, tau, gamma, epsilon, actions_set, rewards_set
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


class DECQLearner_softmax(DECQLearner):
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
        Make a choice based on the current state and softmax policy
        ============================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, tau, gamma):
        """
        Vectorized update of Q-table using Differential Extinction Long-Term Q-learning algorithm
        =========================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        tau: extinction rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(
            Qs[action],
            (1 - tau * tt.eq(reward, 0) - alpha * tt.neq(reward, 0)) * Qs[action]
            + alpha * (reward + gamma * next_best),
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, tau, gamma, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        tau: extinction rate (float)
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
                non_sequences=[alpha, tau, gamma],
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
        tau_params=(1, 1),
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
        Fit the softmax Differential Extinction Long-Term QLearning agent to a given dataset
        ====================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        tau_params: extinction rate beta prior parameters (default: (1,1))
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
            tau = pm.Beta("tau", tau_params[0], tau_params[1])  # extinction rate (beta distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, tau, gamma, beta, actions_set, rewards_set
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


class DECQLearner_esoftmax(DECQLearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy
        ===================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state and epsilon-softmax policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, tau, gamma):
        """
        Vectorized update of Q-table using Differential Extinction Long-Term Q-learning algorithm
        =========================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        tau: extinction rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(
            Qs[action],
            (1 - tau * tt.eq(reward, 0) - alpha * tt.neq(reward, 0)) * Qs[action]
            + alpha * (reward + gamma * next_best),
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, tau, gamma, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        tau: extinction rate (float)
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
                non_sequences=[alpha, tau, gamma],
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
        tau_params=(1, 1),
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
        Fit the epsilon-softmax Differential Extinction Long-Term Q Learning agent to a given dataset
        =============================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        tau_params: extinction rate beta prior parameters (default: (1,1))
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
            tau = pm.Beta("tau", tau_params[0], tau_params[1])  # extinction rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, tau, gamma, beta, epsilon, actions_set, rewards_set
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


class DECQLearner_acceptreject(DECQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy
        ===================================================

        Parameters:
        weight: q-value weight (float)
        intercept: q-value intercept (float)
        """
        self.weight = weight
        self.intercept = intercept

    def make_choice(self, state):
        """
        Make a choice based on the current state and accept-reject policy
        ===================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, tau, gamma):
        """
        Vectorized update of Q-table using Differential Extinction Long-Term Q-learning algorithm
        =========================================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        tau: extinction rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(
            Qs[action],
            (1 - tau * tt.eq(reward, 0) - alpha * tt.neq(reward, 0)) * Qs[action]
            + alpha * (reward + gamma * next_best),
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, tau, gamma, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha: learning rate (float)
        tau: extinction rate (float)
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
                non_sequences=[alpha, tau, gamma],
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
        tau_params=(1, 1),
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
        Fit the accept-reject Differential Extinction Long-Term Q Learning agent to a given dataset
        =============================================================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        tau_params: extinction rate beta prior parameters (default: (1,1))
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
            tau = pm.Beta("tau", tau_params[0], tau_params[1])  # extinction rate (beta distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, tau, gamma, weight, intercept, actions_set, rewards_set
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


class HCQLearner(FlYMazeAgent):
    def init_variables(self, alpha_r=0.1, gamma=0.9, alpha_h=0.1, weight_r=0.1, weight_h=0.1, weight_b=0.0):
        """
        Initialize variables for the Habitual Q-Learning agent
        ======================================================

        Parameters:
        alpha_r: learning rate for reward value (float)
        gamma: discount factor (float)
        alpha_h: learning rate for habit value (float)
        weight_r: weight for reward value (float)
        weight_h: weight for habit value (float)
        weight_b: bias (float)
        """
        self.alpha_r = alpha_r  # Reward learning rate
        self.gamma = gamma  # Discount factor
        self.alpha_h = alpha_h  # Habit learning rate
        self.weight_r = weight_r  # Arbiter weight for reward
        self.weight_h = weight_h  # Arbiter weight for habits
        self.weight_b = weight_b  # Arbiter bias
        self.w = 0.5  # Controller weight
        self.r_table = np.zeros((self.state_space_size, self.action_space_size))  # Reward table
        self.h_table = np.zeros((self.state_space_size, self.action_space_size))  # Habit table

    def init_recorder(self):
        """
        Initialize recorder for the Habitual Q-Learning agent
        """
        self.r_history = []
        self.h_history = []
        self.w_history = []
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Habitual Q-Learning step in the environment
        - Make a choice based on the current state
        - Get reward and new state
        - Update according to the equation:
            R[state, action] = (1 - alpha_r) * R[state, action] + alpha_r * (reward + gamma * max(R[new_state]))
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
        Update value-table of the agent according to the Habitual Q-Learning equation:
            R[state, action] = (1 - alpha_r) * R[state, action] + alpha_r * (reward + gamma * max(R[new_state]))
            H[state, action] = (1- alpha_h) * H[state, action] + alpha_h * habit
            H[state, other actions] = (1- alpha_h) * H[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.r_table[state, action] = (1 - self.alpha_r) * self.r_table[state, action] + self.alpha_r * (
            reward + self.gamma * np.max(self.r_table[new_state])
        )  # update reward value

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
        Describe a single episode of the Habitual Q-Learning agent
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Habitual Q-Learning agent
        """
        self.r_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset reward table
        self.h_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset habit table

    def reset_recorder(self):
        """
        Reset the recorder for the Habitual Q-Learning agent
        """
        self.r_history = []
        self.h_history = []
        self.w_history = []
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class HCQLearner_egreedy(HCQLearner):
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
        self, action, reward, tables, alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon
    ):
        """
        Vectorized update of value-table using Habitual Q-learning Algorithm
        ====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: learning rate for reward value (Theano tensor)
        gamma: discount factor (float)
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
        tables = tt.set_subtensor(
            tables[action],
            (1 - alpha_r) * tables[action] + alpha_r * (reward + gamma * tt.max(tables[: self.action_space_size])),
        )
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
        self, alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon, actions_set, rewards_set
    ):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha_r: learning rate for reward value (Theano tensor)
        gamma: discount factor (Theano tensor)
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
                non_sequences=[alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon],
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
        gamma_params=(1, 1),
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
        Fit the epsilon-greedy Habitual Q-Learning model
        ================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_r_params: Reward learning rate beta prior parameters (default: (1,1))
        gamma_params: Discount factor beta prior parameters (default: (1,1))
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
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount factor
            alpha_h = pm.Beta("alpha_h", alpha_h_params[0], alpha_h_params[1])  # habit learning rate
            weight_r = pm.Normal("weight_r", weight_r_params[0], weight_r_params[1])  # reward weight
            weight_h = pm.Normal("weight_h", weight_h_params[0], weight_h_params[1])  # habit weight
            weight_b = pm.Normal("weight_b", weight_b_params[0], weight_b_params[1])  # bias
            theta_r = pm.Beta("theta_r", theta_r_params[0], theta_r_params[1])  # reward scaling
            theta_h = pm.Beta("theta_h", theta_h_params[0], theta_h_params[1])  # habit scaling
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha_r,
                gamma,
                alpha_h,
                weight_r,
                weight_h,
                weight_b,
                theta_r,
                theta_h,
                epsilon,
                actions_set,
                rewards_set,
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


class HCQLearner_softmax(HCQLearner):
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
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action, action_probabilities

    def vectorizedUpdate(
        self, action, reward, tables, alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta
    ):
        """
        Vectorized update of value-table using Habitual Q-learning Algorithm
        ====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: learning rate for reward value (Theano tensor)
        gamma: discount factor (Theano tensor)
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
        tables = tt.set_subtensor(
            tables[action],
            (1 - alpha_r) * tables[action] + alpha_r * (reward + gamma * tt.max(tables[: self.action_space_size])),
        )
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
        self, alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, actions_set, rewards_set
    ):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha_r: reward learning rate (Theano tensor)
        gamma: discount factor (Theano tensor)
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
                non_sequences=[alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta],
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
        gamma_params=(1, 1),
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
        Fit the softmax Habitual Q-Learning model
        =========================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_r_params: Reward learning rate beta prior parameters (default: (1,1))
        gamma_params: Discount factor beta prior parameters (default: (1,1))
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
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount factor
            alpha_h = pm.Beta("alpha_h", alpha_h_params[0], alpha_h_params[1])  # habit learning rate
            weight_r = pm.Normal("weight_r", weight_r_params[0], weight_r_params[1])  # reward weight
            weight_h = pm.Normal("weight_h", weight_h_params[0], weight_h_params[1])  # habit weight
            weight_b = pm.Normal("weight_b", weight_b_params[0], weight_b_params[1])  # bias
            theta_r = pm.Beta("theta_r", theta_r_params[0], theta_r_params[1])  # reward scaling
            theta_h = pm.Beta("theta_h", theta_h_params[0], theta_h_params[1])  # habit scaling
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax

            action_probabilities = self.vectorizedActionProbabilities(
                alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, actions_set, rewards_set
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


class HCQLearner_esoftmax(HCQLearner):
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
        self,
        action,
        reward,
        tables,
        alpha_r,
        gamma,
        alpha_h,
        weight_r,
        weight_h,
        weight_b,
        theta_r,
        theta_h,
        beta,
        epsilon,
    ):
        """
        Vectorized update of Q-table using Habitual Q-Learning model
        ============================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: reward learning rate (Theano tensor)
        gamma: discount factor (Theano tensor)
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

        tables = tt.set_subtensor(
            tables[action],
            (1 - alpha_r) * tables[action] + alpha_r * (reward + gamma * tt.max(tables[: self.action_space_size])),
        )
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
        self,
        alpha_r,
        gamma,
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
    ):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha_r: reward learning rate (float)
        gamma: discount factor (float)
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
                non_sequences=[alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, epsilon],
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
        gamma_params=(1, 1),
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
        Fit the epsilon-softmax Habitual Q-learning model
        =================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_r_params: Reward learning rate beta prior parameters (default: (1,1))
        gamma_params: Discount factor beta prior parameters (default: (1,1))
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
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount factor
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
                gamma,
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


class HCQLearner_acceptreject(HCQLearner):
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
        gamma,
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
        Vectorized update of Q-table using Habitual Q-Learning model
        ============================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: reward learning rate (Theano tensor)
        gamma: discount factor (Theano tensor)
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

        tables = tt.set_subtensor(
            tables[action],
            (1 - alpha_r) * tables[action] + alpha_r * (reward + gamma * tt.max(tables[: self.action_space_size])),
        )
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
        gamma,
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
        gamma: discount factor (float)
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
                non_sequences=[
                    alpha_r,
                    gamma,
                    alpha_h,
                    weight_r,
                    weight_h,
                    weight_b,
                    theta_r,
                    theta_h,
                    weight,
                    intercept,
                ],
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
        gamma_params=(1, 1),
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
        Fit the accept-reject Habitual Q-learning model
        =================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_r_params: Reward learning rate beta prior parameters (default: (1,1))
        gamma_params: Discount factor beta prior parameters (default: (1,1))
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
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount factor
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
                gamma,
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


class FHCQLearner(FlYMazeAgent):
    def init_variables(self, alpha_r=0.1, gamma=0.9, alpha_h=0.1, weight_r=0.1, weight_h=0.1, weight_b=0.0):
        """
        Initialize variables for the Forgetting Habitual Q-Learning agent
        ======================================================

        Parameters:
        alpha_r: learning rate for reward value (float)
        gamma: discount factor (float)
        alpha_h: learning rate for habit value (float)
        weight_r: weight for reward value (float)
        weight_h: weight for habit value (float)
        weight_b: bias (float)
        """
        self.alpha_r = alpha_r  # Reward learning rate
        self.gamma = gamma  # Discount factor
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
            R[state, action] = (1 - alpha_r) * R[state, action] + alpha_r * (reward + gamma * max(R[new_state]))
            R[state, other_action] = (1 - alpha_h) * R[state, other_action]
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
            R[state, action] = (1 - alpha_r) * R[state, action] + alpha_r * (reward + gamma * max(R[new_state]))
            R[state, other_action] = (1 - alpha_h) * R[state, other_action]
            H[state, action] = (1- alpha_h) * H[state, action] + alpha_h * habit
            H[state, other actions] = (1- alpha_h) * H[state, other actions]
        ======================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.r_table[state, :] = (1 - self.alpha_r) * self.r_table[state, :]
        self.r_table[state, action] += self.alpha_r * (
            reward + self.gamma * np.max(self.r_table[new_state])
        )  # update reward value

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


class FHCQLearner_egreedy(FHCQLearner):
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
        self, action, reward, tables, alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon
    ):
        """
        Vectorized update of value-table using Forgetting Habitual Q-learning Algorithm
        ====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: learning rate for reward value (Theano tensor)
        gamma: discount factor (float)
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
        next_best = tt.max(tables[: self.action_space_size])
        tables = tt.set_subtensor(tables[: self.action_space_size], (1 - alpha_r) * tables[: self.action_space_size],)
        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * (reward + gamma * next_best),)
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
        self, alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon, actions_set, rewards_set
    ):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha_r: learning rate for reward value (Theano tensor)
        gamma: discount factor (Theano tensor)
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
                non_sequences=[alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, epsilon],
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
        gamma_params=(1, 1),
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
        gamma_params: Discount factor beta prior parameters (default: (1,1))
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
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount factor
            alpha_h = pm.Beta("alpha_h", alpha_h_params[0], alpha_h_params[1])  # habit learning rate
            weight_r = pm.Normal("weight_r", weight_r_params[0], weight_r_params[1])  # reward weight
            weight_h = pm.Normal("weight_h", weight_h_params[0], weight_h_params[1])  # habit weight
            weight_b = pm.Normal("weight_b", weight_b_params[0], weight_b_params[1])  # bias
            theta_r = pm.Beta("theta_r", theta_r_params[0], theta_r_params[1])  # reward scaling
            theta_h = pm.Beta("theta_h", theta_h_params[0], theta_h_params[1])  # habit scaling
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha_r,
                gamma,
                alpha_h,
                weight_r,
                weight_h,
                weight_b,
                theta_r,
                theta_h,
                epsilon,
                actions_set,
                rewards_set,
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


class FHCQLearner_softmax(FHCQLearner):
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
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action, action_probabilities

    def vectorizedUpdate(
        self, action, reward, tables, alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta
    ):
        """
        Vectorized update of value-table using Forgetting Habitual Q-learning Algorithm
        ====================================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: learning rate for reward value (Theano tensor)
        gamma: discount factor (Theano tensor)
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
        next_best = tt.max(tables[: self.action_space_size])
        tables = tt.set_subtensor(tables[: self.action_space_size], (1 - alpha_r) * tables[: self.action_space_size],)
        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * (reward + gamma * next_best),)
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
        self, alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, actions_set, rewards_set
    ):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha_r: reward learning rate (Theano tensor)
        gamma: discount factor (Theano tensor)
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
                non_sequences=[alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta],
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
        gamma_params=(1, 1),
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
        gamma_params: Discount factor beta prior parameters (default: (1,1))
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
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount factor
            alpha_h = pm.Beta("alpha_h", alpha_h_params[0], alpha_h_params[1])  # habit learning rate
            weight_r = pm.Normal("weight_r", weight_r_params[0], weight_r_params[1])  # reward weight
            weight_h = pm.Normal("weight_h", weight_h_params[0], weight_h_params[1])  # habit weight
            weight_b = pm.Normal("weight_b", weight_b_params[0], weight_b_params[1])  # bias
            theta_r = pm.Beta("theta_r", theta_r_params[0], theta_r_params[1])  # reward scaling
            theta_h = pm.Beta("theta_h", theta_h_params[0], theta_h_params[1])  # habit scaling
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax

            action_probabilities = self.vectorizedActionProbabilities(
                alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, actions_set, rewards_set
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


class FHCQLearner_esoftmax(FHCQLearner):
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
        self,
        action,
        reward,
        tables,
        alpha_r,
        gamma,
        alpha_h,
        weight_r,
        weight_h,
        weight_b,
        theta_r,
        theta_h,
        beta,
        epsilon,
    ):
        """
        Vectorized update of Q-table using Forgetting Habitual Q-Learning model
        ============================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        tables: value tables (Theano tensor)
        alpha_r: reward learning rate (Theano tensor)
        gamma: discount factor (Theano tensor)
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

        next_best = tt.max(tables[: self.action_space_size])
        tables = tt.set_subtensor(tables[: self.action_space_size], (1 - alpha_r) * tables[: self.action_space_size],)
        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * (reward + gamma * next_best),)
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
        self,
        alpha_r,
        gamma,
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
    ):
        """
        Vectorized action probabilities for second alternative
        ======================================================

        Parameters:
        alpha_r: reward learning rate (float)
        gamma: discount factor (float)
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
                non_sequences=[alpha_r, gamma, alpha_h, weight_r, weight_h, weight_b, theta_r, theta_h, beta, epsilon],
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
        gamma_params=(1, 1),
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
        gamma_params: Discount factor beta prior parameters (default: (1,1))
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
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount factor
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
                gamma,
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


class FHCQLearner_acceptreject(FHCQLearner):
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
        gamma,
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
        gamma: discount factor (Theano tensor)
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

        next_best = tt.max(tables[: self.action_space_size])
        tables = tt.set_subtensor(tables[: self.action_space_size], (1 - alpha_r) * tables[: self.action_space_size],)
        tables = tt.set_subtensor(tables[action], tables[action] + alpha_r * (reward + gamma * next_best),)
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
        gamma,
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
        gamma: discount factor (float)
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
                non_sequences=[
                    alpha_r,
                    gamma,
                    alpha_h,
                    weight_r,
                    weight_h,
                    weight_b,
                    theta_r,
                    theta_h,
                    weight,
                    intercept,
                ],
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
        gamma_params=(1, 1),
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
        gamma_params: Discount factor beta prior parameters (default: (1,1))
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
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount factor
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
                gamma,
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


class OSCQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, theta=-1, gamma=0.5):
        """
        Initialize variables for the Omission Sensitive Long-Term Q-Learning agent
        =======================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        theta: omission sensitivity for the agent (float) (default: -1)
        gamma: discount rate for the agent (float) (default: 0.5)
        """
        self.alpha = alpha  # learning rate
        self.theta = theta  # omission sensitivity
        self.gamma = gamma  # discount rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Omission Sensitive Long-Term Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Omission Sensitive Long-Term Q-Learning step in the environment
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:])), if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (theta + gamma * max(Q[new_state,:])), if not rewarded
        ==============================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Omission Sensitive Long-Term Q-learning update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:])), if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (theta + gamma * max(Q[new_state,:])), if not rewarded
        ==============================================================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (
            reward + self.gamma * np.max(self.q_table[new_state, :] + self.theta * (reward == 0))
        )  # update Q-table

        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Omission Sensitive Long-Term Q-Learning agent.
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Omission Sensitive Long-Term Q-Learning agent.
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Omission Sensitive Long-Term Q-Learning agent.
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class OSCQLearner_egreedy(OSCQLearner):
    def init_policy_variables(self, epsilon=0.2):
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
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta, gamma):
        """
        Vectorized update of Q-table using Omission Sensitive Long-Term Q-Learning Algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(
            Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * tt.max(Qs) - Qs[action])
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, gamma, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        theta: omission sensitivity (float)
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
                non_sequences=[alpha, theta, gamma],
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
        theta_params=(0, 1),
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
        Fit the epsilon-greedy Omission Sensitive Long-Term QLearning agent to a given dataset.
        ==========================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, gamma, epsilon, actions_set, rewards_set
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


class OSCQLearner_softmax(OSCQLearner):
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
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta, gamma):
        """
        Vectorized update of Q-table using Omission Sensitive Long-Term Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(
            Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * tt.max(Qs) - Qs[action])
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, gamma, beta, actions_set, rewards_set):
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
                non_sequences=[alpha, theta, gamma],
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
        theta_params=(0, 1),
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
        Fit the softmax Omission Sensitive Long-Term QLearning agent to a given dataset.
        ===================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, gamma, beta, actions_set, rewards_set
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


class OSCQLearner_esoftmax(OSCQLearner):
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
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta, gamma):
        """
        Vectorized update of Q-table using Omission Sensitive Long-Term Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(
            Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * tt.max(Qs) - Qs[action])
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, gamma, beta, epsilon, actions_set, rewards_set):
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
                non_sequences=[alpha, theta, gamma],
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
        theta_params=(0, 1),
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
        Fit the epsilon-softmax Omission Sensitive Long-Term agent to a given dataset.
        =================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, gamma, beta, epsilon, actions_set, rewards_set
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


class OSCQLearner_acceptreject(OSCQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy.
        ====================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta, gamma):
        """
        Vectorized update of Q-table using Omission Sensitive Long-Term Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(
            Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * tt.max(Qs) - Qs[action])
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, gamma, weight, intercept, actions_set, rewards_set):
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
                non_sequences=[alpha, theta, gamma],
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
        theta_params=(0, 1),
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
        Fit the accept-reject Omission Sensitive Long-Term agent to a given dataset.
        =================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
        gamma_params: discount rate beta prior parameters (default: (1,1))
        weight_params: q-value weight normal prior parameters (default: (0,1))
        intercept_params: q-value weight normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, gamma, weight, intercept, actions_set, rewards_set
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


class FOSCQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, theta=-1, gamma=0.5):
        """
        Initialize variables for the Forgetting Omission Sensitive Long-Term Q-Learning agent
        =======================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        theta: omission sensitivity for the agent (float) (default: -1)
        gamma: discount rate for the agent (float) (default: 0.5)
        """
        self.alpha = alpha  # learning rate
        self.theta = theta  # omission sensitivity
        self.gamma = gamma  # discount rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Forgetting Omission Sensitive Long-Term Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Forgetting Omission Sensitive Long-Term Q-Learning step in the environment
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:])), if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (theta + gamma * max(Q[new_state,:])), if not rewarded
            Q[state, other action] = (1 - alpha) * Q[state, other action]
        ==============================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Forgetting Omission Sensitive Long-Term Q-learning update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:])), if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (theta + gamma * max(Q[new_state,:])), if not rewarded
            Q[state, other action] = (1 - alpha) * Q[state, other action]
        ==============================================================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        next_best = np.max(self.q_table[new_state, :])  # best value in the new state
        self.q_table[state, :] = (1 - self.alpha) * self.q_table[state, :]
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * next_best + self.theta * (reward == 0)
        )  # update Q-table
        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Forgetting Omission Sensitive Long-Term Q-Learning agent.
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Omission Sensitive Long-Term Q-Learning agent.
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Omission Sensitive Long-Term Q-Learning agent.
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class FOSCQLearner_egreedy(FOSCQLearner):
    def init_policy_variables(self, epsilon=0.2):
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
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta, gamma):
        """
        Vectorized update of Q-table using Forgetting Omission Sensitive Long-Term Q-Learning Algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * next_best))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, gamma, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        theta: omission sensitivity (float)
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
                non_sequences=[alpha, theta, gamma],
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
        theta_params=(0, 1),
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
        Fit the epsilon-greedy Forgetting Omission Sensitive Long-Term QLearning agent to a given dataset.
        ==========================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, gamma, epsilon, actions_set, rewards_set
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


class FOSCQLearner_softmax(FOSCQLearner):
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
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta, gamma):
        """
        Vectorized update of Q-table using Forgetting Omission Sensitive Long-Term Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * next_best))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, gamma, beta, actions_set, rewards_set):
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
                non_sequences=[alpha, theta, gamma],
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
        theta_params=(0, 1),
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
        Fit the softmax Forgetting Omission Sensitive Long-Term QLearning agent to a given dataset.
        ===================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, gamma, beta, actions_set, rewards_set
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


class FOSCQLearner_esoftmax(FOSCQLearner):
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
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta, gamma):
        """
        Vectorized update of Q-table using Forgetting Omission Sensitive Long-Term Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * next_best))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, gamma, beta, epsilon, actions_set, rewards_set):
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
                non_sequences=[alpha, theta, gamma],
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
        theta_params=(0, 1),
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
        Fit the epsilon-softmax Forgetting Omission Sensitive Long-Term agent to a given dataset.
        =================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, gamma, beta, epsilon, actions_set, rewards_set
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


class FOSCQLearner_acceptreject(FOSCQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy.
        ====================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, theta, gamma):
        """
        Vectorized update of Q-table using Forgetting Omission Sensitive Long-Term Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = (1 - alpha) * Qs
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * next_best))
        return Qs

    def vectorizedActionProbabilities(self, alpha, theta, gamma, weight, intercept, actions_set, rewards_set):
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
                non_sequences=[alpha, theta, gamma],
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
        theta_params=(0, 1),
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
        Fit the accept-reject Forgetting Omission Sensitive Long-Term agent to a given dataset.
        =================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
        gamma_params: discount rate beta prior parameters (default: (1,1))
        weight_params: q-value weight normal prior parameters (default: (0,1))
        intercept_params: q-value weight normal prior parameters (default: (0,1))
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
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, theta, gamma, weight, intercept, actions_set, rewards_set
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


class DFOSCQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, kappa=0.1, theta=-1, gamma=0.5):
        """
        Initialize variables for the Differential Forgetting Omission Sensitive Long-Term Q-Learning agent
        =======================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        kappa: forgetting rate for the agent (float) (default: 0.1)
        theta: omission sensitivity for the agent (float) (default: -1)
        gamma: discount rate for the agent (float) (default: 0.5)
        """
        self.alpha = alpha  # learning rate
        self.kappa = kappa  # forgetting rate
        self.theta = theta  # omission sensitivity
        self.gamma = gamma  # discount rate
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Q-table

    def init_recorder(self):
        """
        Initialize recorder for the Forgetting Omission Sensitive Long-Term Q-Learning agent
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Differential Forgetting Omission Sensitive Long-Term Q-Learning step in the environment
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:])), if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (theta + gamma * max(Q[new_state,:])), if not rewarded
            Q[state, other action] = (1 - kappa) * Q[state, other action]
        ==============================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)  # choose action

        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
            self.reward_history.append(reward)

        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        self.update_knowledge(state, action, new_state, reward)  # update Q-table

        return new_state, done

    def make_choice(self, state):
        """
        Make a choice based on the current state and random policy
        ==========================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)  # random action
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q-table of the agent according to the Forgetting Omission Sensitive Long-Term Q-learning update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[new_state,:])), if rewarded
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (theta + gamma * max(Q[new_state,:])), if not rewarded
            Q[state, other action] = (1 - kappa) * Q[state, other action]
        ==============================================================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        next_best = np.max(self.q_table[new_state, :])  # best value in the new state
        self.q_table[state, :action] = (1 - self.kappa) * self.q_table[state, :action]
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (
            reward + self.gamma * next_best + self.theta * (reward == 0)
        )  # update Q-table
        self.q_table[state, action + 1 :] = (1 - self.kappa) * self.q_table[state, action + 1 :]
        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Differential Forgetting Omission Sensitive Long-Term Q-Learning agent.
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        """
        Reset variables for the Differential Forgetting Omission Sensitive Long-Term Q-Learning agent.
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))  # Reset Q-table

    def reset_recorder(self):
        """
        Reset the recorder for the Differential Forgetting Omission Sensitive Long-Term Q-Learning agent.
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class DFOSCQLearner_egreedy(DFOSCQLearner):
    def init_policy_variables(self, epsilon=0.2):
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
        action_probabilities = np.ones(self.action_space_size) * (
            self.epsilon / self.action_space_size
        )  # equal probability of each action
        action_probabilities[np.argmax(self.q_table[state, :])] += (
            1 - self.epsilon
        )  # add probability of the best action
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, theta, gamma):
        """
        Vectorized update of Q-table using Differential Forgetting Omission Sensitive Long-Term Q-Learning Algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        theta: omission sensitivity (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(
            Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * next_best)
        )
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, theta, gamma, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        theta: omission sensitivity (float)
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
                non_sequences=[alpha, kappa, theta, gamma],
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
        kappa_params=(1, 1),
        theta_params=(0, 1),
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
        Fit the epsilon-greedy Forgetting Omission Sensitive Long-Term QLearning agent to a given dataset.
        ==========================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, theta, gamma, epsilon, actions_set, rewards_set
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


class DFOSCQLearner_softmax(DFOSCQLearner):
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
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, theta, gamma):
        """
        Vectorized update of Q-table using Differential Forgetting Omission Sensitive Long-Term Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(
            Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * next_best)
        )
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, theta, gamma, beta, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                non_sequences=[alpha, kappa, theta, gamma],
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
        kappa_params=(1, 1),
        theta_params=(0, 1),
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
        Fit the softmax Differential Forgetting Omission Sensitive Long-Term QLearning agent to a given dataset.
        ===================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, theta, gamma, beta, actions_set, rewards_set
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


class DFOSCQLearner_esoftmax(DFOSCQLearner):
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
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = (
            action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        )  # add exploration
        action = np.random.choice(self.action_space_size, p=action_probabilities)  # choose action
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, theta, gamma):
        """
        Vectorized update of Q-table using Differential Forgetting Omission Sensitive Long-Term Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        theta: omission sensitivity (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(
            Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * next_best)
        )
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, theta, gamma, beta, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
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
                non_sequences=[alpha, kappa, theta, gamma],
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
        kappa_params=(1, 1),
        theta_params=(0, 1),
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
        Fit the epsilon-softmax Differential Forgetting Omission Sensitive Long-Term agent to a given dataset.
        =================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            beta = pm.HalfNormal("beta", beta_params[0])  # temperature for softmax (halfnormal distribution)
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])  # exploration rate (beta distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, theta, gamma, beta, epsilon, actions_set, rewards_set
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


class DFOSCQLearner_acceptreject(DFOSCQLearner):
    def init_policy_variables(self, weight=1.0, intercept=0.0):
        """
        Initialize variables for the accept-reject policy.
        ====================================================

        Parameters:
        epsilon: exploration rate (default: 0.1)
        beta: temperature for softmax (default: 0.2)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, reward, Qs, alpha, kappa, theta, gamma):
        """
        Vectorized update of Q-table using Differential Forgetting Omission Sensitive Long-Term Q-learning algorithm.
        ========================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        theta: omission sensitivity (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        next_best = tt.max(Qs)
        Qs = tt.set_subtensor(Qs[:action], (1 - kappa) * Qs[:action])  # update Q-table
        Qs = tt.set_subtensor(
            Qs[action], (1 - alpha) * Qs[action] + alpha * (reward + theta * tt.eq(reward, 0) + gamma * next_best)
        )
        Qs = tt.set_subtensor(Qs[action + 1 :], (1 - kappa) * Qs[action + 1 :])
        return Qs

    def vectorizedActionProbabilities(self, alpha, kappa, theta, gamma, weight, intercept, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative.
        =======================================================

        Parameters:
        alpha: learning rate (float)
        kappa: forgetting rate (float)
        theta: omission sensitivity (float)
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
                non_sequences=[alpha, kappa, theta, gamma],
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
        kappa_params=(1, 1),
        theta_params=(0, 1),
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
        Fit the accept-reject Differential Forgetting Omission Sensitive Long-Term agent to a given dataset.
        =================================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)
        alpha_params: learning rate beta prior parameters (default: (1,1))
        kappa_params: forgetting rate beta prior parameters (default: (1,1))
        theta_params: omission sensitivity normal prior parameters (default: (0,1))
        gamma_params: discount rate beta prior parameters (default: (1,1))
        weight_params: q-value weight normal prior parameters (default: (0,1))
        intercept_params: q-value weight normal prior parameters (default: (0,1))
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
            kappa = pm.Beta("kappa", kappa_params[0], kappa_params[1])  # forgetting rate (beta distribution)
            theta = pm.Normal("theta", theta_params[0], theta_params[1])  # omission sensitivity (normal distribution)
            gamma = pm.Beta("gamma", gamma_params[0], gamma_params[1])  # discount rate (beta distribution)
            weight = pm.Normal("weight", weight_params[0], weight_params[1])  # q-value weight (normal distribution)
            intercept = pm.Normal(
                "intercept", intercept_params[0], intercept_params[1]
            )  # q-value intercept (normal distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, kappa, theta, gamma, weight, intercept, actions_set, rewards_set
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


class SARSALearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, gamma=0.5):
        """
        Initialize variables for the SARSA agent.
        ===================================

        Parameters:
        alpha: learning rate (float) for the agent (default: 0.1)
        gamma: discount rate (float) for the agent (default: 0.5)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))

    def init_recorder(self):
        """
        Initialize recorder for the SARSA agent.
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state, action):
        """
        Perform a single SARSA step in the environment.
        - Get reward and new state.
        - Make a choice based on the new state.
        - Update according to the equation:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * Q[new_state,new_action])
        ==================================================================================================================================

        Parameters:
        state: current state (int)
        action: current action (int)

        Returns:
        new_state: new state (int)
        new_action: new action (int)
        done: whether the episode is over (bool)
        """
        new_state, reward, done, info = self.env.step(action)  # get reward and new state

        if self.history:
            self.action_history.append(action)
        if action == self.biased_action:
            self.bias += 1  # update bias estimate

        if self.history:
            self.reward_history.append(reward)

        new_action = self.make_choice(new_state)  # make a choice based on the new state

        self.update_knowledge(state, action, new_state, new_action, reward)  # update Q-table

        return new_state, new_action, done

    def make_choice(self, state):
        """
        Make a choice based on the current state using a random policy.
        ===============================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action = np.random.choice(self.action_space_size)
        return action

    def update_knowledge(self, state, action, new_state, new_action, reward):
        """
        Update the knowledge of the agent according to the SARSA update rule:
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * Q[new_state,new_action])
        ==================================================================================================================================

        Parameters:
        state: current state (int)
        action: current action (int)
        new_state: new state (int)
        new_action: new action (int)
        reward: reward (float)
        """
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (
            reward + self.gamma * self.q_table[new_state, new_action]
        )
        if self.history:
            self.q_history.append([self.q_table[0, 0], self.q_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the SARSA agent.
        """
        state = self.env.reset()
        done = False
        action = self.make_choice(state)  # make a choice based on the initial state

        while not done:
            state, action, done = self.trial_step(state, action)  # perform a single SARSA step

    def reset_variables(self):
        """
        Reset variables for the SARSA agent.
        """
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))

    def reset_recorder(self):
        """
        Reset the recorder for the SARSA agent.
        """
        self.q_history = []
        self.reward_history = []
        self.action_history = []


class SARSALearner_egreedy(SARSALearner):
    def init_policy_variables(self, epsilon=0.1):
        """
        Initialize variables for the epsilon-greedy policy
        ==================================================

        Parameters:
        epsilon: exploration rate for the agent (float) (default: 0.1)
        """
        self.epsilon = epsilon

    def make_choice(self, state):
        """
        Make a choice based on the current state using an epsilon-greedy policy
        =======================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        action_probabilities = np.ones(self.action_space_size) * (self.epsilon / self.action_space_size)
        action_probabilities[np.argmax(self.q_table[state, :])] += 1 - self.epsilon
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action

    def vectorizedUpdate(self, action, next_action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using SARSA Algorithm
        ==================================================

        Parameters:
        action: action taken (0 or 1)
        next_action: action to be taken in the next state (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * Qs[next_action] - Qs[action]))
        return Qs

    def vectorizedActionProbabilities(self, alpha, gamma, epsilon, actions_set, rewards_set):
        """
        Vectorized action probabilities for second alternative
        ======================================================

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
        for (actions_, rewards_) in zip(actions_set, rewards_set):
            rewards = theano.shared(np.asarray(rewards_[:-1], dtype="int16"))
            actions = theano.shared(np.asarray(actions_[:-1], dtype="int16"))
            next_actions = theano.shared(np.asarray(actions_[1:], dtype="int16"))

            # Compute the Qs values
            Qs = tt.zeros((2), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, next_actions, rewards],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma],
            )

            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((2), dtype="float64")], [tt.zeros((2), dtype="float64")], Qs], axis=0)
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
        Fit the epsilon-greedy SARSA agent to a given dataset.
        ======================================================

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
        m: fitted model (PyMC3 model)
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


class SARSALearner_softmax(SARSALearner):
    def init_policy_variables(self, beta=0.2):
        """
        Initialize variables for the softmax policy.
        ============================================

        Parameters:
        beta: temperature for softmax (float)
        """
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state using a softmax policy.
        ================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action

    def vectorizedUpdate(self, action, next_action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using SARSA Algorithm.
        ===================================================

        Parameters:
        action: action taken (0 or 1)
        next_action: action to be taken in the next state (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * Qs[next_action] - Qs[action]))
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
        for (actions_, rewards_) in zip(actions_set, rewards_set):
            rewards = theano.shared(np.asarray(rewards_[:-1], dtype="int16"))
            actions = theano.shared(np.asarray(actions_[:-1], dtype="int16"))
            next_actions = theano.shared(np.asarray(actions_[1:], dtype="int16"))

            # Compute the Qs values
            Qs = tt.zeros((2), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, next_actions, rewards],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma],
            )

            # Apply the softmax function
            Qs_ = tt.concatenate([[tt.zeros((2), dtype="float64")], [tt.zeros((2), dtype="float64")], Qs], axis=0)
            Qs_ = Qs_ / beta
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
        Fit the softmax SARSA agent to a given dataset.
        ===============================================

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


class SARSALearner_esoftmax(SARSALearner):
    def init_policy_variables(self, epsilon=0.1, beta=0.2):
        """
        Initialize variables for the epsilon-softmax policy.
        ====================================================

        Parameters:
        epsilon: exploration rate for epsilon-softmax (float)
        beta: temperature for epsilon-softmax (float)
        """
        self.epsilon = epsilon
        self.beta = beta

    def make_choice(self, state):
        """
        Make a choice based on the current state using a epsilon-softmax policy.
        ========================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
        """
        logits = self.q_table[state, :] / self.beta
        logits -= np.max(logits)
        log_prob_actions = logits - np.log(np.sum(np.exp(logits)))
        action_probabilities = np.exp(log_prob_actions)
        action_probabilities = action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action

    def vectorizedUpdate(self, action, next_action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using SARSA Algorithm.
        ===================================================

        Parameters:
        action: action taken (0 or 1)
        next_action: action to be taken in the next state (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * Qs[next_action] - Qs[action]))
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
        for (actions_, rewards_) in zip(actions_set, rewards_set):
            rewards = theano.shared(np.asarray(rewards_[:-1], dtype="int16"))
            actions = theano.shared(np.asarray(actions_[:-1], dtype="int16"))
            next_actions = theano.shared(np.asarray(actions_[1:], dtype="int16"))

            # Compute the Qs values
            Qs = tt.zeros((2), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, next_actions, rewards],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma],
            )

            # Apply the softmax function
            Qs_ = tt.concatenate([[tt.zeros((2), dtype="float64")], [tt.zeros((2), dtype="float64")], Qs], axis=0)
            Qs_ = Qs_[:-1] / beta
            log_prob_actions = Qs_ - pm.math.logsumexp(Qs_, axis=1)

            # Return the probabilities for the right action, in the original scale
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
        Fit the epsilon-softmax SARSA agent to a given dataset.
        =======================================================

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


class SARSALearner_acceptreject(SARSALearner):
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
        Make a choice based on the current state using a accept-reject policy.
        ========================================================================

        Parameters:
        state: current state (int)

        Returns:
        action: action to be taken (int)
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
        action = np.random.choice(
            self.action_space_size, p=[1 - action_1_probability, action_1_probability]
        )  # make choice
        return action

    def vectorizedUpdate(self, action, next_action, reward, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using SARSA Algorithm.
        ===================================================

        Parameters:
        action: action taken (0 or 1)
        next_action: action to be taken in the next state (0 or 1)
        reward: reward received (0 or 1)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        Qs = tt.set_subtensor(Qs[action], Qs[action] + alpha * (reward + gamma * Qs[next_action] - Qs[action]))
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
        for (actions_, rewards_) in zip(actions_set, rewards_set):
            rewards = theano.shared(np.asarray(rewards_[:-1], dtype="int16"))
            actions = theano.shared(np.asarray(actions_[:-1], dtype="int16"))
            next_actions = theano.shared(np.asarray(actions_[1:], dtype="int16"))

            # Compute the Qs values
            Qs = tt.zeros((2), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, next_actions, rewards],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma],
            )

            # Apply the softmax function
            Qs_ = tt.concatenate([[tt.zeros((2), dtype="float64")], [tt.zeros((2), dtype="float64")], Qs], axis=0)
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
        Fit the accept-reject SARSA agent to a given dataset.
        =======================================================

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


class DQLearner(FlYMazeAgent):
    def init_variables(self, alpha=0.1, gamma=0.5):
        """
        Initialize variables for the Double Q-Learning agent.
        =====================================================

        Parameters:
        alpha: learning rate for the agent (float) (default: 0.1)
        gamma: discount rate for the agent (float) (default: 0.5)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.q1_table = np.zeros((self.state_space_size, self.action_space_size))
        self.q2_table = np.zeros((self.state_space_size, self.action_space_size))

    def init_recorder(self):
        """
        Initialize recorder for the Q-Learning agent.
        """
        self.q1_history = []
        self.q2_history = []
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a single Double Q-Learning step in the environment.
        - Make a choice based on the current state.
        - Get reward and new state.
        - Update randomly according to either of the following equations:
            q1[state, action] = q1[state, action] + alpha * (reward + gamma * q2[new_state, argmax(q1[new_state,:])] - q1[state, action])
         OR q2[state, action] = q2[state, action] + alpha * (reward + gamma * q1[new_state, argmax(q2[new_state,:])] - q2[state, action])
        =================================================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: new state (int)
        done: whether the episode is done (bool)
        """
        action = self.make_choice(state)
        if self.history:
            self.action_history.append(action)
        if action == self.biased_action:
            self.bias += 1
        new_state, reward, done, info = self.env.step(action)
        if self.history:
            self.reward_history.append(reward)
        self.update_knowledge(state, action, new_state, reward)

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
        action = np.random.choice(self.action_space_size)
        return action

    def update_knowledge(self, state, action, new_state, reward):
        """
        Update Q Tables of agent randomly according to either of the following equations:
            q1[state, action] = q1[state, action] + alpha * (reward + gamma * q2[new_state, argmax(q1[new_state,:])] - q1[state, action])
         OR q2[state, action] = q2[state, action] + alpha * (reward + gamma * q1[new_state, argmax(q2[new_state,:])] - q2[state, action])
        ==============================================================================================================================

        Parameters:
        state: current state (int)
        action: action taken (int)
        new_state: new state (int)
        reward: reward received (float)
        """
        if np.random.uniform() < 0.5:
            self.q1_table[state, action] = (1 - self.alpha) * self.q1_table[state, action] + self.alpha * (
                reward + self.gamma * self.q2_table[new_state, np.argmax(self.q1_table[new_state, :])]
            )
        else:
            self.q2_table[state, action] = (1 - self.alpha) * self.q2_table[state, action] + self.alpha * (
                reward + self.gamma * self.q1_table[new_state, np.argmax(self.q2_table[new_state, :])]
            )
        if self.history:
            self.q1_history.append([self.q1_table[0, 0], self.q1_table[0, 1]])
            self.q2_history.append([self.q2_table[0, 0], self.q2_table[0, 1]])

    def run_episode(self):
        """
        Describe a single episode of the Double Q-Learning agent.
        """
        state = self.env.reset()
        done = False
        while not done:
            state, done = self.trial_step(state)

    def reset_variables(self):
        """
        Reset variables for the Double Q-Learning agent.
        """
        self.q1_table = np.zeros((self.state_space_size, self.action_space_size))
        self.q2_table = np.zeros((self.state_space_size, self.action_space_size))

    def reset_recorder(self):
        """
        Reset the recorder.
        """
        self.q1_history = []
        self.q2_history = []
        self.reward_history = []
        self.action_history = []


class DQLearner_egreedy(DQLearner):
    def init_policy_variables(self, epsilon=0.2):
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
        action_probabilities[np.argmax(self.q1_table[state, :] + self.q2_table[state, :])] += 1 - self.epsilon
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action

    def vectorizedUpdate(self, action, reward, table, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Double Q-Learning Algorithm.
        ===============================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        table: Q-table position to be updated (0 or 2)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        table_ = 2 - table
        Qs = tt.set_subtensor(
            Qs[action + table],
            Qs[action + table]
            + alpha * (reward + gamma * Qs[table_ : table_ + 2][tt.argmax(Qs[table : table + 2])] - Qs[action + table]),
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, gamma, epsilon, table, actions_set, rewards_set):
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
            Qs = tt.zeros((4), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards, table],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma],
            )

            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((4), dtype="float64")], Qs], axis=0)
            Qs_ = (Qs_[:-1, :2] + Qs_[:-1, 2:]) / 2
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
        plot_trace=False,
        plot_posterior=False,
        print_summary=True,
    ):
        """
        Fit the epsilon-greedy Double QLearning agent to a given dataset.
        =================================================================

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
            table = pm.Categorical(
                "p", p=[0.5, 0, 0.5], shape=self.env.n_trials_per_session
            )  # Q-table position (categorical distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, gamma, epsilon, table, actions_set, rewards_set
            )  # action probabilities
            like = pm.Bernoulli(
                "like", p=action_probabilities, observed=np.concatenate([a for a in actions_set])
            )  # Bernoulli likelihood
            trace = pm.sample(
                tune=ntune, draws=niters, chains=nchains, cores=nparallel, return_inferencedata=True,
            )  # sample posterior

        if plot_trace:  # plot trace
            az.plot_trace(trace, var_names=["alpha", "gamma", "epsilon"])

        if plot_posterior:  # plot posterior
            az.plot_posterior(trace, hdi_prob=0.95, var_names=["alpha", "gamma", "epsilon"])

        summary = az.summary(trace, hdi_prob=0.95, var_names=["alpha", "gamma", "epsilon"])  # compute summary

        if print_summary:  # print summary
            print(summary)

        return model, trace


class DQLearner_softmax(DQLearner):
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
        action_probabilities = np.exp((self.q1_table[state, :] + self.q2_table[state, :]) / self.beta)
        action_probabilities /= np.sum(action_probabilities)
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action

    def vectorizedUpdate(self, action, reward, table, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Double Q-Learning Algorithm.
        ===============================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        table: Q-table position to be updated (0 or 2)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        table_ = 2 - table
        Qs = tt.set_subtensor(
            Qs[action + table],
            Qs[action + table]
            + alpha * (reward + gamma * Qs[table_ : table_ + 2][tt.argmax(Qs[table : table + 2])] - Qs[action + table]),
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, gamma, beta, table, actions_set, rewards_set):
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
            Qs = tt.zeros((4), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards, table],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma],
            )

            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((4), dtype="float64")], Qs], axis=0)
            Qs_ = (Qs_[:-1, :2] + Qs_[:-1, 2:]) / (2 * beta)
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
        plot_trace=False,
        plot_posterior=False,
        print_summary=True,
    ):
        """
        Fit the softmax QLearning agent to a given dataset.
        ===================================================

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
            table = pm.Categorical(
                "p", p=[0.5, 0, 0.5], shape=self.env.n_trials_per_session
            )  # Q-table position (categorical distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, gamma, beta, table, actions_set, rewards_set
            )  # action probabilities
            like = pm.Bernoulli(
                "like", p=action_probabilities, observed=np.concatenate([a for a in actions_set])
            )  # Bernoulli likelihood
            trace = pm.sample(
                tune=ntune, draws=niters, chains=nchains, cores=nparallel, return_inferencedata=True,
            )  # sample posterior

        if plot_trace:  # plot trace
            az.plot_trace(trace, var_names=["alpha", "gamma", "beta"])

        if plot_posterior:  # plot posterior
            az.plot_posterior(trace, hdi_prob=0.95, var_names=["alpha", "gamma", "beta"])

        summary = az.summary(trace, hdi_prob=0.95, var_names=["alpha", "gamma", "beta"])  # compute summary

        if print_summary:  # print summary
            print(summary)

        return model, trace


class DQLearner_esoftmax(DQLearner):
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
        """
        action_probabilities = np.exp((self.q1_table[state, :] + self.q2_table[state, :]) / self.beta)
        action_probabilities /= np.sum(action_probabilities)
        action_probabilities = action_probabilities * (1 - self.epsilon) + self.epsilon / self.action_space_size
        action = np.random.choice(self.action_space_size, p=action_probabilities)
        return action

    def vectorizedUpdate(self, action, reward, table, Qs, alpha, gamma):
        """
        Vectorized update of Q-table using Double Q-Learning Algorithm.
        ===============================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received (0 or 1)
        table: Q-table position to be updated (0 or 2)
        Qs: current Q-value (Theano tensor)
        alpha: learning rate (float)
        gamma: discount rate (float)

        Returns:
        Qs: updated Q-value (Theano tensor)
        """
        table_ = 2 - table
        Qs = tt.set_subtensor(
            Qs[action + table],
            Qs[action + table]
            + alpha * (reward + gamma * Qs[table_ : table_ + 2][tt.argmax(Qs[table : table + 2])] - Qs[action + table]),
        )
        return Qs

    def vectorizedActionProbabilities(self, alpha, gamma, beta, epsilon, table, actions_set, rewards_set):
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
            Qs = tt.zeros((4), dtype="float64")
            Qs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards, table],
                outputs_info=[Qs],
                non_sequences=[alpha, gamma],
            )

            # Compute the action probabilities
            Qs_ = tt.concatenate([[tt.zeros((4), dtype="float64")], Qs], axis=0)
            Qs_ = (Qs_[:-1, :2] + Qs_[:-1, 2:]) / (2 * beta)
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
        plot_trace=False,
        plot_posterior=False,
        print_summary=True,
    ):
        """
        Fit the epsilon-softmax agent to a given dataset.
        =================================================

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
            table = pm.Categorical(
                "p", p=[0.5, 0, 0.5], shape=self.env.n_trials_per_session
            )  # Q-table position (categorical distribution)

            action_probabilities = self.vectorizedActionProbabilities(
                alpha, gamma, beta, epsilon, table, actions_set, rewards_set
            )  # action probabilities
            like = pm.Bernoulli(
                "like", p=action_probabilities, observed=np.concatenate([a for a in actions_set])
            )  # Bernoulli likelihood
            trace = pm.sample(
                tune=ntune, draws=niters, chains=nchains, cores=nparallel, return_inferencedata=True,
            )  # sample posterior

        if plot_trace:  # plot trace
            az.plot_trace(trace, var_names=["alpha", "gamma", "beta", "epsilon"])

        if plot_posterior:  # plot posterior
            az.plot_posterior(trace, hdi_prob=0.95, var_names=["alpha", "gamma", "beta", "epsilon"])

        summary = az.summary(trace, hdi_prob=0.95, var_names=["alpha", "gamma", "beta", "epsilon"])  # compute summary

        if print_summary:  # print summary
            print(summary)

        return model, trace
