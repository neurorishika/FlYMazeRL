import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
import arviz as az

from flymazerl.agents.base import FlYMazeAgent


class BayesianIdealObserver(FlYMazeAgent):
    def init_variables(self, value_weight=1.0, explore_weight=0.0, bias_weight=0.0):
        """
        Initialize the variables for the Beta-Bernoulli Bayesian Ideal Observer
        =======================================================================

        Parameters:
        value_weight: weight given to the differences in reward expectations (float) (default: 1.0)
        explore_weight: weight given to the differences in reward variability (float) (default: 0.0)
        bias_weight: weight attributed to side preference (float) (default: 0.0)
        """
        self.value_weight = value_weight
        self.explore_weight = explore_weight
        self.bias_weight = bias_weight
        self.parameterized_beliefs = np.ones((self.action_space_size, 2))  # initialize beliefs
        self.reward_history = []
        self.action_history = []

    def trial_step(self, state):
        """
        Perform a step in the environment according to the Beta-Bernoulli Bayesian Ideal Observer algorithm
        ===================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: next state (int)
        done: whether the episode is done (bool)
        """

        alpha = self.parameterized_beliefs[:, 0]  # alpha parameters of the beta distributed beliefs
        beta = self.parameterized_beliefs[:, 1]  # beta parameters of the beta distributed beliefs
        E = alpha / (alpha + beta)  # Mean of the beta distributed beliefs
        V = alpha * beta / (alpha + beta) ** 2 * (alpha + beta + 1)  # Variance of the beta distributed beliefs

        sigmoid = lambda z: 1 / (1 + np.exp(-z))  # Sigmoid function

        action_prob = sigmoid(
            self.value_weight * (E[1] - E[0])
            + self.explore_weight * (V[1] - V[0])
            + self.bias_weight  # Weighted sum of the mean and variance of the beta distributed beliefs along with the bias
        )
        action = np.random.choice(
            self.action_space_size, p=[1 - action_prob, action_prob]
        )  # Choose an action according to the probability distribution

        new_state, reward, done, _ = self.env.step(action)

        self.parameterized_beliefs[
            action, int(reward)
        ] += 1  # Update the parameterized beliefs according to the beta-bernoulli conjugate prior equations

        self.reward_history.append(reward)  # Update the reward history
        self.action_history.append(action)  # Update the action history

        if action == self.biased_action:
            self.bias += 1 / self.env.n_trials_per_session  # update bias estimate

        return new_state, done

    def run_episode(self):
        """
        Run an episode of the environment according to the Beta-Bernoulli Bayesian Ideal Observer algorithm
        """
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        self.action_history = []
        self.reward_history = []
        self.parameterized_beliefs = np.ones((self.action_space_size, 2))

    def vectorizedUpdate(self, action, reward, parameterized_beliefs):
        """
        Vectorized update of Q-table using Q-learning algorithm
        =======================================================

        Parameters:
        action: action taken (0 or 1)
        reward: reward received 0 or 1)
        parameterized_beliefs: parameterized beliefs (theano tensor)

        Returns:
        parameterized_beliefs: updated parameterized beliefs (numpy array)
        """
        parameterized_beliefs = tt.set_subtensor(
            parameterized_beliefs[action, reward], parameterized_beliefs[action, reward] + 1
        )
        return parameterized_beliefs

    def vectorizedActionProbabilities(self, value_weight, explore_weight, bias_weight, actions_set, rewards_set):
        """
        Vectorized action probabilities using the Beta-Bernoulli Bayesian Ideal Observer algorithm
        ==========================================================================================

        Parameters:
        value_weight: weight given to the differences in reward expectations (float)
        explore_weight: weight given to the differences in reward variability (float)
        bias_weight: weight attributed to side preference (float)
        actions_set: set of actions (theano tensor)
        rewards_set: set of rewards (theano tensor)

        Returns:
        action_probabilities: action probabilities (theano tensor)
        """
        probabilities = []

        for (actions_, rewards_) in zip(actions_set, rewards_set):
            rewards = theano.shared(np.asarray(rewards_, dtype="int16"))  # convert to theano tensor
            actions = theano.shared(np.asarray(actions_, dtype="int16"))  # convert to theano tensor

            parameterized_beliefs = tt.ones((self.action_space_size, 2))  # initialize parameterized beliefs

            parameterized_beliefs, _ = theano.scan(
                fn=self.vectorizedUpdate,
                sequences=[actions, rewards],
                outputs_info=[parameterized_beliefs],  # update parameterized beliefs
            )
            parameterized_beliefs = tt.concatenate(
                [[tt.ones((self.action_space_size, 2))], parameterized_beliefs],
                axis=0,  # concatenate initial parameterized beliefs
            )

            alpha = parameterized_beliefs[:-1, :, 0]  # alpha parameters of the beta distributed beliefs
            beta = parameterized_beliefs[:-1, :, 1]  # beta parameters of the beta distributed beliefs
            E = alpha / (alpha + beta)  # Mean of the beta distributed beliefs
            V = alpha * beta / (alpha + beta) ** 2 * (alpha + beta + 1)  # Variance of the beta distributed beliefs

            sigmoid = lambda z: 1 / (1 + tt.exp(-z))  # Sigmoid function

            probabilities.append(
                sigmoid(
                    value_weight * (E[:, 1] - E[:, 0]) + explore_weight * (V[:, 1] - V[:, 0]) + bias_weight
                )  # Weighted sum of the mean and variance of the beta distributed beliefs along with the bias
            )
        action_probabilities = tt.concatenate(
            probabilities, axis=0
        )  # concatenate action probabilities for all sessions
        return action_probabilities

    def fit(
        self,
        actions_set,
        rewards_set,
        value_weight_params=(0, 1),
        explore_weight_params=(0, 1),
        bias_weight_params=(0, 1),
        niters=1000,
        ntune=1000,
        nchains=2,
        nparallel=1,
        plot_trace=True,
        plot_posterior=True,
        print_summary=True,
    ):
        with pm.Model() as m:  # define model
            value_weight = pm.Normal("value_weight", mu=value_weight_params[0], sd=value_weight_params[1])
            explore_weight = pm.Normal("explore_weight", mu=explore_weight_params[0], sd=explore_weight_params[1])
            bias_weight = pm.Normal("bias_weight", mu=bias_weight_params[0], sd=bias_weight_params[1])

            action_probabilities = self.vectorizedActionProbabilities(
                value_weight, explore_weight, bias_weight, actions_set, rewards_set
            )  # action probabilities
            like = pm.Bernoulli(
                "like", p=action_probabilities, observed=np.concatenate([a for a in actions_set])
            )  # Bernoulli likelihood
            trace = pm.sample(
                tune=ntune, draws=niters, chains=nchains, cores=nparallel, return_inferencedata=True
            )  # sample posterior

        if plot_trace:  # plot trace
            az.plot_trace(trace)

        if plot_posterior:  # plot posterior
            az.plot_posterior(trace, hdi_prob=0.95)

        summary = az.summary(trace, hdi_prob=0.95)  # compute summary

        if print_summary:  # print summary
            print(summary)

        return m, trace


class CATIELearner(FlYMazeAgent):
    def init_variables(self, tau=0.1, epsilon=0.2, phi=0.1, K=15):
        """
        Initialize the variables for the CATIE Learner
        ==============================================

        Parameters:
        tau: probability of trend mode given conditions satisfied (float) (default: 0.1)
        epsilon: maximum probability of explore mode given conditions satisfied (float) (default: 0.2)
        phi: probability of inertial mode given conditions satisfied (float) (default: 0.1)
        K: maximum number of past trials to consider (int) (default: 15)
        """
        self.tau = tau
        self.epsilon = epsilon
        self.phi = phi
        self.K = K
        self.action_history = []
        self.reward_history = []
        self.surprise = []
        self.sampled_alternatives = np.array([False] * self.action_space_size)

    def search_history(self, arr, seq):
        """
        Find the indices after a sequence in an array
        =============================================

        Parameters:
        arr: array to search (list or numpy array)
        seq: sequence to search for (list or numpy array)

        Returns:
        indices: indices of the sequence in the array (list)
        """
        arr = np.array(arr)
        seq = np.array(seq)
        Na, Nseq = arr.size, seq.size
        indices = [i + Nseq for i in range(0, Na) if list(arr[i : i + Nseq]) == list(seq)]
        return indices

    def get_k_contingent_averages(self, k, action, reward_history=None, action_history=None):
        """
        Find the k-contingent averages for a given action
        =================================================

        Parameters:
        k: number of past trials to consider (int)
        action: action to consider (int)
        reward_history: reward history (None or numpy array) (default: None)
        action_history: action history (None or numpy array) (default: None)
        """

        # Default to the current reward history and action history
        if reward_history is None:
            reward_history = np.array(self.reward_history)
        if action_history is None:
            action_history = np.array(self.action_history)

        if k == 0:
            # If k is 0, return the average reward for the action
            return np.mean(reward_history[action_history == action])
        else:
            contingency = reward_history[-k:]
            contingency_indices = self.search_history(reward_history[:-1], contingency)
            contingency_action_history = action_history[contingency_indices]
            contingency_reward_history = reward_history[contingency_indices]
            if np.any(contingency_action_history == action):
                # If the action is in the k-contingent, return the average reward for the action
                return np.mean(contingency_reward_history[contingency_action_history == action])
            else:
                # If the action is not in the k-contingent, find if there is a k-contingent with the action after it
                # If not, evaluate using a k-1 contingency
                try:
                    viable_indices = np.arange(k, len(action_history))[action_history[k:] == action]
                except:
                    return self.get_k_contingent_averages(k - 1, action)
                if viable_indices.size == 0:
                    return self.get_k_contingent_averages(k - 1, action)
                # If there is a k-contingent with the action after it, randomly choose one
                viable_contingencies = [reward_history[i - k : i] for i in viable_indices]
                contingency = viable_contingencies[np.random.choice(np.arange(len(viable_contingencies)))]
                # Find the average reward for the action with that k-contingent
                contingency_indices = self.search_history(reward_history, contingency)
                contingency_action_history = action_history[contingency_indices]
                contingency_reward_history = reward_history[contingency_indices]
                return np.mean(contingency_reward_history[contingency_action_history == action])

    def update_surprise(self, action, reward):
        """
        Update the surprise for the given action and reward as the variance normalized deviation from the expected reward
        =================================================================================================================

        Parameters:
        action: action to consider (int)
        reward: reward for the action (float)
        """
        ObsSD = np.std(np.array(self.reward_history)[np.array(self.action_history) == action])
        Exp = np.mean(np.array(self.reward_history)[np.array(self.action_history) == action])
        if ObsSD > 0:
            self.surprise.append(np.abs(reward - Exp) / (ObsSD + np.abs(reward - Exp)))
        else:
            self.surprise.append(0)
        pass

    def trial_step(self, state):
        """
        Perform a step in the environment according to the Contingent Average Trend Inertia Exploration (CATIE) algorithm
        =================================================================================================================

        Parameters:
        state: current state (int)

        Returns:
        new_state: next state (int)
        done: whether the episode is done (bool)
        """

        # Get the action, reward, and surprise histories as numpy arrays
        a_history = np.array(self.action_history)
        r_history = np.array(self.reward_history)
        s_history = np.array(self.surprise)

        if not np.all(self.sampled_alternatives):
            # If the alternatives have not been sampled, sample them
            action = np.random.choice(np.arange(self.action_space_size)[np.logical_not(self.sampled_alternatives)])
            self.sampled_alternatives[action] = True
        else:
            # If the alternatives have been sampled, find the probability of choosing the second alternative according to the Trend Mode
            adjusted_tau = int((a_history[-1] == a_history[-2]) and (r_history[-1] != r_history[-2])) * self.tau
            if r_history[-1] > r_history[-2]:
                trend_mode_action_prob = a_history[-1]
            else:
                if a_history[-1] == 1:
                    trend_mode_action_prob = 0
                else:
                    trend_mode_action_prob = 1 / (self.action_space_size - 1)
            # Find the probability of exploring the second alternative given the surprise according to the Exploration Mode
            adjusted_epsilon = self.epsilon * (1 + s_history[-1] + np.mean(s_history)) / 3
            explore_mode_action_prob = 1 / self.action_space_size
            # Find the probability of choosing the second alternative according to the Inertial Mode
            inertial_mode_action_prob = a_history[-1]
            # Find the probability of choosing the second alternative according to the Contingent Average Mode
            k = np.random.choice(np.arange(1, self.K + 1))  # Randomly choose the size of the k-contingent
            CAs = [
                self.get_k_contingent_averages(k, action, a_history, r_history)
                for action in np.arange(self.action_space_size)
            ]
            CA_mode_action_prob = float(np.argmax(CAs) == 1)
            # Find the total action probability for the second alternative
            action_prob = adjusted_tau * (trend_mode_action_prob) + (1 - adjusted_tau) * (
                adjusted_epsilon * explore_mode_action_prob
                + (1 - adjusted_epsilon) * (self.phi * inertial_mode_action_prob + (1 - self.phi) * CA_mode_action_prob)
            )
            # Sample action according to the action probability for the second alternative
            action = np.random.choice([0, 1], p=[1 - action_prob, action_prob])

        new_state, reward, done, _ = self.env.step(action)

        self.update_surprise(action, reward)

        self.reward_history.append(reward)
        self.action_history.append(action)

        if action == self.biased_action:
            self.bias += 1 / self.env.n_trials_per_session  # update bias estimate

        return new_state, done

    def run_episode(self):
        state = self.env.reset()  # reset environment
        done = False
        while not done:
            state, done = self.trial_step(state)  # trial step

    def reset_variables(self):
        self.action_history = []
        self.reward_history = []
        self.surprise = []
        self.sampled_alternatives = np.array([False] * self.action_space_size)

    def vectorizedActionProbabilities(self, tau, epsilon, phi, actions_set, rewards_set):
        probabilities = []

        for (actions, rewards) in zip(actions_set, rewards_set):
            action_prob = [1 / self.action_space_size]
            s_history = [0]

            for i in range(1, len(actions)):

                a_history = np.array(actions[:i])
                r_history = np.array(rewards[:i])
                ObsSD = np.std(r_history[a_history == actions[i]])
                Exp = np.mean(r_history[a_history == actions[i]])
                if ObsSD > 0:
                    surprise = np.abs(rewards[i] - Exp) / (ObsSD + np.abs(rewards[i] - Exp))
                else:
                    surprise = 0
                s_history.append(surprise)

                if i == 1:
                    action_prob.append(np.float(np.logical_xor(a_history[-1], 1)))
                else:
                    adjusted_tau = int((a_history[-1] == a_history[-2]) and (r_history[-1] != r_history[-2])) * tau
                    if r_history[-1] > r_history[-2]:
                        trend_mode_action_prob = a_history[-1]
                    else:
                        if a_history[-1] == 1:
                            trend_mode_action_prob = 0
                        else:
                            trend_mode_action_prob = 1 / (self.action_space_size - 1)

                    adjusted_epsilon = epsilon * (1 + s_history[-1] + np.mean(s_history)) / 3
                    explore_mode_action_prob = 1 / self.action_space_size
                    inertial_mode_action_prob = a_history[-1]

                    k = np.random.choice(np.arange(1, self.K + 1))
                    CAs = [
                        self.get_k_contingent_averages(k, action, a_history, r_history)
                        for action in np.arange(self.action_space_size)
                    ]
                    CA_mode_action_prob = float(np.argmax(CAs) == 1)

                    action_prob.append(
                        adjusted_tau * (trend_mode_action_prob)
                        + (1 - adjusted_tau)
                        * (
                            adjusted_epsilon * explore_mode_action_prob
                            + (1 - adjusted_epsilon)
                            * (phi * inertial_mode_action_prob + (1 - phi) * CA_mode_action_prob)
                        )
                    )

            probabilities.append(action_prob)
        action_probabilities = tt.concatenate(probabilities, axis=0)
        return tt.clip(action_probabilities, 0.001, 0.999)

    def fit(
        self,
        actions_set,
        rewards_set,
        tau_params=(1, 1),
        epsilon_params=(1, 1),
        phi_params=(1, 1),
        niters=1000,
        ntune=1000,
        nchains=2,
        nparallel=1,
        plot_trace=True,
        plot_posterior=True,
        print_summary=True,
    ):
        """
        Fit the CATIE agent to a given dataset
        ======================================

        Parameters:
        actions_set: dataset of actions (0 or 1) across multiple sessions (list of lists)
        rewards_set: dataset of rewards (0 or 1) across multiple sessions (list of lists)

        niters: number of iterations in the MC-MC sampling (default: 1000)
        nchains: number of parallel chains (default: 2)
        nparallel: number of parallel sampling processes (default: 1)
        plot_trace: plot trace of posterior distributions (default: True)
        plot_posterior: plot posterior distributions (default: True)
        print_summary: print summary of posterior distributions (default: False)

        Returns:
        trace: trace of posterior distributions (PyMC3 trace object)
        """
        with pm.Model() as m:  # define model
            tau = pm.Beta("tau", tau_params[0], tau_params[1])
            epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])
            phi = pm.Beta("phi", phi_params[0], phi_params[1])

            action_probabilities = self.vectorizedActionProbabilities(
                tau, epsilon, phi, actions_set, rewards_set
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
            )  # sample posterior

        if plot_trace:  # plot trace
            az.plot_trace(trace)

        if plot_posterior:  # plot posterior
            az.plot_posterior(trace, hdi_prob=0.95)

        summary = az.summary(trace, hdi_prob=0.95)  # compute summary

        if print_summary:  # print summary
            print(summary)

        return m, trace


# class ExploitativeSampler(FlYMazeAgent):
#     def init_variables(self, epsilon, delta, w, rho):
#         """
#     Initialize the variables for the Exploitative Sampler agent.
#     ============================================================

#     Parameters:
#     epsilon: asymptotic exploration rate (float)
#     delta: sensitivity to the length of the experiment (float)
#     K: maximum number of histories to sample (int)
#     w: weighting of the regressive term (float)
#     rho: diminishing sensitivity factor (float)
#     recency: should the most recent history be used (bool)
#     """
#         self.epsilon = epsilon
#         self.delta = delta
#         self.K = 15
#         self.w = w
#         self.rho = rho
#         self.recency = True
#         self.action_history = []
#         self.reward_history = []
#         self.trial_step_count = 1

#     def get_subjective_value(self, x, action, w, rho, r_history, a_history):
#         Rx = (1 - w) * x + w * np.mean(r_history[a_history == action])
#         Vt = np.mean(np.abs(np.diff(r_history)))
#         at = (1 + Vt) ** (-rho)
#         sv = np.sign(Rx) * np.abs(Rx) ** at
#         return sv

#     def trial_step(self, state):

#         r_history = np.array(self.reward_history)
#         a_history = np.array(self.action_history)

#         pexplore = self.epsilon ** (
#             (self.trial_step_count - 1) / (self.trial_step_count + self.env.n_trials_per_session ** self.delta)
#         )
#         mt = np.random.choice(range(1, self.K))
#         if self.recency:
#             mean_sv = [
#                 (
#                     np.mean(
#                         [
#                             self.get_subjective_value(x, action, self.w, self.rho, r_history, a_history)
#                             for x in np.concatenate(
#                                 [
#                                     [r_history[a_history == action][-1]],
#                                     np.random.choice(r_history[a_history == action], mt - 1, replace=True),
#                                 ]
#                             )
#                         ]
#                     )
#                     if r_history[a_history == action].shape[0] != 0
#                     else 0
#                 )
#                 for action in np.arange(self.action_space_size)
#             ]
#         else:
#             mean_sv = [
#                 (
#                     np.mean(
#                         [
#                             self.get_subjective_value(x, action, self.w, self.rho, r_history, a_history)
#                             for x in np.random.choice(r_history[a_history == action], mt, replace=True)
#                         ]
#                     )
#                     if r_history[a_history == action].shape[0] != 0
#                     else 0
#                 )
#                 for action in np.arange(self.action_space_size)
#             ]

#         action_prob = pexplore * 1 / self.action_space_size + (1 - pexplore) * (np.argmax(mean_sv) == 1)
#         action = np.random.choice([0, 1], p=[1 - action_prob, action_prob])
#         # if np.random.uniform() < pexplore:
#         #     ## EXPLORE MODE ##
#         #     action = np.random.choice(np.arange(self.action_space_size))
#         # else:
#         #     ## EXPLOIT MODE ##
#         #     mt = np.random.choice(range(1, self.K))
#         #     if self.recency:
#         #         mean_sv = [
#         #             (
#         #                 np.mean(
#         #                     [
#         #                         self.get_subjective_value(x, action, self.w, self.rho, r_history, a_history)
#         #                         for x in np.concatenate(
#         #                             [
#         #                                 [r_history[a_history == action][-1]],
#         #                                 np.random.choice(r_history[a_history == action], mt - 1, replace=True),
#         #                             ]
#         #                         )
#         #                     ]
#         #                 )
#         #                 if r_history[a_history == action].shape[0] != 0
#         #                 else 0
#         #             )
#         #             for action in np.arange(self.action_space_size)
#         #         ]
#         #     else:
#         #         mean_sv = [
#         #             (
#         #                 np.mean(
#         #                     [
#         #                         self.get_subjective_value(x, action, self.w, self.rho, r_history, a_history)
#         #                         for x in np.random.choice(r_history[a_history == action], mt, replace=True)
#         #                     ]
#         #                 )
#         #                 if r_history[a_history == action].shape[0] != 0
#         #                 else 0
#         #             )
#         #             for action in np.arange(self.action_space_size)
#         #         ]
#         #     action = np.argmax(mean_sv)

#         new_state, reward, done, _ = self.env.step(action)

#         self.reward_history.append(reward)
#         self.action_history.append(action)

#         if action == self.biased_action:
#             self.bias += 1 / self.env.n_trials_per_session  # update bias estimate

#         self.trial_step_count += 1
#         return new_state, done

#     def run_episode(self):
#         state = self.env.reset()  # reset environment
#         done = False
#         while not done:
#             state, done = self.trial_step(state)  # trial step

#     def reset_variables(self):
#         self.action_history = []
#         self.reward_history = []
#         self.trial_step_count = 1

#     def vectorizedActionProbabilities(self, epsilon, delta, w, rho, actions_set, rewards_set):
#         probabilities = []
#         for (actions, rewards) in zip(actions_set, rewards_set):
#             action_prob = []
#             for i in range(len(actions)):
#                 a_history = np.array(actions[:i])
#                 r_history = np.array(rewards[:i])
#                 pexplore = epsilon ** (i / (i + 1 + self.env.n_trials_per_session ** delta))
#                 mt = np.random.choice(range(1, self.K))
#                 if self.recency:
#                     mean_sv = [
#                         (
#                             np.mean(
#                                 [
#                                     self.get_subjective_value(x, action, w, rho, r_history, a_history)
#                                     for x in np.concatenate(
#                                         [
#                                             [r_history[a_history == action][-1]],
#                                             np.random.choice(r_history[a_history == action], mt - 1, replace=True),
#                                         ]
#                                     )
#                                 ]
#                             )
#                             if r_history[a_history == action].shape[0] != 0
#                             else 0
#                         )
#                         for action in np.arange(self.action_space_size)
#                     ]
#                 else:
#                     mean_sv = [
#                         (
#                             np.mean(
#                                 [
#                                     self.get_subjective_value(x, action, w, rho, r_history, a_history)
#                                     for x in np.random.choice(r_history[a_history == action], mt, replace=True)
#                                 ]
#                             )
#                             if r_history[a_history == action].shape[0] != 0
#                             else 0
#                         )
#                         for action in np.arange(self.action_space_size)
#                     ]
#                 action_prob.append(pexplore * 1 / self.action_space_size + (1 - pexplore) * (np.argmax(mean_sv) == 1))
#             probabilities.append(action_prob)
#         action_probabilities = np.concatenate(probabilities, axis=0)
#         return action_probabilities

#     def fit(
#         self,
#         actions_set,
#         rewards_set,
#         epsilon_params=(1, 1),
#         delta_params=(1.25,),
#         w_params=(1, 1),
#         rho_params=(1.25,),
#         niters=1000,
#         nchains=2,
#         nparallel=1,
#         plot_trace=True,
#         plot_posterior=True,
#         print_summary=True,
#     ):
#         with pm.Model() as m:  # define model
#             epsilon = pm.Beta("epsilon", epsilon_params[0], epsilon_params[1])
#             delta = pm.HalfNormal("delta", delta_params[0])
#             w = pm.Beta("w", w_params[0], w_params[1])
#             rho = pm.HalfNormal("rho", rho_params[0])

#             action_probabilities = self.vectorizedActionProbabilities(
#                 epsilon, delta, w, rho, actions_set, rewards_set
#             )  # action probabilities
#             like = pm.Bernoulli(
#                 "like", p=action_probabilities, observed=np.concatenate([a for a in actions_set])
#             )  # Bernoulli likelihood
#             trace = pm.sample(draws=niters, chains=nchains, cores=nparallel, return_inferencedata=True)  # sample posterior

#         if plot_trace:  # plot trace
#             az.plot_trace(trace)

#         if plot_posterior:  # plot posterior
#             az.plot_posterior(trace, hdi_prob=0.95)

#         summary = az.summary(trace, hdi_prob=0.95)  # compute summary

#         if print_summary:  # print summary
#             print(summary)

#         return m, trace


# class ACT_R(FlYMazeAgent):
#     def init_variables(self, s, tau, d, default_value):
#         """
#     Initialize the variables for the CATIE Learner
#     ==============================================

#     Parameters:
#     """
#         self.d = d
#         self.K = 2
#         self.s = s
#         self.default_value = default_value
#         self.tau = tau
#         self.action_history = []
#         self.reward_history = []
#         self.memory = []
#         self.trial_step_count = 1

#     def update_memory(self, context, action, reward):
#         if (context, action, reward) not in [(i["context"], i["action"], i["reward"]) for i in self.memory]:
#             self.memory.append(
#                 {"context": context, "action": action, "reward": reward, "used_trials": [self.trial_step_count]}
#             )
#         else:
#             index = np.argmax(
#                 [(context, action, reward) == (i["context"], i["action"], i["reward"]) for i in self.memory]
#             )
#             self.memory[index]["used_trials"].append(self.trial_step_count)

#     def recall(self, context):
#         relevant_memory = [i for i in range(len(self.memory)) if self.memory[i]["context"] == context]

#         if len(relevant_memory) == 0:
#             action = np.random.choice(self.action_space_size)
#             return action

#         activations = [
#             np.log(np.sum((self.trial_step_count - np.array(i["used_trials"])) ** (-self.d)))
#             + np.random.logistic(0, self.s)
#             for i in self.memory
#         ]

#         recalled_memories = []
#         recalled_weights = []
#         for i in range(len(relevant_memory)):
#             if activations[relevant_memory[i]] > self.tau:
#                 recalled_memories.append(self.memory[relevant_memory[i]])
#                 recalled_weights.append(np.exp(activations[relevant_memory[i]] / (np.sqrt(2) * self.s)))

#         if len(recalled_memories) == 0:
#             return np.random.choice(self.action_space_size)

#         probabilities = [
#             [
#                 recalled_weights[i]
#                 / np.sum(
#                     [
#                         recalled_weights[j]
#                         for j in range(len(recalled_memories))
#                         if recalled_memories[j]["action"] == action
#                     ]
#                 )
#                 if recalled_memories[i]["action"] == action
#                 else 0
#                 for i in range(len(recalled_memories))
#             ]
#             for action in range(self.action_space_size)
#         ]

#         values = [
#             np.sum(
#                 [
#                     recalled_memories[i]["reward"] * probabilities[action][i]
#                     for i in range(len(recalled_memories))
#                     if recalled_memories[i]["action"] == action
#                 ]
#             )
#             if np.any([i["action"] == action for i in recalled_memories])
#             else self.default_value
#             for action in range(self.action_space_size)
#         ]

#         if len(np.unique(values)) == 1:
#             return np.random.choice(self.action_space_size)

#         return np.argmax(values)

#     def trial_step(self, state):

#         context = self.reward_history[-self.K :]
#         context = [0] * (self.K - len(context)) + context

#         action = self.recall(context)

#         new_state, reward, done, _ = self.env.step(action)

#         self.reward_history.append(reward)
#         self.action_history.append(action)

#         self.update_memory(context, action, reward)

#         self.trial_step_count += 1

#         if action == self.biased_action:
#             self.bias += 1 / self.env.n_trials_per_session  # update bias estimate

#         return new_state, done

#     def run_episode(self):
#         state = self.env.reset()  # reset environment
#         done = False
#         while not done:
#             state, done = self.trial_step(state)  # trial step

#     def reset_variables(self):
#         self.action_history = []
#         self.reward_history = []
#         self.memory = []

#     def vectorizedActionProbabilities(self, s, tau, d, default_value, actions_set, rewards_set):
#         pass


# class InstanceBasedLearner(FlYMazeAgent):
#     def init_variables(self, pInertia, sigma, d, default_value):
#         """
#     Initialize the variables for the CATIE Learner
#     ==============================================

#     Parameters:
#     """
#         self.d = d
#         self.pInertia = pInertia
#         self.sigma = sigma
#         self.default_value = default_value
#         self.action_history = []
#         self.reward_history = []
#         self.memory = []
#         self.trial_step_count = 1

#     def update_memory(self, action, reward):
#         if (action, reward) not in [(i["action"], i["reward"]) for i in self.memory]:
#             self.memory.append({"action": action, "reward": reward, "used_trials": [self.trial_step_count]})
#         else:
#             index = np.argmax([(action, reward) == (i["action"], i["reward"]) for i in self.memory])
#             self.memory[index]["used_trials"].append(self.trial_step_count)

#     def choose_action(self):
#         if self.trial_step_count == 1:
#             return np.random.choice(self.action_space_size)

#         if np.random.uniform() < self.pInertia:
#             return self.action_history[-1]
#         else:
#             A = [
#                 np.log(np.sum((self.trial_step_count - np.array(i["used_trials"])) ** (-self.d)))
#                 + np.random.logistic(0, self.sigma)
#                 for i in self.memory
#             ]
#             values = []
#             for action in range(self.action_space_size):
#                 if action in [i["action"] for i in self.memory]:
#                     tau = np.sqrt(2) * self.sigma
#                     exps = np.array(
#                         [np.exp(A[i] / tau) for i in range(len(self.memory)) if self.memory[i]["action"] == action]
#                     )
#                     rewards = np.array(
#                         [
#                             self.memory[i]["reward"]
#                             for i in range(len(self.memory))
#                             if self.memory[i]["action"] == action
#                         ]
#                     )
#                     exps = np.append(
#                         exps,
#                         np.exp((np.log(self.trial_step_count ** (-self.d)) + np.random.logistic(0, self.sigma)) / tau),
#                     )
#                     rewards = np.append(rewards, self.default_value)
#                     p = exps / np.sum(exps)
#                     # print(action,p,rewards)
#                     values.append(np.sum(rewards * p))
#                 else:
#                     values.append(self.default_value)
#             # print("Values:",values)
#             return np.argmax(values)

#     def trial_step(self, state):

#         action = self.choose_action()

#         new_state, reward, done, _ = self.env.step(action)

#         self.reward_history.append(reward)
#         self.action_history.append(action)

#         self.update_memory(action, reward)

#         self.trial_step_count += 1

#         if action == self.biased_action:
#             self.bias += 1 / self.env.n_trials_per_session  # update bias estimate

#         return new_state, done

#     def run_episode(self):
#         state = self.env.reset()  # reset environment
#         done = False
#         while not done:
#             state, done = self.trial_step(state)  # trial step

#     def reset_variables(self):
#         self.action_history = []
#         self.reward_history = []
#         self.memory = []

#     def vectorizedActionProbabilities(self, pInertia, sigma, d, default_value, actions_set, rewards_set):
#         pass
