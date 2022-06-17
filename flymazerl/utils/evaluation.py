global model_database

from tkinter import E
import numpy as np
from multiprocessing import Pool
import os
import time
import pandas as pd

model_database = pd.read_csv("https://raw.githubusercontent.com/neurorishika/flymazerl/main/model_description.csv")


def get_agent_history(env, agentClass, params=None, policy_params=None, include_reward=False):
    """
    Simulate an agent and return its action and reward history.
    ===========================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)

    Returns:
    action_history: action history (np.array)
    """
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    if params is None and policy_params is None:
        agent = agentClass(env, history=True)
        fit_dir = model_database.loc[model_database.AgentClass == agentClass.__name__, "FitDir"].values[0]
        assert fit_dir is not None, "FitDir not found in model_database"
        if fit_dir.endswith(".nc"):
            agent.load(fit_dir, sample_from_population=True)
        else:
            agent.load(fit_dir)
    else:
        agent = agentClass(env, params=params, policy_params=policy_params, history=True)
    agent.next_episode()

    if include_reward:
        return agent.action_history, agent.reward_history
    else:
        return agent.action_history


def get_schedule_histories(env, agentClass, n_agents, params=None, policy_params=None, parallelize=False, include_reward=False):
    """
    Simulate n_agents agents in parallelize and return their action histories.
    =======================================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    n_agents: number of agents (int)
    params: parameters for the agent (dict or list of dicts)
    policy_params: parameters for the policy (dict or list of dicts)

    Returns:
    action_histories: action histories (list of np.arrays)
    """
    if include_reward:
        if isinstance(params, list) and isinstance(policy_params, list):
            assert len(params) == n_agents, "params must be a list of dicts with length n_agents"
            assert len(policy_params) == n_agents, "policy_params must be a list of dicts of length n_agents"
            if parallelize:
                with Pool(n_agents) as p:
                    histories = p.starmap(
                        get_agent_history, [(env, agentClass, params[i], policy_params[i], True) for i in range(n_agents)]
                    )
                    action_histories = [histories[i][0] for i in range(n_agents)]
                    reward_histories = [histories[i][1] for i in range(n_agents)]
            else:
                action_histories = []
                reward_histories = []
                for i in range(n_agents):
                    history = get_agent_history(env, agentClass, params[i], policy_params[i], True)
                    action_histories.append(history[0])
                    reward_histories.append(history[1])
        else:
            if parallelize:
                with Pool(n_agents) as p:
                    histories = p.starmap(get_agent_history, [(env, agentClass, params, policy_params, True)] * n_agents)
                    action_histories = [histories[i][0] for i in range(n_agents)]
                    reward_histories = [histories[i][1] for i in range(n_agents)]
            else:
                action_histories = []
                reward_histories = []
                for i in range(n_agents):
                    history = get_agent_history(env, agentClass, params, policy_params, True)
                    action_histories.append(history[0])
                    reward_histories.append(history[1])
        return action_histories, reward_histories
    else:
        if isinstance(params, list) and isinstance(policy_params, list):
            assert len(params) == n_agents, "params must be a list of dicts with length n_agents"
            assert len(policy_params) == n_agents, "policy_params must be a list of dicts of length n_agents"
            if parallelize:
                with Pool(n_agents) as p:
                    action_histories = p.starmap(
                        get_agent_history, [(env, agentClass, params[i], policy_params[i]) for i in range(n_agents)]
                    )
            else:
                action_histories = []
                for i in range(n_agents):
                    action_history = get_agent_history(env, agentClass, params[i], policy_params[i])
                    action_histories.append(action_history)
        else:
            if parallelize:
                with Pool(n_agents) as p:
                    action_histories = p.starmap(get_agent_history, [(env, agentClass, params, policy_params)] * n_agents)
            else:
                action_histories = []
                for i in range(n_agents):
                    action_history = get_agent_history(env, agentClass, params, policy_params)
                    action_histories.append(action_history)
        return action_histories


def get_agent_bias(env, agentClass, params=None, policy_params=None):
    """
    Simulate an agent and return its bias estimate.
    ===============================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)

    Returns:
    bias: bias estimate (float)
    """
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    if params is None and policy_params is None:
        agent = agentClass(env, history=True)
        fit_dir = model_database.loc[model_database.AgentClass == agentClass.__name__, "FitDir"].values[0]
        assert fit_dir is not None, "FitDir not found in model_database"
        if fit_dir.endswith(".nc"):
            agent.load(fit_dir, sample_from_population=True)
        else:
            agent.load(fit_dir)
    else:
        agent = agentClass(env, params=params, policy_params=policy_params, history=True)
    agent.next_episode()
    return agent.get_bias_estimate()

def get_agent_performance(env, agentClass, params=None, policy_params=None):
    """
    Simulate an agent and return its fraction of trials rewarded.
    =============================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)

    Returns:
    performance: fraction of trials rewarded (float)
    """
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    if params is None and policy_params is None:
        agent = agentClass(env, history=True)
        fit_dir = model_database.loc[model_database.AgentClass == agentClass.__name__, "FitDir"].values[0]
        assert fit_dir is not None, "FitDir not found in model_database"
        if fit_dir.endswith(".nc"):
            agent.load(fit_dir, sample_from_population=True)
        else:
            agent.load(fit_dir)
    else:
        agent = agentClass(env, params=params, policy_params=policy_params, history=True)
    agent.next_episode()
    return np.mean(agent.reward_history)

def get_agent_failure(env, agentClass, params=None, policy_params=None):
    """
    Simulate an agent and return its fraction of trials unrewarded
    ==============================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)

    Returns:
    performance: fraction of trials rewarded (float)
    """
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    if params is None and policy_params is None:
        agent = agentClass(env, history=True)
        fit_dir = model_database.loc[model_database.AgentClass == agentClass.__name__, "FitDir"].values[0]
        assert fit_dir is not None, "FitDir not found in model_database"
        if fit_dir.endswith(".nc"):
            agent.load(fit_dir, sample_from_population=True)
        else:
            agent.load(fit_dir)
    else:
        agent = agentClass(env, params=params, policy_params=policy_params, history=True)
    agent.next_episode()
    return 1-np.mean(agent.reward_history)

def get_agent_separation(env, agentClass1, agentClass2, params1=None, params2=None, policy_params1=None, policy_params2=None):
    """
    Simulate two agents and return their difference in actions chosen
    =================================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)

    Returns:
    performance: fraction of trials rewarded (float)
    """
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    if params1 is None and policy_params1 is None:
        agent1 = agentClass1(env, history=True)
        fit_dir = model_database.loc[model_database.AgentClass == agentClass1.__name__, "FitDir"].values[0]
        assert fit_dir is not None, "FitDir not found in model_database"
        if fit_dir.endswith(".nc"):
            agent1.load(fit_dir, sample_from_population=True)
        else:
            agent1.load(fit_dir)
    else:
        agent1 = agentClass1(env, params=params1, policy_params=policy_params1, history=True)
    if params2 is None and policy_params2 is None:
        agent2 = agentClass2(env, history=True)
        fit_dir = model_database.loc[model_database.AgentClass == agentClass2.__name__, "FitDir"].values[0]
        assert fit_dir is not None, "FitDir not found in model_database"
        if fit_dir.endswith(".nc"):
            agent2.load(fit_dir, sample_from_population=True)
        else:
            agent2.load(fit_dir)
    else:
        agent2 = agentClass2(env, params=params2, policy_params=policy_params2, history=True)
    agent1.next_episode()
    agent2.next_episode()
    difference = np.mean(np.power(np.array(agent1.action_history) - np.array(agent2.action_history),2))
    return difference

def get_schedule_fitness(env, agentClass, n_agents, params=None, policy_params=None, parallelize=False, fitness_type="bias", ref_agentClass=None, ref_params=None, ref_policy_params=None):
    """
    Simulate n_agents agents in parallelize and return their action histories.
    =======================================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    n_agents: number of agents (int)
    params: parameters for the agent (dict or list of dicts)
    policy_params: parameters for the policy (dict or list of dicts)
    parallelize: whether to parallelize the simulation (bool)
    fitness_type: type of fitness to return (str) (bias, performance, failure, separation)

    Returns:
    fitness: mean fitness (float)
    """
    if isinstance(params, list) and isinstance(policy_params, list):
        assert len(params) == n_agents, "params must be a list of dicts with length n_agents"
        assert len(policy_params) == n_agents, "policy_params must be a list of dicts of length n_agents"
        if parallelize:
            if fitness_type == "bias":
                with Pool(n_agents) as p:
                    bias_pool = p.starmap(
                        get_agent_bias, [(env, agentClass, params[i], policy_params[i]) for i in range(n_agents)]
                    )
                fitness = np.mean(bias_pool)
            elif fitness_type == "performance":
                with Pool(n_agents) as p:
                    performance_pool = p.starmap(
                        get_agent_performance, [(env, agentClass, params[i], policy_params[i]) for i in range(n_agents)]
                    )
                fitness = np.mean(performance_pool)
            elif fitness_type == "failure":
                with Pool(n_agents) as p:
                    failure_pool = p.starmap(
                        get_agent_failure, [(env, agentClass, params[i], policy_params[i]) for i in range(n_agents)]
                    )
                fitness = np.mean(failure_pool)
            elif fitness_type == "separation":
                assert ref_agentClass is not None, "ref_agentClass must be specified for separation"
                if isinstance(ref_params,list) and isinstance(ref_policy_params,list):
                    with Pool(n_agents) as p:
                        separation_pool = p.starmap(
                            get_agent_separation, [(env, agentClass, ref_agentClass, params[i], ref_params[i], policy_params[i], ref_policy_params[i]) for i in range(n_agents)]
                        )
                else:
                    with Pool(n_agents) as p:
                        separation_pool = p.starmap(
                            get_agent_separation, [(env, agentClass, ref_agentClass, params[i], ref_params, policy_params[i], ref_policy_params) for i in range(n_agents)]
                        )
                fitness = np.mean(separation_pool)
            else:
                raise ValueError("Unknown fitness type. Must be 'bias', 'performance', 'failure', or 'separation'")
        else:
            if fitness_type == "bias":
                bias_pool = []
                for i in range(n_agents):
                    bias = get_agent_bias(env, agentClass, params[i], policy_params[i])
                    bias_pool.append(bias)
                fitness = np.mean(bias_pool)
            elif fitness_type == "performance":
                performance_pool = []
                for i in range(n_agents):
                    performance = get_agent_performance(env, agentClass, params[i], policy_params[i])
                    performance_pool.append(performance)
                fitness = np.mean(performance_pool)
            elif fitness_type == "failure":
                failure_pool = []
                for i in range(n_agents):
                    failure = get_agent_failure(env, agentClass, params[i], policy_params[i])
                    failure_pool.append(failure)
                fitness = np.mean(failure_pool)
            elif fitness_type == "separation":
                assert ref_agentClass is not None, "ref_agentClass must be specified for separation"
                if isinstance(ref_params,list) and isinstance(ref_policy_params,list):
                    separation_pool = []
                    for i in range(n_agents):
                        separation = get_agent_separation(env, agentClass, ref_agentClass, params[i], ref_params[i], policy_params[i], ref_policy_params[i])
                        separation_pool.append(separation)
                else:
                    separation_pool = []
                    for i in range(n_agents):
                        separation = get_agent_separation(env, agentClass, ref_agentClass, params[i], ref_params, policy_params[i], ref_policy_params)
                        separation_pool.append(separation)
                fitness = np.mean(separation_pool)
            else:
                raise ValueError("Unknown fitness type. Must be 'bias', 'performance', 'failure', or 'separation'")
    else:
        if parallelize:
            if fitness_type == "bias":
                with Pool(n_agents) as p:
                    bias_pool = p.starmap(get_agent_bias, [(env, agentClass, params, policy_params)] * n_agents)
                fitness = np.mean(bias_pool)
            elif fitness_type == "performance":
                with Pool(n_agents) as p:
                    performance_pool = p.starmap(get_agent_performance, [(env, agentClass, params, policy_params)] * n_agents)
                fitness = np.mean(performance_pool)
            elif fitness_type == "failure":
                with Pool(n_agents) as p:
                    failure_pool = p.starmap(get_agent_failure, [(env, agentClass, params, policy_params)] * n_agents)
                fitness = np.mean(failure_pool)
            elif fitness_type == "separation":
                assert ref_agentClass is not None, "ref_agentClass must be specified for separation"
                if isinstance(ref_params,list) and isinstance(ref_policy_params,list):
                    with Pool(n_agents) as p:
                        separation_pool = p.starmap(
                            get_agent_separation, [(env, agentClass, ref_agentClass, params, ref_params[i], policy_params, ref_policy_params[i]) for i in range(n_agents)]
                        )
                else:
                    with Pool(n_agents) as p:
                        separation_pool = p.starmap(
                            get_agent_separation, [(env, agentClass, ref_agentClass, params, ref_params, policy_params, ref_policy_params) for i in range(n_agents)]
                        )
                fitness = np.mean(separation_pool)
            else:
                raise ValueError("Unknown fitness type. Must be 'bias', 'performance', 'failure', or 'separation'")
        else:
            if fitness_type == "bias":
                bias_pool = []
                for i in range(n_agents):
                    bias = get_agent_bias(env, agentClass, params, policy_params)
                    bias_pool.append(bias)
                fitness = np.mean(bias_pool)
            elif fitness_type == "performance":
                performance_pool = []
                for i in range(n_agents):
                    performance = get_agent_performance(env, agentClass, params, policy_params)
                    performance_pool.append(performance)
                fitness = np.mean(performance_pool)
            elif fitness_type == "failure":
                failure_pool = []
                for i in range(n_agents):
                    failure = get_agent_failure(env, agentClass, params, policy_params)
                    failure_pool.append(failure)
                fitness = np.mean(failure_pool)
            elif fitness_type == "separation":
                assert ref_agentClass is not None, "ref_agentClass must be specified for separation"
                if isinstance(ref_params,list) and isinstance(ref_policy_params,list):
                    separation_pool = []
                    for i in range(n_agents):
                        separation = get_agent_separation(env, agentClass, ref_agentClass, params, ref_params[i], policy_params, ref_policy_params[i])
                        separation_pool.append(separation)
                else:
                    separation_pool = []
                    for i in range(n_agents):
                        separation = get_agent_separation(env, agentClass, ref_agentClass, params, ref_params, policy_params, ref_policy_params)
                        separation_pool.append(separation)
                fitness = np.mean(separation_pool)
            else:
                raise ValueError("Unknown fitness type. Must be 'bias', 'performance', 'failure', or 'separation'")
    return fitness

def get_agent_value_history(env, agentClass, params=None, policy_params=None):
    """
    Simulate an agent and return its value history
    ==============================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)

    Returns:
    value_history: value history (np.array)
    """
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    if params is None and policy_params is None:
        agent = agentClass(env, history=True)
        fit_dir = model_database.loc[model_database.AgentClass == agentClass.__name__, "FitDir"].values[0]
        assert fit_dir is not None, "FitDir not found in model_database"
        if fit_dir.endswith(".nc"):
            agent.load(fit_dir, sample_from_population=True)
        else:
            agent.load(fit_dir)
    else:
        agent = agentClass(env, params=params, policy_params=policy_params, history=True)
    agent.next_episode()
    if hasattr(agent, "q_history"):
        return agent.q_history
    else:
        raise ValueError("Agent does not have q_history")

def get_schedule_values(env, agentClass, n_agents, params=None, policy_params=None, parallelize=False):
    """
    Simulate n_agents agents in parallelize and return their action histories.
    =======================================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    n_agents: number of agents (int)
    params: parameters for the agent (dict or list of dicts)
    policy_params: parameters for the policy (dict or list of dicts)

    Returns:
    value_history: value history (np.array)
    """
    if isinstance(params, list) and isinstance(policy_params, list):
        assert len(params) == n_agents, "params must be a list of dicts with length n_agents"
        assert len(policy_params) == n_agents, "policy_params must be a list of dicts of length n_agents"
        if parallelize:
            with Pool(n_agents) as p:
                value_histories = p.starmap(
                    get_agent_value_history, [(env, agentClass, params[i], policy_params[i]) for i in range(n_agents)]
                )
        else:
            value_histories = []
            for i in range(n_agents):
                value_history = get_agent_value_history(env, agentClass, params[i], policy_params[i])
                value_histories.append(value_history)
    else:
        if parallelize:
            with Pool(n_agents) as p:
                value_histories = p.starmap(get_agent_value_history, [(env, agentClass, params, policy_params)] * n_agents)
        else:
            value_histories = []
            for i in range(n_agents):
                value_history = get_agent_history(env, agentClass, params, policy_params)
                value_histories.append(value_history)
    return value_histories