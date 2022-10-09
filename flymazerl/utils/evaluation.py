global FLYMAZERL_PATH

import numpy as np
from multiprocessing import Pool
import os
import time
import pandas as pd

# get FlYMAZERL PATH from environment variable
try:
    FLYMAZERL_PATH = os.environ["FLYMAZERL_PATH"]
    # replace backslashes with forward slashes
    FLYMAZERL_PATH = FLYMAZERL_PATH.replace("\\", "/")
    # add a trailing slash if not present
    if FLYMAZERL_PATH[-1] != "/":
        FLYMAZERL_PATH += "/"
except KeyError:
    raise Exception("FLYMAZERL_PATH environment variable not set.")


def get_agent_history(env, agentClass, params=None, policy_params=None, include_reward=False, dataset="rajagopalan"):
    """
    Simulate an agent and return its action and reward history.
    ===========================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)
    include_reward: whether to include reward history (bool)
    dataset: dataset to use (str: "rajagopalan" or "mohanta")

    Returns:
    action_history: action history (np.array)
    """
    if dataset == "rajagopalan":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_rajagopalan.csv")
    elif dataset == "mohanta":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_mohanta.csv")
    else:
        raise ValueError("dataset must be either 'rajagopalan' or 'mohanta'")

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


def get_schedule_histories(
    env,
    agentClass,
    n_agents,
    params=None,
    policy_params=None,
    parallelize=False,
    include_reward=False,
    dataset="rajagopalan",
):
    """
    Simulate n_agents agents in parallelize and return their action histories.
    =======================================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    n_agents: number of agents (int)
    params: parameters for the agent (dict or list of dicts)
    policy_params: parameters for the policy (dict or list of dicts)
    parallelize: whether to parallelize (bool)
    include_reward: whether to include reward history (bool)
    dataset: dataset to use (str: "rajagopalan" or "mohanta")

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
                        get_agent_history,
                        [(env, agentClass, params[i], policy_params[i], True, dataset) for i in range(n_agents)],
                    )
                    action_histories = [histories[i][0] for i in range(n_agents)]
                    reward_histories = [histories[i][1] for i in range(n_agents)]
            else:
                action_histories = []
                reward_histories = []
                for i in range(n_agents):
                    history = get_agent_history(env, agentClass, params[i], policy_params[i], True, dataset)
                    action_histories.append(history[0])
                    reward_histories.append(history[1])
        else:
            if parallelize:
                with Pool(n_agents) as p:
                    histories = p.starmap(
                        get_agent_history, [(env, agentClass, params, policy_params, True, dataset)] * n_agents
                    )
                    action_histories = [histories[i][0] for i in range(n_agents)]
                    reward_histories = [histories[i][1] for i in range(n_agents)]
            else:
                action_histories = []
                reward_histories = []
                for i in range(n_agents):
                    history = get_agent_history(env, agentClass, params, policy_params, True, dataset)
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
                        get_agent_history,
                        [(env, agentClass, params[i], policy_params[i], False, dataset) for i in range(n_agents)],
                    )
            else:
                action_histories = []
                for i in range(n_agents):
                    action_history = get_agent_history(env, agentClass, params[i], policy_params[i], False, dataset)
                    action_histories.append(action_history)
        else:
            if parallelize:
                with Pool(n_agents) as p:
                    action_histories = p.starmap(
                        get_agent_history, [(env, agentClass, params, policy_params, False, dataset)] * n_agents
                    )
            else:
                action_histories = []
                for i in range(n_agents):
                    action_history = get_agent_history(env, agentClass, params, policy_params, False, dataset)
                    action_histories.append(action_history)
        return action_histories


def get_agent_bias(env, agentClass, params=None, policy_params=None, dataset="rajagopalan"):
    """
    Simulate an agent and return its bias estimate.
    ===============================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)
    dataset: dataset to use (str: "rajagopalan" or "mohanta")

    Returns:
    bias: bias estimate (float)
    """
    if dataset == "rajagopalan":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_rajagopalan.csv")
    elif dataset == "mohanta":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_mohanta.csv")
    else:
        raise ValueError("dataset must be either 'rajagopalan' or 'mohanta'")

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


def get_agent_performance(env, agentClass, params=None, policy_params=None, dataset="rajagopalan"):
    """
    Simulate an agent and return its fraction of trials rewarded.
    =============================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)
    dataset: dataset to use (str: "rajagopalan" or "mohanta")

    Returns:
    performance: fraction of trials rewarded (float)
    """
    if dataset == "rajagopalan":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_rajagopalan.csv")
    elif dataset == "mohanta":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_mohanta.csv")
    else:
        raise ValueError("dataset must be either 'rajagopalan' or 'mohanta'")

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


def get_agent_failure(env, agentClass, params=None, policy_params=None, dataset="rajagopalan"):
    """
    Simulate an agent and return its fraction of trials unrewarded
    ==============================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)
    dataset: dataset to use (str: "rajagopalan" or "mohanta")

    Returns:
    performance: fraction of trials rewarded (float)
    """
    if dataset == "rajagopalan":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_rajagopalan.csv")
    elif dataset == "mohanta":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_mohanta.csv")
    else:
        raise ValueError("dataset must be either 'rajagopalan' or 'mohanta'")

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
    return 1 - np.mean(agent.reward_history)


def get_agent_separation(
    env,
    agentClass1,
    agentClass2,
    params1=None,
    params2=None,
    policy_params1=None,
    policy_params2=None,
    dataset="rajagopalan",
):
    """
    Simulate two agents and return their difference in actions chosen
    =================================================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)
    dataset: dataset to use (str: "rajagopalan" or "mohanta")

    Returns:
    performance: fraction of trials rewarded (float)
    """
    if dataset == "rajagopalan":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_rajagopalan.csv")
    elif dataset == "mohanta":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_mohanta.csv")
    else:
        raise ValueError("dataset must be either 'rajagopalan' or 'mohanta'")

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
    difference = np.mean(np.power(np.array(agent1.action_history) - np.array(agent2.action_history), 2))
    return difference


def get_schedule_fitness(
    env,
    agentClass,
    n_agents,
    params=None,
    policy_params=None,
    parallelize=False,
    fitness_type="bias",
    ref_agentClass=None,
    ref_params=None,
    ref_policy_params=None,
    dataset="rajagopalan",
):
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
    ref_agentClass: class of reference agent (flymazerl Agent)
    ref_params: parameters for the reference agent (dict)
    ref_policy_params: parameters for the reference policy (dict)
    dataset: dataset to use (str: "rajagopalan" or "mohanta")

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
                        get_agent_bias,
                        [(env, agentClass, params[i], policy_params[i], dataset) for i in range(n_agents)],
                    )
                fitness = np.mean(bias_pool)
            elif fitness_type == "performance":
                with Pool(n_agents) as p:
                    performance_pool = p.starmap(
                        get_agent_performance,
                        [(env, agentClass, params[i], policy_params[i], dataset) for i in range(n_agents)],
                    )
                fitness = np.mean(performance_pool)
            elif fitness_type == "failure":
                with Pool(n_agents) as p:
                    failure_pool = p.starmap(
                        get_agent_failure,
                        [(env, agentClass, params[i], policy_params[i], dataset) for i in range(n_agents)],
                    )
                fitness = np.mean(failure_pool)
            elif fitness_type == "separation":
                assert ref_agentClass is not None, "ref_agentClass must be specified for separation"
                if isinstance(ref_params, list) and isinstance(ref_policy_params, list):
                    with Pool(n_agents) as p:
                        separation_pool = p.starmap(
                            get_agent_separation,
                            [
                                (
                                    env,
                                    agentClass,
                                    ref_agentClass,
                                    params[i],
                                    ref_params[i],
                                    policy_params[i],
                                    ref_policy_params[i],
                                    dataset,
                                )
                                for i in range(n_agents)
                            ],
                        )
                else:
                    with Pool(n_agents) as p:
                        separation_pool = p.starmap(
                            get_agent_separation,
                            [
                                (
                                    env,
                                    agentClass,
                                    ref_agentClass,
                                    params[i],
                                    ref_params,
                                    policy_params[i],
                                    ref_policy_params,
                                    dataset,
                                )
                                for i in range(n_agents)
                            ],
                        )
                fitness = np.mean(separation_pool)
            else:
                raise ValueError("Unknown fitness type. Must be 'bias', 'performance', 'failure', or 'separation'")
        else:
            if fitness_type == "bias":
                bias_pool = []
                for i in range(n_agents):
                    bias = get_agent_bias(env, agentClass, params[i], policy_params[i], dataset)
                    bias_pool.append(bias)
                fitness = np.mean(bias_pool)
            elif fitness_type == "performance":
                performance_pool = []
                for i in range(n_agents):
                    performance = get_agent_performance(env, agentClass, params[i], policy_params[i], dataset)
                    performance_pool.append(performance)
                fitness = np.mean(performance_pool)
            elif fitness_type == "failure":
                failure_pool = []
                for i in range(n_agents):
                    failure = get_agent_failure(env, agentClass, params[i], policy_params[i], dataset)
                    failure_pool.append(failure)
                fitness = np.mean(failure_pool)
            elif fitness_type == "separation":
                assert ref_agentClass is not None, "ref_agentClass must be specified for separation"
                if isinstance(ref_params, list) and isinstance(ref_policy_params, list):
                    separation_pool = []
                    for i in range(n_agents):
                        separation = get_agent_separation(
                            env,
                            agentClass,
                            ref_agentClass,
                            params[i],
                            ref_params[i],
                            policy_params[i],
                            ref_policy_params[i],
                            dataset,
                        )
                        separation_pool.append(separation)
                else:
                    separation_pool = []
                    for i in range(n_agents):
                        separation = get_agent_separation(
                            env,
                            agentClass,
                            ref_agentClass,
                            params[i],
                            ref_params,
                            policy_params[i],
                            ref_policy_params,
                            dataset,
                        )
                        separation_pool.append(separation)
                fitness = np.mean(separation_pool)
            else:
                raise ValueError("Unknown fitness type. Must be 'bias', 'performance', 'failure', or 'separation'")
    else:
        if parallelize:
            if fitness_type == "bias":
                with Pool(n_agents) as p:
                    bias_pool = p.starmap(
                        get_agent_bias, [(env, agentClass, params, policy_params, dataset)] * n_agents
                    )
                fitness = np.mean(bias_pool)
            elif fitness_type == "performance":
                with Pool(n_agents) as p:
                    performance_pool = p.starmap(
                        get_agent_performance, [(env, agentClass, params, policy_params, dataset)] * n_agents
                    )
                fitness = np.mean(performance_pool)
            elif fitness_type == "failure":
                with Pool(n_agents) as p:
                    failure_pool = p.starmap(
                        get_agent_failure, [(env, agentClass, params, policy_params, dataset)] * n_agents
                    )
                fitness = np.mean(failure_pool)
            elif fitness_type == "separation":
                assert ref_agentClass is not None, "ref_agentClass must be specified for separation"
                if isinstance(ref_params, list) and isinstance(ref_policy_params, list):
                    with Pool(n_agents) as p:
                        separation_pool = p.starmap(
                            get_agent_separation,
                            [
                                (
                                    env,
                                    agentClass,
                                    ref_agentClass,
                                    params,
                                    ref_params[i],
                                    policy_params,
                                    ref_policy_params[i],
                                    dataset,
                                )
                                for i in range(n_agents)
                            ],
                        )
                else:
                    with Pool(n_agents) as p:
                        separation_pool = p.starmap(
                            get_agent_separation,
                            [
                                (
                                    env,
                                    agentClass,
                                    ref_agentClass,
                                    params,
                                    ref_params,
                                    policy_params,
                                    ref_policy_params,
                                    dataset,
                                )
                                for i in range(n_agents)
                            ],
                        )
                fitness = np.mean(separation_pool)
            else:
                raise ValueError("Unknown fitness type. Must be 'bias', 'performance', 'failure', or 'separation'")
        else:
            if fitness_type == "bias":
                bias_pool = []
                for i in range(n_agents):
                    bias = get_agent_bias(env, agentClass, params, policy_params, dataset)
                    bias_pool.append(bias)
                fitness = np.mean(bias_pool)
            elif fitness_type == "performance":
                performance_pool = []
                for i in range(n_agents):
                    performance = get_agent_performance(env, agentClass, params, policy_params, dataset)
                    performance_pool.append(performance)
                fitness = np.mean(performance_pool)
            elif fitness_type == "failure":
                failure_pool = []
                for i in range(n_agents):
                    failure = get_agent_failure(env, agentClass, params, policy_params, dataset)
                    failure_pool.append(failure)
                fitness = np.mean(failure_pool)
            elif fitness_type == "separation":
                assert ref_agentClass is not None, "ref_agentClass must be specified for separation"
                if isinstance(ref_params, list) and isinstance(ref_policy_params, list):
                    separation_pool = []
                    for i in range(n_agents):
                        separation = get_agent_separation(
                            env,
                            agentClass,
                            ref_agentClass,
                            params,
                            ref_params[i],
                            policy_params,
                            ref_policy_params[i],
                            dataset,
                        )
                        separation_pool.append(separation)
                else:
                    separation_pool = []
                    for i in range(n_agents):
                        separation = get_agent_separation(
                            env,
                            agentClass,
                            ref_agentClass,
                            params,
                            ref_params,
                            policy_params,
                            ref_policy_params,
                            dataset,
                        )
                        separation_pool.append(separation)
                fitness = np.mean(separation_pool)
            else:
                raise ValueError("Unknown fitness type. Must be 'bias', 'performance', 'failure', or 'separation'")
    return fitness


def get_agent_value_history(env, agentClass, params=None, policy_params=None, dataset="rajagopalan"):
    """
    Simulate an agent and return its value history
    ==============================================

    Parameters:
    env: environment (ymaze_static environment)
    agentClass: class of agent (flymazerl Agent)
    params: parameters for the agent (dict)
    policy_params: parameters for the policy (dict)
    dataset: dataset to use for the environment (str: "rajagopalan" or "mohanta")

    Returns:
    value_history: value history (np.array)
    """
    if dataset == "rajagopalan":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_rajagopalan.csv")
    elif dataset == "mohanta":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_mohanta.csv")
    else:
        raise ValueError("dataset must be either 'rajagopalan' or 'mohanta'")

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


def get_schedule_values(
    env, agentClass, n_agents, params=None, policy_params=None, parallelize=False, dataset="rajagopalan"
):
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
    dataset: dataset to use for the environment (str: "rajagopalan" or "mohanta")

    Returns:
    value_history: value history (np.array)
    """
    if isinstance(params, list) and isinstance(policy_params, list):
        assert len(params) == n_agents, "params must be a list of dicts with length n_agents"
        assert len(policy_params) == n_agents, "policy_params must be a list of dicts of length n_agents"
        if parallelize:
            with Pool(n_agents) as p:
                value_histories = p.starmap(
                    get_agent_value_history,
                    [(env, agentClass, params[i], policy_params[i], dataset) for i in range(n_agents)],
                )
        else:
            value_histories = []
            for i in range(n_agents):
                value_history = get_agent_value_history(env, agentClass, params[i], policy_params[i], dataset)
                value_histories.append(value_history)
    else:
        if parallelize:
            with Pool(n_agents) as p:
                value_histories = p.starmap(
                    get_agent_value_history, [(env, agentClass, params, policy_params, dataset)] * n_agents
                )
        else:
            value_histories = []
            for i in range(n_agents):
                value_history = get_agent_value_history(env, agentClass, params, policy_params, dataset)
                value_histories.append(value_history)
    return value_histories
