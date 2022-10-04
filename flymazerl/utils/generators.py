global FLYMAZERL_PATH

import numpy as np
import pandas as pd
import arviz as az
import os
import platform

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

if platform.system() == "Windows":
    model_fits_directory = "Z:/FlYMazeRL_Fits/"
elif platform.system() == "Linux":
    model_fits_directory = "/groups/turner/turnerlab/Rishika/FlYMazeRL_Fits/"


def generate_random_schedule(n_trials_per_session, reward_fraction, exclusive=False):
    """
    Generates a random schedule of n_trials_per_session trials, with reward_fraction of the trials being rewarded for each alternative.
    ===================================================================================================================================

    Parameters:
    -----------
    n_trials_per_session: number of trials per episode (int)
    reward_fraction: fraction of trials that are rewarded for each alternative (float)
    exclusive: whether to only allow multiple alternatives to be rewarded per trial (bool)

    Returns:
    --------
    schedule: randomly generated schedule (np.array)
    """
    schedule = np.zeros((n_trials_per_session, 2))
    schedule[
        np.random.choice(
            np.arange(n_trials_per_session), replace=False, size=int(reward_fraction * n_trials_per_session)
        ),
        0,
    ] = 1
    if exclusive:
        if reward_fraction <= 0.5:
            schedule[
                np.random.choice(
                    np.arange(n_trials_per_session)[schedule[:, 0] == 0],
                    replace=False,
                    size=int(reward_fraction * n_trials_per_session),
                ),
                1,
            ] = 1
        else:
            raise ValueError("reward_fraction must be less than 0.5 when exclusive is True")
    else:
        schedule[
            np.random.choice(
                np.arange(n_trials_per_session), replace=False, size=int(reward_fraction * n_trials_per_session)
            ),
            1,
        ] = 1
    return schedule


def generate_random_schedule_with_blocks(
    n_trials_per_episode, a1_reward_probabilities, a2_reward_probabilities, equal=True, block_sizes=None
):
    """
    Generates a random schedule of n_trials_per_episode trials with dynamic reward probabilities for each alternative that vary across blocks of trials
    ===================================================================================================================================================

    Parameters:
    -----------
    n_trials_per_episode: number of trials per episode (int)
    a1_reward_probabilities: reward probabilities for alternative 1 (list of floats)
    a2_reward_probabilities: reward probabilities for alternative 2 (list of floats)
    """
    if equal:
        assert len(a1_reward_probabilities) == len(
            a2_reward_probabilities
        ), "Number of blocks must be equal on both sides"
        assert (
            n_trials_per_episode % len(a1_reward_probabilities) == 0
        ), "Number of trials per episode must be divisible by number of blocks"

        n_blocks = len(a1_reward_probabilities)
        blocksize = n_trials_per_episode // n_blocks
        schedule = []
        for n in range(n_blocks):
            temp = np.zeros((blocksize, 2))
            temp[:, 0] = np.random.choice(
                2, size=blocksize, p=[1 - a1_reward_probabilities[n], a1_reward_probabilities[n]]
            )
            temp[:, 1] = np.random.choice(
                2, size=blocksize, p=[1 - a2_reward_probabilities[n], a2_reward_probabilities[n]]
            )
            schedule.append(temp)
    else:
        assert len(a1_reward_probabilities) == len(
            a2_reward_probabilities
        ), "Number of blocks must be equal on both sides"
        assert type(block_sizes) == list, "block_sizes must be a list"
        assert np.sum(block_sizes) == n_trials_per_episode, "Number of trials per episode must equal sum of block sizes"

        n_blocks = len(a1_reward_probabilities)
        schedule = []
        for n in range(n_blocks):
            temp = np.zeros((block_sizes[n], 2))
            temp[:, 0] = np.random.choice(
                2, size=block_sizes[n], p=[1 - a1_reward_probabilities[n], a1_reward_probabilities[n]]
            )
            temp[:, 1] = np.random.choice(
                2, size=block_sizes[n], p=[1 - a2_reward_probabilities[n], a2_reward_probabilities[n]]
            )
            schedule.append(temp)

    schedule = np.concatenate(schedule, axis=0)
    return schedule


def generate_params_from_fits(agentClass, n_samples, sample_from_population=True, dataset="rajagopalan"):
    """
    Generates params and policy_params from fitted agent
    ====================================================

    Parameters:
    -----------
    agentClass: the agent class (class)
    n_samples: number of samples to generate (int)
    sample_from_population: whether to sample from the population (bool)
    dataset: the dataset to sample from (str: "rajagopalan" or "mohanta")
    """
    if dataset == "rajagopalan":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_rajagopalan.csv")
    elif dataset == "mohanta":
        model_database = pd.read_csv(FLYMAZERL_PATH + "model_description_mohanta.csv")
    else:
        raise ValueError("dataset must be either 'rajagopalan' or 'mohanta'")

    fit_dir = (
        model_fits_directory + model_database.loc[model_database.AgentClass == agentClass.__name__, "FitDir"].values[0]
    )
    print(fit_dir)
    assert fit_dir is not None, "Fit Directory not found in model_database"
    assert os.path.exists(fit_dir), "Fit Directory not found"
    assert fit_dir.endswith(".nc"), "Fit Directory must be a netcdf file for this function"

    sub_database = model_database.loc[model_database.AgentClass == agentClass.__name__, :]

    params_pool = []
    policy_params_pool = []

    inferenceData = az.from_netcdf(fit_dir)
    if sample_from_population:
        space_size = inferenceData.posterior.dims["chain"] * inferenceData.posterior.dims["draw"]
        samples = np.random.choice(space_size, size=n_samples, replace=False)
        for sample in samples:
            params = {}
            policy_params = {}
            for i in list(inferenceData.posterior.data_vars):
                if sub_database.loc[sub_database["Parameter Name"] == i, "Parameter Type"].values[0] == "learningrule":
                    exec(f"params['{i}'] = inferenceData.posterior.{i}.values.flatten()[{sample}]")
                elif sub_database.loc[sub_database["Parameter Name"] == i, "Parameter Type"].values[0] == "policy":
                    exec(f"policy_params['{i}'] = inferenceData.posterior.{i}.values.flatten()[{sample}]")
            params_pool.append(params)
            policy_params_pool.append(policy_params)
    else:
        for _ in range(n_samples):
            params = {}
            policy_params = {}
            for i in list(inferenceData.posterior.data_vars):
                if sub_database.loc[sub_database["Parameter Name"] == i, "Parameter Type"].values[0] == "learningrule":
                    exec(f"params['{i}'] = np.mean(inferenceData.posterior.{i}.values.flatten())")
                elif sub_database.loc[sub_database["Parameter Name"] == i, "Parameter Type"].values[0] == "policy":
                    exec(f"policy_params['{i}'] = np.mean(inferenceData.posterior.{i}.values.flatten())")
            params_pool.append(params)
            policy_params_pool.append(policy_params)

    del inferenceData
    if n_samples == 1:
        return params_pool[0], policy_params_pool[0]
    else:
        return params_pool, policy_params_pool
