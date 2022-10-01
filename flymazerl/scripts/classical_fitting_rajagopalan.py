import argparse
import datetime
import pickle
from subprocess import call
import os
import requests
import io

from flymazerl.agents.classical import *
from flymazerl.agents.phenomenological import *
from flymazerl.gym.environment import *

import numpy as np
import pandas as pd

start_text = """
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
███████╗██╗     ██╗   ██╗███╗   ███╗ █████╗ ███████╗███████╗██████╗ ██╗     
██╔════╝██║     ╚██╗ ██╔╝████╗ ████║██╔══██╗╚══███╔╝██╔════╝██╔══██╗██║     
█████╗  ██║      ╚████╔╝ ██╔████╔██║███████║  ███╔╝ █████╗  ██████╔╝██║     
██╔══╝  ██║       ╚██╔╝  ██║╚██╔╝██║██╔══██║ ███╔╝  ██╔══╝  ██╔══██╗██║     
██║     ███████╗   ██║   ██║ ╚═╝ ██║██║  ██║███████╗███████╗██║  ██║███████╗
╚═╝     ╚══════╝   ╚═╝   ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Developed by:
    Rishika Mohanta, Research Technician, Turner Lab, Janelia Research Campus
"""

print(start_text)

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

model_database = pd.read_csv(
    "https://raw.githubusercontent.com/neurorishika/flymazerl/main/model_description_rajagopalan.csv"
)

argument_parser = argparse.ArgumentParser(
    description=start_text
    + "Script to fit the parameters of the a variety of Classical RL agents to the a 2AFC maze task using Bayesian MCMC."
)
argument_parser.add_argument("--save_path", type=str, default="../fits/", help="Path to save the fits.")
argument_parser.add_argument(
    "--agent",
    type=str,
    default="CQES",
    help="The agent to fit the dynamics to (Ref: SHORTCODE column in model_description.csv)",
)
argument_parser.add_argument(
    "--action-set-data",
    type=str,
    default=FLYMAZERL_PATH + "/data/rajagopalan2022/training_choice_set.csv",
    help="The file to load the action set data from.",
)
argument_parser.add_argument(
    "--reward-set-data",
    type=str,
    default=FLYMAZERL_PATH + "/data/rajagopalan2022/training_reward_set.csv",
    help="The file to load the reward set data from.",
)
argument_parser.add_argument("--n_chains", type=int, default=1, help="Number of MC-MC chains to sample")
argument_parser.add_argument("--n_burn", type=int, default=1000, help="Number of MC-MC burn-in steps per chain")
argument_parser.add_argument("--n_steps", type=int, default=2500, help="Number of MC-MC sampling steps per chain")
argument_parser.add_argument(
    "--target_acceptance_rate", type=float, default=0.8, help="Target acceptance rate for MC-MC"
)
argument_parser.add_argument(
    "--n_cores", type=int, default=1, help="Number of cores to use for parallelization",
)
argument_parser.add_argument("--n_splits", type=int, default=1, help="Number of subdatasets to split the data into")
args = argument_parser.parse_args()

assert args.agent in model_database.SHORTCODE.values, "Agent not found in model_description.csv"
assert args.save_path[-1] == "/", "Save path must end with a '/'"
assert os.path.exists(args.save_path), "Save path does not exist"

agentClass = eval(model_database.loc[model_database.SHORTCODE == args.agent, "AgentClass"].values[0])

timestring = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

print("Fitting agent: {}".format(agentClass.__name__))
print("Loading data from:\n{}\n{}".format(args.action_set_data, args.reward_set_data))

if args.n_splits == 1:
    if args.action_set_data.startswith("http"):
        # use requests to download the data
        response = requests.get(args.action_set_data)
        response.raise_for_status()
        action_set = np.loadtxt(io.BytesIO(response.content), delimiter=",", dtype=np.int32)
    else:
        action_set = np.loadtxt(args.action_set_data, delimiter=",", dtype=np.int32)
    if args.reward_set_data.startswith("http"):
        # use requests to download the data
        response = requests.get(args.reward_set_data)
        response.raise_for_status()
        reward_set = np.loadtxt(io.BytesIO(response.content), delimiter=",", dtype=np.int32)
    else:
        reward_set = np.loadtxt(args.reward_set_data, delimiter=",", dtype=np.int32)

    print("Loaded action and reward sets.")

    assert len(action_set) == len(reward_set), "Action and reward set must have the same length."
    n_sessions = len(action_set)
    n_trials_in_each_session = set([len(action_set[i]) for i in range(n_sessions)])

    assert len(n_trials_in_each_session) == 1, "All sessions must have the same number of trials."
    n_trials_per_session = n_trials_in_each_session.pop()

    print("Found {} sessions with {} trials each.".format(n_sessions, n_trials_per_session))

    fitAgent = agentClass(ymaze_static(n_trials_per_session))

    extra_code = model_database.loc[model_database.SHORTCODE == args.agent, "FittingExtras"].values[0]
    if extra_code != "None":
        print("Found extra code for agent initialization.")
        exec(extra_code)

    print("Agent successfully initialized. Starting fitting.")

    if model_database.loc[model_database.SHORTCODE == args.agent, "OptimizationAlgorithm"].values[0] == "NUTS":
        model, data = fitAgent.fit(
            action_set,
            reward_set,
            niters=args.n_steps,
            ntune=args.n_burn,
            nchains=args.n_chains,
            nparallel=args.n_cores,
            target_accept=args.target_acceptance_rate,
        )
    elif model_database.loc[model_database.SHORTCODE == args.agent, "OptimizationAlgorithm"].values[0] == "CG+NUTS":
        model, data = fitAgent.fit(
            action_set,
            reward_set,
            niters=args.n_steps,
            ntune=args.n_burn,
            nchains=args.n_chains,
            nparallel=args.n_cores,
        )
    else:
        raise ValueError("Optimization algorithm not recognized.")

    print("Fitting finished. Saving model.")

    data.to_netcdf(args.save_path + args.agent + "_" + timestring + ".nc")
    with open(args.save_path + args.agent + "_" + timestring + "_model.pkl", "wb") as buffer:
        pickle.dump(model, buffer)

    with open(args.save_path + "model_fitting_log.csv", "a") as buffer:
        buffer.write(
            args.agent + "," + timestring + "," + str(args.n_burn) + "," + str(args.n_steps) + "," + str(args.n_chains)
        )
        buffer.write("\n")

    print("Model saved.")

else:
    assert args.n_splits > 1, "Number of splits must be greater than 1."
    n_sessions = np.loadtxt(args.action_set_data, delimiter=",").shape[0]

    assert n_sessions % args.n_splits == 0, "Number of splits must divide the number of sessions."
    n_sessions_per_split = n_sessions // args.n_splits

    if args.action_set_data.startswith("http"):
        # use requests to download the data
        response = requests.get(args.action_set_data)
        response.raise_for_status()
        full_action_set = np.loadtxt(io.BytesIO(response.content), delimiter=",", dtype=np.int32)
    else:
        full_action_set = np.loadtxt(args.action_set_data, delimiter=",", dtype=np.int32)
    if args.reward_set_data.startswith("http"):
        # use requests to download the data
        response = requests.get(args.reward_set_data)
        response.raise_for_status()
        full_reward_set = np.loadtxt(io.BytesIO(response.content), delimiter=",", dtype=np.int32)
    else:
        full_reward_set = np.loadtxt(args.reward_set_data, delimiter=",", dtype=np.int32)

    print("Loaded action and reward sets.")

    assert len(full_action_set) == len(full_reward_set), "Action and reward set must have the same length."
    n_trials_in_each_session = set([len(full_action_set[i]) for i in range(n_sessions)])

    assert len(n_trials_in_each_session) == 1, "All sessions must have the same number of trials."
    n_trials_per_session = n_trials_in_each_session.pop()

    print("Found {} sessions with {} trials each.".format(n_sessions, n_trials_per_session))

    for i in range(args.n_splits):
        print("Starting split {} of {}.".format(i + 1, args.n_splits))

        action_set = full_action_set[i * n_sessions_per_split : (i + 1) * n_sessions_per_split]
        reward_set = full_reward_set[i * n_sessions_per_split : (i + 1) * n_sessions_per_split]

        n_sessions = len(action_set)

        fitAgent = agentClass(ymaze_static(n_trials_per_session))

        extra_code = model_database.loc[model_database.SHORTCODE == args.agent, "FittingExtras"].values[0]
        if extra_code != "None":
            print("Found extra code for agent initialization.")
            exec(extra_code)

        print("Agent successfully initialized. Starting fitting.")

        if model_database.loc[model_database.SHORTCODE == args.agent, "OptimizationAlgorithm"].values[0] == "NUTS":
            model, data = fitAgent.fit(
                action_set,
                reward_set,
                niters=args.n_steps,
                ntune=args.n_burn,
                nchains=args.n_chains,
                nparallel=args.n_cores,
                target_accept=args.target_acceptance_rate,
            )
        elif model_database.loc[model_database.SHORTCODE == args.agent, "OptimizationAlgorithm"].values[0] == "CG+NUTS":
            model, data = fitAgent.fit(
                action_set,
                reward_set,
                niters=args.n_steps,
                ntune=args.n_burn,
                nchains=args.n_chains,
                nparallel=args.n_cores,
            )
        else:
            raise ValueError("Optimization algorithm not recognized.")

        print("Fitting finished. Saving model.")

        data.to_netcdf(args.save_path + args.agent + "_" + str(i) + "_" + timestring + ".nc")
        with open(args.save_path + args.agent + "_" + str(i) + "_" + timestring + "_model.pkl", "wb") as buffer:
            pickle.dump(model, buffer)

        print("Model saved.")

    with open(args.save_path + "model_fitting_log.csv", "a") as buffer:
        buffer.write(
            args.agent + "," + timestring + "," + str(args.n_burn) + "," + str(args.n_steps) + "," + str(args.n_chains)
        )
        buffer.write("\n")

    print("All splits finished.")

print("Fitting is complete. The model fitting log is available at {}.".format(args.save_path + "model_fitting_log.csv"))
print("The model is available at {}.".format(args.save_path + args.agent + "_" + timestring + ".pkl"))
print("Thank you for using flymazerl. Have a nice day :)")

