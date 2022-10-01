import argparse
import datetime
import os
import json
import random
import string
import requests
import io

from flymazerl.agents.nn import *
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

argument_parser = argparse.ArgumentParser(
    description=start_text
    + "Script to fit the parameters of the a variety of NN-based RL agents to the a 2AFC maze task using Backpropagation (through time)."
)
argument_parser.add_argument("--save_path", type=str, default="../fits/nn/", help="Path to save the fits.")
argument_parser.add_argument(
    "--agent", type=str, default="GQNN", help="Agent to use. Options: GQNN, GRNN",
)
# training data
argument_parser.add_argument(
    "--action-set-data",
    type=str,
    default=FLYMAZERL_PATH + "/data/mohanta2022/training_choice_set.csv",
    help="The file to load the action set data from.",
)
argument_parser.add_argument(
    "--reward-set-data",
    type=str,
    default=FLYMAZERL_PATH + "/data/mohanta2022/training_reward_set.csv",
    help="The file to load the reward set data from.",
)
# Training parameters
argument_parser.add_argument("--n_folds", type=int, default=3, help="Number of K-fold cross-validation folds to use.")
argument_parser.add_argument("--n_ensemble", type=int, default=100, help="Number of ensemble models to use.")
argument_parser.add_argument("--history_size", type=int, default=10, help="History size to use.")
argument_parser.add_argument("--max_epochs", type=int, default=100000, help="Maximum number of epochs to use.")
argument_parser.add_argument(
    "--early_stopping", type=int, default=5000, help="Number of epochs to wait before stopping."
)
argument_parser.add_argument("--minibatch_size", type=int, default=1, help="Minibatch size to use.")
argument_parser.add_argument("--minibatch_seed", type=int, default=15403997, help="Seed for minibatch selection.")
argument_parser.add_argument("--use_lr_scheduler", type=bool, default=False, help="Use learning rate scheduler.")
argument_parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate to use.")
argument_parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay to use.")
argument_parser.add_argument("--print_every", type=int, default=500, help="Number of epochs to wait before printing.")
argument_parser.add_argument("--train_test_split", type=float, default=0.8, help="Train/test split to use.")
argument_parser.add_argument("--tolerance", type=float, default=1e-4, help="Tolerance to use validation.")

# Model parameters : GRNN
argument_parser.add_argument("--reservoir_size", type=int, default=100, help="Reservoir size to use. (Only for GRNN)")
argument_parser.add_argument(
    "--num_reservoirs", type=int, default=1, help="Number of Reservoirs to use. (Only for GRNN)"
)
argument_parser.add_argument("--encoder_size", type=int, default=None, help="Encoder size to use. (Only for GRNN)")
argument_parser.add_argument(
    "--kind", type=str, default="RNN", help="Kind of network to use. (Only for GRNN; options: RNN, LSTM, GRU)"
)

# Model parameters : GQNN
argument_parser.add_argument(
    "--hidden_state_sizes", nargs="+", type=int, default=[10], help="Hidden state sizes to use. (Only for GQNN)"
)
argument_parser.add_argument(
    "--activation",
    type=str,
    default="relu",
    help="Activation function to use. (Only for GQNN; options: relu, tanh, sigmoid)",
)

# Common parameters
argument_parser.add_argument(
    "--policy_type",
    type=str,
    default="acceptreject",
    help="Policy type to use. (options: softmax, greedy, acceptreject)",
)
argument_parser.add_argument(
    "--device", type=str, default="cpu", help="Device to use. (Only for GRNN; options: cpu, cuda)"
)
argument_parser.add_argument(
    "--symmetric", type=str, default="yes", help="Whether to use symmetric or asymmetric networks (yes/no)."
)
argument_parser.add_argument("--allow_negative", type=str, default="no", help="Whether to allow negative Q values.")
argument_parser.add_argument(
    "--omission_is_punishment", type=str, default="no", help="Whether to use omission as a punishment."
)
argument_parser.add_argument("--hardness", type=float, default=0.8141, help="Balance between Sigmoid and Hardsigmoid.")

args = argument_parser.parse_args()

# process booleans
args.symmetric = True if args.symmetric == "yes" else False
args.allow_negative = True if args.allow_negative == "yes" else False
args.omission_is_punishment = True if args.omission_is_punishment == "yes" else False

assert args.save_path[-1] == "/", "Save path must end with a '/'"
assert os.path.exists(args.save_path), "Save path does not exist"
assert args.agent in ["GQNN", "GRNN"], "Agent must be one of: GQNN, GRNN"
assert args.policy_type in [
    "softmax",
    "egreedy",
    "acceptreject",
], "Policy type must be one of: softmax, greedy, acceptreject"
assert args.device in ["cpu", "cuda"], "Device must be one of: cpu, cuda"
assert args.kind in ["RNN", "LSTM", "GRU"], "Kind must be one of: RNN, LSTM, GRU"
assert args.activation in ["relu", "tanh", "sigmoid"], "Activation function must be one of: relu, tanh, sigmoid"


agentClass = GQLearner if args.agent == "GQNN" else GRNNLearner

# get time stamp down to the millisecond
timestring = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

print("Fitting agent: {}".format(agentClass.__name__))
print("Loading data from:\n{}\n{}".format(args.action_set_data, args.reward_set_data))

# generate the parameter string
parameter_string = "GQNN" if args.agent == "GQNN" else "GRNN"
if args.agent == "GQNN":
    parameter_string += "_" + "-".join([str(x) for x in args.hidden_state_sizes])
    parameter_string += "_" + args.activation
elif args.agent == "GRNN":
    if args.encoder_size is not None:
        parameter_string += "_" + str(args.encoder_size)
        parameter_string += "-" + str(args.num_reservoirs)
    else:
        parameter_string += "_" + str(args.num_reservoirs)
    parameter_string += "x" + str(args.reservoir_size)
    parameter_string += "_" + args.kind
parameter_string += "_" + args.policy_type
parameter_string += "_" + ("symmetric" if args.symmetric else "asymmetric")
parameter_string += "_" + ("qpn" if args.allow_negative else "qp")
parameter_string += "_" + ("punishment" if args.omission_is_punishment else "no-punishment")
parameter_string += "_" + timestring

print("Model: {}".format(parameter_string))

# create directory for storing the fits
save_path = args.save_path + parameter_string + "/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# save the arguments used to fit the model
with open(save_path + "params.json", "w") as f:
    json.dump(vars(args), f, indent=4)

# load the training data
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

# create the dataframe to store the results
results_df = pd.DataFrame(
    columns=[
        "ModelID",
        "EnsembleID",
        "FoldID",
        "FlyID",
        "nParams",
        "MSE(train)",
        "NMSE(train)",
        "LogLikelihood(train)",
        "AIC(train)",
        "BIC(train)",
        "MSE(test)",
        "NMSE(test)",
        "LogLikelihood(test)",
        "AIC(test)",
        "BIC(test)",
        "P(action)",
        "Model Path",
        "Training Loss",
        "Validation Loss",
        "Best Validation Loss",
        "Epochs",
        "Best Validation Epoch",
        "Training Time",
    ]
)

# create the toy environment for training
env = ymaze_static(len(action_set))

# create the fit parameters
fit_params = {
    "train_test_split": args.train_test_split,
    "n_replications": 1,
    "early_stopping": True if args.early_stopping > 0 else False,
    "early_stopping_patience": args.early_stopping,
    "max_epochs": args.max_epochs,
    "learning_rate": args.learning_rate,
    "print_every": args.print_every,
    "weight_decay": args.weight_decay,
    "filter_best": False,
    "tolerance": args.tolerance,
    "scheduler": args.use_lr_scheduler,
    "minibatch_size": args.minibatch_size,
    "minibatch_seed": args.minibatch_seed,
}

# create learner parameters
learner_params = {
    "allow_negative_values": args.allow_negative,
    "symmetric_q_function": args.symmetric,
    "omission_is_punishment": args.omission_is_punishment,
    "policy_type": args.policy_type,
    "device": args.device,
    "pre_trained": False,
    "model_path": None,
    "multi_agent": False,
    "n_agents": 1,
    "hardness": args.hardness,
}
if args.agent == "GQNN":
    learner_params["hidden_state_sizes"] = args.hidden_state_sizes
    learner_params["activation"] = args.activation
elif args.agent == "GRNN":
    learner_params["encoder_size"] = args.encoder_size
    learner_params["num_layers"] = args.num_reservoirs
    learner_params["reservoir_size"] = args.reservoir_size
    learner_params["kind"] = args.kind

# create the agent
agent = agentClass(env, learner_params)

# create KFold cross-validation dataset
if args.n_folds > 1:
    indices = np.arange(len(action_set))
    np.random.shuffle(indices)
    K_folds = np.array_split(indices, args.n_folds)
    training_sets = []
    validation_sets = []
    for i in range(args.n_folds):
        training_indices = np.array(list(set(range(len(action_set))) - set(K_folds[i])), dtype=np.int32)
        validation_indices = np.array(K_folds[i], dtype=np.int32)
        training_sets.append(
            {
                "actions": action_set[training_indices],
                "rewards": reward_set[training_indices],
                "indices": training_indices,
            }
        )
        validation_sets.append(
            {
                "actions": action_set[validation_indices],
                "rewards": reward_set[validation_indices],
                "indices": validation_indices,
            }
        )
else:
    training_sets = [{"actions": action_set, "rewards": reward_set, "indices": np.arange(len(action_set))}]
    validation_sets = [{"actions": action_set, "rewards": reward_set, "indices": np.arange(len(action_set))}]

# start loop over the ensemble
for ensemble_id in range(args.n_ensemble):
    # start loop over the folds
    for fold_id, (train_set, val_set) in enumerate(zip(training_sets, validation_sets)):
        # start fitting the model
        print("Fitting model {}/{}. Fold {}/{}".format(ensemble_id + 1, args.n_ensemble, fold_id + 1, args.n_folds))

        # generate a unique model ID as a random 8-character string
        model_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

        fit_statistics = agent.fit(train_set["actions"], train_set["rewards"], uid=model_id, **fit_params)

        # move model_0.pt to save_path after renaming it to ensemble_id_fold_id.pt
        os.rename(f"model_{model_id}_0.pt", save_path + "ensemble_{}_fold_{}.pt".format(ensemble_id, fold_id))

        # load the model
        agent.load_pre_trained_model(save_path + "ensemble_{}_fold_{}.pt".format(ensemble_id, fold_id))

        # evaluate the model on training data
        p_action_train = agent.get_action_probabilities_from_data(train_set["actions"], train_set["rewards"])
        for i, fly_id in enumerate(train_set["indices"]):
            obs_actions = train_set["actions"][i][1:] if args.agent == "GRNN" else train_set["actions"][i]
            obs_action_smooth = np.convolve(
                obs_actions, np.ones((args.history_size,)) / args.history_size, mode="full"
            )[args.history_size :]
            pred_action_prob = p_action_train[i, :, -1]
            pred_action_smooth = np.convolve(
                pred_action_prob, np.ones((args.history_size,)) / args.history_size, mode="full"
            )[args.history_size :]

            # calculate MSE
            mse_train = np.mean((obs_action_smooth - pred_action_smooth) ** 2)

            # calculate NMSE by dividing by the variance of the observed action probabilities
            nmse_train = mse_train / np.std(obs_action_smooth ** 2)

            n_params = sum(p.numel() for p in agent.agent.parameters() if p.requires_grad)

            # calculate bernoulli log likelihood of the actions given the predicted action probabilities
            ll_train = np.sum(np.log(pred_action_prob[obs_actions == 1])) + np.sum(
                np.log(1 - pred_action_prob[obs_actions == 0])
            )

            # calculate AIC
            aic_train = -2 / len(obs_actions) * mse_train + 2 * n_params

            # calculate BIC
            bic_train = -2 / len(obs_actions) * mse_train + n_params * np.log(len(obs_actions))

            # append to dataframe
            results_df.loc[len(results_df)] = [
                parameter_string,
                ensemble_id,
                fold_id,
                fly_id,
                n_params,
                mse_train,
                nmse_train,
                ll_train,
                aic_train,
                bic_train,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                np.NaN,
                pred_action_prob.tolist(),
                "ensemble_{}_fold_{}".format(ensemble_id, fold_id),
                fit_statistics[0]["training_loss"],
                fit_statistics[0]["validation_loss"],
                fit_statistics[0]["best_val_loss"],
                fit_statistics[0]["epoch"],
                fit_statistics[0]["best_val_epoch"],
                fit_statistics[0]["training_time"],
            ]

        if args.n_folds > 1:
            # evaluate the model on validation data
            p_action_val = agent.get_action_probabilities_from_data(val_set["actions"], val_set["rewards"])
            for i, fly_id in enumerate(val_set["indices"]):
                obs_actions = val_set["actions"][i][1:] if args.agent == "GRNN" else val_set["actions"][i]
                obs_action_smooth = np.convolve(
                    obs_actions, np.ones((args.history_size,)) / args.history_size, mode="full"
                )[args.history_size :]
                pred_action_prob = p_action_val[i, :, -1]
                pred_action_smooth = np.convolve(
                    pred_action_prob, np.ones((args.history_size,)) / args.history_size, mode="full"
                )[args.history_size :]

                # calculate MSE
                mse_val = np.mean((obs_action_smooth - pred_action_smooth) ** 2)

                # calculate NMSE by dividing by the variance of the observed action probabilities
                nmse_val = mse_val / np.std(obs_action_smooth ** 2)

                n_params = sum(p.numel() for p in agent.agent.parameters() if p.requires_grad)

                # calculate bernoulli log likelihood of the actions given the predicted action probabilities
                ll_val = np.sum(np.log(pred_action_prob[obs_actions == 1])) + np.sum(
                    np.log(1 - pred_action_prob[obs_actions == 0])
                )

                # calculate AIC
                aic_val = -2 / len(obs_actions) * mse_val + 2 * n_params

                # calculate BIC
                bic_val = -2 / len(obs_actions) * mse_val + n_params * np.log(len(obs_actions))

                # append to dataframe
                results_df.loc[len(results_df)] = [
                    parameter_string,
                    ensemble_id,
                    fold_id,
                    fly_id,
                    n_params,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    np.NaN,
                    mse_val,
                    nmse_val,
                    ll_val,
                    aic_val,
                    bic_val,
                    pred_action_prob.tolist(),
                    "ensemble_{}_fold_{}".format(ensemble_id, fold_id),
                    fit_statistics[0]["training_loss"],
                    fit_statistics[0]["validation_loss"],
                    fit_statistics[0]["best_val_loss"],
                    fit_statistics[0]["epoch"],
                    fit_statistics[0]["best_val_epoch"],
                    fit_statistics[0]["training_time"],
                ]

# save the results to a file with compression
results_df.to_csv(save_path + "results.csv.gz", index=False, compression="gzip")

print("Fitting is complete. The model fitting log is available at {}.".format(args.save_path + "model_fitting_log.csv"))
print("The model is available at {}.".format(save_path))
print("Thank you for using flymazerl. Have a nice day :)")

