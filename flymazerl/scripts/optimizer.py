from flymazerl.agents.classical import *
from flymazerl.agents.phenomenological import *
from flymazerl.agents.neuralnetworks import *
from flymazerl.optimize.static import *
from flymazerl.utils.generators import generate_random_schedule_with_blocks

from flymazerl.utils.visualization import draw_schedule
from flymazerl.utils.evaluation import get_schedule_fitness
from flymazerl.optimize.static import thermal_annealing, genetic_optimization

import numpy as np
import pandas as pd

import os
import argparse
import datetime
import pickle


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
    description=start_text + "Script to optimize pre-trained RL agents to maximize bias in a 2AFC maze task."
)
argument_parser.add_argument(
    "--exit_on_completion", type=bool, default=True, help="Whether to end the job after completion."
)
argument_parser.add_argument("--save_path", type=str, default="../optimized_schedules/", help="Path to save schedules.")
argument_parser.add_argument(
    "--save_intermediate", type=bool, default=True, help="Whether to save intermediate schedules."
)
argument_parser.add_argument(
    "--agent",
    type=str,
    default="CQES",
    help="The agent to optimize for (Ref: SHORTCODE column in model_description.csv)",
)
argument_parser.add_argument("--n_trials_per_session", type=int, default=100, help="Number of trials per episode")
argument_parser.add_argument("--reward_fraction", type=float, default=0.5, help="Fraction of rewarded trials")
argument_parser.add_argument("--n_replicates", type=int, default=1, help="Number of replicates for consensus")
argument_parser.add_argument(
    "--optimization_method", type=str, default="annealing", help="Optimization method [genetic, annealing]"
)
argument_parser.add_argument("--early_stopping", type=bool, default=True, help="Whether to stop optimization early.")
argument_parser.add_argument(
    "--early_stopping_patience", type=int, default=10, help="Number of iterations to wait before stopping."
)
argument_parser.add_argument(
    "--initialization_method", type=str, default="random", help="Initialization method ['random' or 'primed']"
)
argument_parser.add_argument("--n_agents", type=int, default=500, help="Number of agents for fitness evaluation")
argument_parser.add_argument(
    "--parallelize", type=bool, default=False, help="Whether to parallelize fitness evaluation"
)
argument_parser.add_argument("--n_generations", type=int, default=10, help="Number of generations for optimization")
argument_parser.add_argument(
    "--m", type=int, default=20, help="(Thermal Annealing only) Number of shuffles in each generation"
)
argument_parser.add_argument(
    "--population_size", type=int, default=100, help="(Genetic Optimization only) Population size in each generation"
)
argument_parser.add_argument(
    "--fitness_function", type=str, default="bias", help="Fitness function to optimize (bias, performance, failure)"
)
argument_parser.add_argument(
    "--independent_shuffles", type=bool, default=True, help="Whether to shuffle the schedule independently"
)
argument_parser.add_argument(
    "--reference_agent",
    type=str,
    default="CQES",
    help="Reference agent for fitness evaluation (Ref: SHORTCODE column in model_description.csv)",
)
args = argument_parser.parse_args()

assert args.agent in model_database.SHORTCODE.values, "Agent not found in model_description.csv"
assert args.reference_agent in model_database.SHORTCODE.values, "Reference agent not found in model_description.csv"
assert args.save_path[-1] == "/", "Save path must end with a '/'"
assert os.path.exists(args.save_path), "Save path does not exist"

agentClass = eval(model_database.loc[model_database.SHORTCODE == args.agent, "AgentClass"].values[0])
ref_agentClass = eval(model_database.loc[model_database.SHORTCODE == args.reference_agent, "AgentClass"].values[0])
timestring = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")
unique_id = np.random.randint(100000, 999999)

print("Optimizing agent: {}".format(agentClass.__name__))

log_file = args.save_path + args.agent + "_" + timestring + "_" + str(unique_id) + ".log"
print("Saving to: {}".format(log_file))

with open(log_file, "w", encoding="utf-8") as f:
    f.write(start_text + "\n")
    f.write("Optimizing agent: {}\n\n".format(agentClass.__name__))
    f.write("Arguments:\n===========\n")
    for arg in vars(args):
        f.write(arg + ": " + str(getattr(args, arg)) + "\n")
    f.write("\n\n")

print("Starting optimization...")

for i in range(args.n_replicates):
    print("Replicate {}/{}".format(i + 1, args.n_replicates))

    if args.initialization_method == "random":
        init_schedule = None
    elif args.initialization_method == "primed":
        init_schedule = generate_random_schedule_with_blocks(args.n_trials_per_session, [1.0, 0.0], [0.0, 1.0])

    if args.optimization_method == "annealing":
        best_schedule, fitnesses, all_schedules = thermal_annealing(
            args.n_trials_per_session,
            args.reward_fraction,
            agentClass,
            args.n_agents,
            args.n_generations,
            args.m,
            parallelize=args.parallelize,
            initial_schedule=init_schedule,
            fitness_type=args.fitness_function,
            independent_shuffles=args.independent_shuffles,
            early_stopping=args.early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            ref_agentClass=ref_agentClass,
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("Replicate {}/{}:".format(i + 1, args.n_replicates))
            f.write("Best fitness = {}\n".format(np.max(fitnesses[-1])))

        with open(
            args.save_path + args.agent + "_" + timestring + "_best_schedule_" + str(i) + "_" + str(unique_id) + ".pkl",
            "wb",
        ) as f:
            pickle.dump(best_schedule, f)
        with open(
            args.save_path + args.agent + "_" + timestring + "_fitnesses_" + str(i) + "_" + str(unique_id) + ".pkl",
            "wb",
        ) as f:
            pickle.dump(fitnesses, f)
        if args.save_intermediate:
            with open(
                args.save_path + args.agent + "_" + timestring + "_schedules_" + str(i) + "_" + str(unique_id) + ".pkl",
                "wb",
            ) as f:
                pickle.dump(all_schedules, f)

    elif args.optimization_method == "genetic":
        best_schedule, fitnesses, all_schedules = genetic_optimization(
            args.n_trials_per_session,
            args.reward_fraction,
            agentClass,
            args.n_agents,
            args.n_generations,
            args.population_size,
            parallelize=args.parallelize,
            initial_schedule=init_schedule,
            fitness_type=args.fitness_function,
            independent_mutations=args.independent_shuffles,
            early_stopping=args.early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            ref_agentClass=ref_agentClass,
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("Replicate {}/{}:".format(i + 1, args.n_replicates))
            f.write("Best fitness = {}\n".format(np.max(fitnesses[-1])))

        with open(
            args.save_path + args.agent + "_" + timestring + "_best_schedule_" + str(i) + "_" + str(unique_id) + ".pkl",
            "wb",
        ) as f:
            pickle.dump(best_schedule, f)
        with open(
            args.save_path + args.agent + "_" + timestring + "_fitnesses_" + str(i) + "_" + str(unique_id) + ".pkl",
            "wb",
        ) as f:
            pickle.dump(fitnesses, f)
        if args.save_intermediate:
            with open(
                args.save_path + args.agent + "_" + timestring + "_schedules_" + str(i) + "_" + str(unique_id) + ".pkl",
                "wb",
            ) as f:
                pickle.dump(all_schedules, f)

print("Optimization complete. Log can be found at {}".format(log_file))
print("Thank you for using the flymazerl. Have a nice day!")
