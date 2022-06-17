import easygui
import pickle
import numpy as np
from flymazerl.agents.classical import *
from flymazerl.gym.environment import ymaze_dynamic, ymaze_static
from flymazerl.utils.visualization import draw_schedule
from flymazerl.utils.generators import generate_params_from_fits, generate_random_schedule_with_blocks
from flymazerl.utils.evaluation import get_schedule_histories, get_schedule_fitness
import pandas as pd

model_database = pd.read_csv("https://raw.githubusercontent.com/neurorishika/flymazerl/main/model_description.csv")
# Filter out the agents that have not been trained
model_database = model_database.loc[model_database.FitDir != "None"]

# Simple GUI to open a schedule.pkl file and view the average action histories
start_message = "Welcome to the schedule viewer! \n Please select a schedule file."
easygui.msgbox(start_message, "flymazerl Schedule Viewer", ok_button="Select File")

# Load the schedule file
schedule_dir = easygui.fileopenbox(msg="Select a schedule file", title="Schedule Viewer", filetypes=["*.pkl"])
with open(schedule_dir, "rb") as f:
    schedule = pickle.load(f)
env = ymaze_static(n_trials_per_session=len(schedule), schedule=schedule)

# Select the agents to evaluate on the schedule
agents_to_evaluate = easygui.multchoicebox(
    msg="Please select the agents to evaluate", title="Schedule Viewer", choices=model_database.SHORTCODE.unique()
)
# Get the parameters for the evaluation
n_agents = int(easygui.enterbox(msg="How many agents do you want to evaluate?", title="Schedule Viewer", default="1"))

# Compare with naive bias?
compare_naive = easygui.boolbox(
    msg="Do you want to compare with the naive bias?", title="Schedule Viewer", choices=["Yes", "No"]
)

for i in agents_to_evaluate:
    agentClass = eval(model_database.loc[model_database.SHORTCODE == i, "AgentClass"].values[0])
    params, policy_params = generate_params_from_fits(agentClass, n_agents)
    action_histories = get_schedule_histories(env, agentClass, n_agents, params, policy_params)

    if compare_naive:
        schedule_ = generate_random_schedule_with_blocks(len(schedule), [1, 0], [0, 1])
        env_ = ymaze_static(n_trials_per_session=len(schedule), schedule=schedule_)
        action_histories_naive = get_schedule_histories(env_, agentClass, n_agents, params, policy_params)
        draw_schedule(schedule, action_histories, compare_to=action_histories_naive)
    else:
        draw_schedule(schedule, action_histories)
