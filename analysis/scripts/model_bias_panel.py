import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/neurorishika/flymazerl/main/model_description.csv")
df.head()
df = df[df.Variant != "e-greedy"]
df = df[df.Variant != "softmax"]
panel_size = int(input("Enter panel size: "))
for agentclass in np.unique(df["AgentClass"]):
    filtered_df = df[df["AgentClass"] == agentclass]
    panel_string = ""
    for params in filtered_df["Parameter Name"]:
        dist = filtered_df[filtered_df["Parameter Name"] == params]["Default Prior Distribution"].values[0]
        if dist.split("(")[0] == "Beta":
            panel = np.linspace(0, 1, panel_size + 2)[1:-1]
        if dist.split("(")[0] == "Halfnormal":
            max_val = np.log10(float(dist.split("(")[1].split(")")[0]))
            panel = np.logspace(max_val - panel_size // 2 - 1, max_val + panel_size // 2 + 1, panel_size + 2)[1:-1]
        if dist.split("(")[0] == "Normal":
            mean = float(dist.split("(")[1].split(",")[0])
            std = float(dist.split("(")[1].split(",")[1].split(")")[0])
            panel = np.linspace(mean - std * (1 + panel_size // 2), mean + std * (1 + panel_size // 2), panel_size + 2)[
                1:-1
            ]

        panel_string += f"{params}_panel = [{','.join(map(str,panel))}]\n"

    value_param_set = filtered_df[filtered_df["Parameter Type"] == "learningrule"]["Parameter Name"].values
    policy_param_set = filtered_df[filtered_df["Parameter Type"] == "policy"]["Parameter Name"].values
    code = f"""
from itertools import product
from flymazerl.gym.environment import ymaze_static
from flymazerl.agents.classical import {agentclass}
from tqdm import tqdm
import pickle

print('Starting Panel Sweep for {agentclass}...')

n_trials_per_session = 100
reward_fraction = 0.5
n_agents = 20

{panel_string}
for {', '.join(filtered_df['Parameter Name'].values)} in tqdm(product({', '.join(map(lambda v:v+'_panel',filtered_df['Parameter Name'].values))})):
    params = {{ {', '.join(map(lambda v:'"'+v+'" : '+v,value_param_set))} }}
    policy_params = {{ {', '.join(map(lambda v:'"'+v+'" : '+v,policy_param_set))} }}
        
    actions_set = []
    rewards_set = []

    for i in range(n_agents):
        env = ymaze_static(n_trials_per_session,reward_fraction,)
        simAgent = {agentclass}(env,params,policy_params)
        simAgent.next_episode()
        actions_set.append(simAgent.action_history)
        rewards_set.append(simAgent.reward_history)

    fitAgent = {agentclass}(ymaze_static(n_trials_per_session,reward_fraction))
    model, fitted_data = fitAgent.fit(actions_set,rewards_set,niters=1000,nchains=4,nparallel=4,plot_trace=False,plot_posterior=False,print_summary=False)
    fitted_data.to_netcdf(../fits/bias_testing/{agentclass}_{'_'.join(map(lambda v:'{'+v+':0.2f}',filtered_df['Parameter Name'].values))} + ".nc")

print('Panel Sweep for {agentclass} complete!')
"""
    with open(f"bias_sweep_{agentclass}.py", "w") as f:
        f.write(code)
