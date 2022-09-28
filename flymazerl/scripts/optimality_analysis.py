from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
import time
import datetime

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import seaborn as sns
from tqdm import tqdm

from flymazerl.agents.classical import *
from flymazerl.gym.environment import ymaze_static
from flymazerl.utils import generate_params_from_fits,get_schedule_histories
from flymazerl.utils import generate_random_schedule_with_blocks
from flymazerl.utils import get_schedule_histories

def generate_non_stationary_2AFC_schedule(max_trials,hazard_rate,reward_contrast,reward_gain):
    trace = np.zeros(max_trials)
    trace[0] = np.random.choice([0,1])
    for i in range(max_trials):
        if np.random.rand()<hazard_rate:
            trace[i] = 1 - trace[i-1]
        else:
            trace[i] = trace[i-1]
    choice_0 = np.ones(max_trials)*(reward_gain+reward_contrast/2)
    choice_1 = np.ones(max_trials)*(reward_gain-reward_contrast/2)
    a1 = np.where(trace==0,choice_0,choice_1)
    a2 = np.where(trace==1,choice_0,choice_1)
    return generate_random_schedule_with_blocks(max_trials, a1, a2)

def get_model_performance(model,n_schedules,n_agents_per_schedule,schedule_size,hazard_rate,reward_contrast,reward_gain,verbose=False):
    performance = []
    if verbose:
        r = tqdm(range(n_schedules))
    else:
        r = range(n_schedules)
    for i in r:
        schedule = generate_non_stationary_2AFC_schedule(schedule_size,hazard_rate,reward_contrast,reward_gain)
        env = ymaze_static(100,schedule=schedule)
        params, policyparams = generate_params_from_fits(model, n_samples=n_agents_per_schedule, sample_from_population=True)
        _,reward_history = get_schedule_histories(env, model, n_agents_per_schedule, params, policyparams, False, True)
        reward_history = np.array(reward_history,dtype=float)
        performance.append(np.mean(reward_history,axis=1))
    return np.concatenate(performance)

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

model_database = pd.read_csv("https://raw.githubusercontent.com/neurorishika/flymazerl/main/model_description.csv")

# filter to only acceptreject models
model_database = model_database[model_database['Variant'] == 'acceptreject']

# generate maps
model_name_map = {model_database.iloc[i]['SHORTCODE']:model_database.iloc[i]['Model'] for i in range(len(model_database))}
model_simple_abv_map = {model_database.iloc[i]['SHORTCODE']:model_database.iloc[i]['ModelAbv'] for i in range(len(model_database))}
model_class_name_map_inv = {model_database.iloc[i]['SHORTCODE']:model_database.iloc[i]['AgentClass'] for i in range(len(model_database))}
model_class_name_map = {model_database.iloc[i]['AgentClass']:model_database.iloc[i]['SHORTCODE'] for i in range(len(model_database))}

models = [eval(x) for x in model_class_name_map.keys()]

argument_parser = argparse.ArgumentParser(
    description="Script to test model across the space of non-stationary 2AFC tasks"
    )
argument_parser.add_argument(
    "--agent",
    type=str,
    default="CQBR",
    help="The agent to optimize for (Ref: SHORTCODE column in model_description.csv)",
)
argument_parser.add_argument(
    "--resolution",
    default=10,
    type=int,
    help="Number of subdivisions to divide the parameter space into",
)
argument_parser.add_argument(
    "--n_schedule",
    default=1000,
    type=int,
    help="Number of schedules to test",
)
argument_parser.add_argument(
    "--n_agents_per_schedule",
    type=int,
    default=1,
    help="Number of agents to run per schedule",
)
argument_parser.add_argument(
    "--schedule_size",
    type=int,
    default=1000,
    help="Size of the schedule to test",
)
argument_parser.add_argument(
    "--generate_figure",
    type=bool,
    default=True,
    help="Whether to generate a GIF of the task performance",
)

args = argument_parser.parse_args()

agent_name = args.agent
model = eval(model_class_name_map_inv[agent_name])

hazard_rates = np.logspace(-2,0,args.resolution)
reward_contrasts = np.linspace(0,1,args.resolution)
reward_gains = np.linspace(0,1,args.resolution)
n_max = len(hazard_rates)*len(reward_contrasts)*len(reward_gains)

n_schedules = args.n_schedule
n_agents_per_schedule = args.n_agents_per_schedule
schedule_size = args.schedule_size

i = 1

time_per_iteration = 0
start_time = datetime.datetime.now()

param_combinations = []
for hazard_rate in hazard_rates:
    for reward_contrast in reward_contrasts:
        for reward_gain in reward_gains:
            print('{}/{}, time remaining: {:0.2f}s, completion at: {}'.format(i,n_max,time_per_iteration*(n_max-i),(datetime.datetime.now()+datetime.timedelta(seconds=time_per_iteration*(n_max-i))).strftime("%d/%m/%Y %H:%M:%S")),end='\r')
            i += 1
            # check if reward contrast and reward gain combination is valid
            if reward_gain+reward_contrast/2>1 or reward_gain-reward_contrast/2<0:
                continue
            performance = get_model_performance(model,n_schedules,n_agents_per_schedule,schedule_size,hazard_rate,reward_contrast,reward_gain,verbose=False)
            param_combinations.append([hazard_rate,reward_contrast,reward_gain,np.mean(performance),np.std(performance),performance])
            
            time_per_iteration = (datetime.datetime.now()-start_time).total_seconds()/i
print('\n')

df = pd.DataFrame(param_combinations,columns=['hazard_rate','reward_contrast','reward_gain','performance','performance_std','performance_history'])
df['norm_performance'] = df['performance']/df['reward_gain']
df['norm_performance_std'] = df['performance_std']/df['reward_gain']

df.to_pickle(f'{model_simple_abv_map[model_class_name_map[model.__name__]]}_performance.pkl')


if args.generate_figure:
    print('Generating figure...')
    sns.set(font_scale=1.0)
    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122,projection='3d')

    def init():
        vmin = df['norm_performance'].min()
        vmax = df['norm_performance'].max()
        ax1.scatter3D(df['reward_contrast'],df['reward_gain'],np.log10(df['hazard_rate']),c=df['norm_performance'],s=50*df['performance'],vmin=vmin,vmax=vmax,cmap='Blues')
        vmin = df['norm_performance_std'].min()
        vmax = df['norm_performance_std'].max()
        ax2.scatter3D(df['reward_contrast'],df['reward_gain'],np.log10(df['hazard_rate']),c=df['norm_performance_std'],s=1e3*df['performance_std'],vmin=vmin,vmax=vmax,cmap='Reds')
        ax1.set_xlabel('Reward Contrast')
        ax1.set_ylabel('Reward Gain')
        ax1.set_zlabel('log10(Hazard Rate)')
        ax2.set_xlabel('Reward Contrast')
        ax2.set_ylabel('Reward Gain')
        ax2.set_zlabel('log10(Hazard Rate)')
        ax1.grid(False)
        ax2.grid(False)
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.suptitle('Task Performance for {} Model'.format(model_simple_abv_map[model_class_name_map[model.__name__]]))
        return fig,

    def animate(i):
        ax1.view_init(elev=10., azim=i)
        ax2.view_init(elev=10., azim=i)
        return fig,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)

    anim.save(f'{model_simple_abv_map[model_class_name_map[model.__name__]]}.gif', fps=30)

print('Done!')