import numpy as np
from flymazerl.gym.environment import ymaze_static
from flymazerl.utils.evaluation import get_schedule_fitness, get_schedule_histories
from flymazerl.utils.visualization import draw_schedule
from flymazerl.utils.generators import generate_random_schedule, generate_params_from_fits
from tqdm import tqdm
import gc


def phi(k, A=100, B=2):
    """
    Returns the number of maximum number of indexes shuffled for a given generation k using the formula: phi(k) = ceil[(A-k)/B])
    =====================================================================================================================

    Parameters:
    k: generation number (int)
    A: constant (int) (default: 2000)
    B: constant (int) (default: 20)

    Returns:
    phi(k): number of maximum number of indexes shuffled for a given generation k (int)
    """
    return int(np.ceil(A - (k/B)))


def thermal_reshuffle(env, k, m, independent_shuffles=True):
    """
    Returns a list of m+1 schedules, where the first element is the original schedule and the rest are the m+1 shuffles.
    ====================================================================================================================

    Parameters:
    env: ymaze_static environment object
    k: generation number (int)
    m: number of shuffles (int)
    independent_shuffles: whether to use independent shuffles or not (bool)

    Returns:
    output: list of m+1 schedules (list)
    """
    output = [env]  # First element is the original schedule

    for i in range(m):
        t = np.random.choice(range(2, phi(k)))  # Number of shuffled trials
        schedule_ = np.copy(env.schedule)

        if independent_shuffles:
            indices1 = np.random.choice(env.n_trials_per_session, replace=False, size=t)  # Indices of shuffled trials (1)
            indices2 = np.random.choice(env.n_trials_per_session, replace=False, size=t)  # Indices of shuffled trials (2)
            schedule_[np.sort(indices1), 0] = env.schedule[indices1, 0]  # Swap the rewards
            schedule_[np.sort(indices2), 1] = env.schedule[indices2, 1]  # Swap the rewards
        else:
            indices = np.random.choice(env.n_trials_per_session, replace=False, size=t) # Indices of shuffled trials
            schedule_[indices, :] = env.schedule[indices, :]  # Swap the rewards

        output.append(
            ymaze_static(env.n_trials_per_session, env.reward_fraction, None, schedule_)
        )  # Append the shuffled schedule
    return output


def thermal_annealing(
    n_trials_per_session,
    reward_fraction,
    agentClass,
    n_agents,
    n_generations,
    m,
    params=None,
    policy_params=None,
    pregenerate_population=True,
    parallelize=False,
    initial_schedule=None,
    fitness_type = 'bias',
    independent_shuffles=True,
    ref_agentClass=None,
    ref_params=None,
    ref_policy_params=None,
    draw_progress=False,
    early_stopping=True,
    early_stopping_patience=10,
):
    """
    Returns the best schedule found by the thermal annealing algorithm.
    ===================================================================

    Parameters:
    n_trials_per_session: number of trials per episode (int)
    reward_fraction: fraction of rewarded trials  in each alternative(float)
    agentClass: class of the agent (FlYMazeAgent)
    n_agents: number of agents to be used (int)
    n_generations: number of generations of the algorithm (int)
    m: number of shuffles  (int)
    params: dictionary of parameters for the agent (dict)
    policy_params: dictionary of parameters for the policy (dict)
    parallelize: whether to parallelize or not (bool)
    initial_schedule: initial schedule (list)
    fitness_type: type of fitness function to be used (str)
    independent_shuffles: whether to use independent shuffles or not (bool)
    draw_progress: boolean to draw the progress of the algorithm (bool)
    early_stopping: whether to use early stopping or not (bool)
    early_stopping_patience: number of generations without improvement before stopping (int)

    Returns:
    best_schedule: best schedule found by the algorithm (ymaze_static object)
    fitnesses: list of fitnesses of the population found by the algorithm across generation (list of lists)
    """
    if policy_params is None and params is None and pregenerate_population:
        params, policy_params = generate_params_from_fits(agentClass, n_agents)
    
    if fitness_type == 'separation':
        if ref_policy_params is None and ref_params is None and pregenerate_population:
            ref_params, ref_policy_params = generate_params_from_fits(ref_agentClass, n_agents)

    if initial_schedule is None:
        init_schedule = generate_random_schedule(n_trials_per_session, reward_fraction)  # Initial schedule
    else:
        init_schedule = initial_schedule

    best_schedule = ymaze_static(n_trials_per_session, reward_fraction, None, init_schedule)  # Best schedule
    patience = early_stopping_patience
    best_fitness = -np.inf

    fitnesses = []  # Population fitnesses
    all_schedules = []  # Population
    k_start = 1  # Generation number

    for k in range(k_start, n_generations + k_start):
        schedules = thermal_reshuffle(best_schedule, k, m, independent_shuffles)  # Shuffled schedules

        if draw_progress:
            action_histories = get_schedule_histories(
                best_schedule, agentClass, n_agents, params, policy_params, parallelize
            )
            fitness_value = get_schedule_fitness(
                best_schedule, agentClass, n_agents, params, policy_params, parallelize, fitness_type, ref_agentClass, ref_params, ref_policy_params,
            )
            draw_schedule(best_schedule.schedule, action_histories, fitness_value)

        schedule_fitnesses = []
        for i in tqdm(range((m + 1)), desc=f"Generation {k}"):  # Fitnesses of the shuffled schedules
            schedule_fitnesses.append(
                get_schedule_fitness(
                    schedules[i],
                    agentClass,
                    n_agents,
                    params,
                    policy_params,
                    parallelize,
                    fitness_type,
                    ref_agentClass,
                    ref_params,
                    ref_policy_params,
                )
            )
        
        schedule_fitnesses = np.array(schedule_fitnesses).flatten()

        if best_fitness < schedule_fitnesses.max():
            best_schedule = schedules[np.argmax(schedule_fitnesses)]  # Best schedule
            best_fitness = schedule_fitnesses.max()
            patience = early_stopping_patience
        else:
            patience -= 1

        print("Best Schedule fitness:", best_fitness)

        fitnesses.append(schedule_fitnesses)  # Append Population fitnesses
        all_schedules.append([i.schedule for i in schedules])  # Append Population

        del schedules  # Delete the population to free memory
        del schedule_fitnesses  # Delete schedule fitnesses to free memory
        gc.collect()  # Garbage collection

        if early_stopping and patience == 0:
            break
    return best_schedule.schedule, fitnesses, all_schedules


def generate_random_child(
    population, population_fitness, surviving_fraction=0.2, mean_crossover=1.25, mutate=True, mean_mutation_rate=0.05, independent_mutations=True
):
    """
    Generates a random child from a population of schedules.
    ========================================================

    Parameters:
    population: list of ymaze_static environment objects
    population_fitness: list of fitnesses of the population
    surviving_fraction: fraction of the population  that survives the selection process (float) (default: 0.2)
    mean_crossover: mean number of the crossovers with a zero-truncated poisson distribution (float) (default: 1.25)
    mutate: whether or not to mutate the child (bool) (default: True)
    mean_mutation_rate: mean mutation rate (float) (default: 0.05)

    Returns:
    child: child generated from the selected population (ymaze_static object)
    """
    adjusted_population_fitness = np.int32(
        population_fitness >= np.quantile(population_fitness, 1 - surviving_fraction)
    )  # Adjusted population fitness
    relative_fitness = adjusted_population_fitness / np.sum(
        adjusted_population_fitness
    )  # Normalized population fitness

    p1index, p2index = np.random.choice(
        len(population), p=relative_fitness, size=2, replace=False
    )  # Indices of the parents
    parent = [population[p1index].schedule, population[p2index].schedule]  # Parent schedules

    crossover_points = (
        [0]
        + list(
            np.sort(
                np.random.choice(
                    population[0].n_trials_per_session, size=np.random.poisson(mean_crossover - 1) + 1, replace=False
                )
            )
        )
        + [population[0].n_trials_per_session]
    )  # Crossover points

    child = []
    parent_id = 0  # Parent id initialization
    for cp in range(len(crossover_points) - 1):
        child.append(parent[parent_id][crossover_points[cp] : crossover_points[cp + 1]])  # Get genes from the parent
        parent_id = int(np.logical_xor(parent_id, 1))  # Switch parent id
    child = np.concatenate(child, axis=0)

    reward_count = np.int32(
        np.sum(child, axis=0) - int(population[0].n_trials_per_session * population[0].reward_fraction)
    )  # Number of surplus/deficit rewards in the child
    if reward_count[0] > 0:  # Adjust the child if the number of rewards is greater than 0 in the first choice
        child[
            np.random.choice(
                np.arange(population[0].n_trials_per_session)[child[:, 0] == 1],
                size=np.abs(reward_count[0]),
                replace=False,
            ),
            0,
        ] = 0
    elif reward_count[0] < 0:  # Adjust the child if the number of rewards is less than 0 in the first choice
        child[
            np.random.choice(
                np.arange(population[0].n_trials_per_session)[child[:, 0] == 0],
                size=np.abs(reward_count[0]),
                replace=False,
            ),
            0,
        ] = 1
    if reward_count[1] > 0:  # Adjust the child if the number of rewards is greater than 0 in the second choice
        child[
            np.random.choice(
                np.arange(population[0].n_trials_per_session)[child[:, 1] == 1],
                size=np.abs(reward_count[1]),
                replace=False,
            ),
            1,
        ] = 0
    elif reward_count[1] < 0:  # Adjust the child if the number of rewards is less than 0 in the second choice
        child[
            np.random.choice(
                np.arange(population[0].n_trials_per_session)[child[:, 1] == 0],
                size=np.abs(reward_count[1]),
                replace=False,
            ),
            1,
        ] = 1

    if mutate:  # Shuffle the child if mutate is True
        n_mutations = 1 + np.clip(
            np.random.geometric(1 / (mean_mutation_rate * population[0].n_trials_per_session)),
            0,
            population[0].n_trials_per_session,
        )  # Number of indices to shuffle
        if n_mutations > 1:
            t = n_mutations
            if independent_mutations:
                indices1 = np.random.choice(
                    population[0].n_trials_per_session, replace=False, size=t
                )  # Indices of the shuffled trials (1)
                indices2 = np.random.choice(
                    population[0].n_trials_per_session, replace=False, size=t
                )  # Indices of the shuffled trials (2)
                child[np.sort(indices1), 0] = child[indices1, 0]  # Mutate the child (1)
                child[np.sort(indices2), 1] = child[indices2, 1]  # Mutate the child (2)
            else:
                indices = np.random.choice(
                    population[0].n_trials_per_session, replace=False, size=t
                )
                child[np.sort(indices), :] = child[indices, :]  # Mutate the child

    return ymaze_static(
        population[0].n_trials_per_session, population[0].reward_fraction, None, child
    )  # Return the child as a ymaze_static environment object


def genetic_optimization(
    n_trials_per_session,
    reward_fraction,
    agentClass,
    n_agents,
    n_generations,
    population_size,
    params=None,
    policy_params=None,
    pregenerate_population=True,
    parallelize=False,
    initial_schedule=None,
    fitness_type="bias",
    independent_mutations=False,
    ref_agentClass=None,
    ref_params=None,
    ref_policy_params=None,
    early_stopping=True,
    early_stopping_patience=10,
):
    """
    Returns the best schedule and the fitnesses of the population found by the genetic optimization algorithm.
    ==========================================================================================================

    Parameters:
    n_trials_per_session: number of trials per episode (int)
    reward_fraction: fraction of the reward (float)
    agentClass: class of the agent (FlYMazeAgent)
    n_agents: number of agents (int)
    n_generations: number of generations of the algorithm (int)
    population_size: size of the population (int)
    params: dictionary of parameters for the agent (dict)
    policy_params: dictionary of parameters for the policy (dict)
    parallelize: whether to parallelize or not (bool)
    initial_schedule: initial schedule for the population (list of list of int)

    Returns:
    best_schedule: best schedule found by the genetic optimization algorithm (list)
    fitnesses: fitnesses of the population found by the algorithm across generations (list)
    all_schedules: all schedules found by the algorithm across generations (list of list)
    """
    if policy_params is None and params is None and pregenerate_population:
        params, policy_params = generate_params_from_fits(agentClass, n_agents)

    if fitness_type == 'separation':
        if ref_policy_params is None and ref_params is None and pregenerate_population:
            ref_params, ref_policy_params = generate_params_from_fits(ref_agentClass, n_agents)


    if initial_schedule is not None:  # If an initial schedule is provided
        population = [
            ymaze_static(n_trials_per_session, reward_fraction, None, initial_schedule) for i in range(population_size)
        ]  # Initialize the population with the initial schedule
        population = [
            generate_random_child(population, np.ones(population_size) / population_size, independent_mutations=independent_mutations)
            for i in range(population_size)
        ]  # Introduce random mutations to the population
    else:  # If no initial schedule is provided
        population = [
            ymaze_static(
                n_trials_per_session,
                reward_fraction,
                None,
                generate_random_schedule(n_trials_per_session, reward_fraction),
            )
            for i in range(population_size)
        ]  # Initialize the population with random schedules

    fitnesses = []
    all_schedules = []
    k_start = 1  # Generation number

    best_schedule = population[0]
    best_fitness = -np.inf
    patience = early_stopping_patience

    for k in range(k_start, n_generations + k_start):
        population_fitness = []

        for i in tqdm(range(population_size), desc=f"Generation {k}"):  # Evaluate the population fitness
            fitness = get_schedule_fitness(population[i], agentClass, n_agents, params, policy_params, parallelize, fitness_type, ref_agentClass, ref_params, ref_policy_params)
            population_fitness.append(fitness)
            del fitness  # Delete the fitness to free memory
        population_fitness = np.array(population_fitness).flatten()

        if best_fitness < np.max(population_fitness):  # Update the best schedule and fitness
            best_schedule = population[np.argmax(population_fitness)]
            best_fitness = np.max(population_fitness)
            patience = early_stopping_patience
        else:
            patience -= 1

        print("Best Schedule fitness:", best_fitness)
        fitnesses.append(population_fitness)  # Append Fitnesses of the population
        all_schedules.append([i.schedule for i in population])  # Append all schedules of the population

        population = [
            generate_random_child(population, population_fitness, independent_mutations=independent_mutations) for i in range(population_size)
        ]  # Generate new population

        del population_fitness  # Delete the population fitness to free memory
        gc.collect()  # Garbage collection

        if early_stopping and patience == 0:
            break
    return best_schedule.schedule, fitnesses, all_schedules
