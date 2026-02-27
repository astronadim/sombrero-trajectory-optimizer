# main_parallelized.py

# ============================================================================
# CRITICAL: Set thread limits BEFORE importing numpy/torch/scipy to prevent
# OpenMP/MKL oversubscription in multi-process runs (fixes libiomp5md.dll errors)
# ============================================================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # LAST RESORT ONLY - unsafe workaround

"""
To Do in code: 
- define boundary conditions and input parameters from trajectory_simulator in main parallelized
- save trajectory data only in one file, that is iteratively overwritten. otherwise too much space is used
"""

import random
import warnings
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import concurrent.futures
import multiprocessing

from functions_evolutionary import (
    initialize_first_gen,
    evaluate_fitness,
    tournament_selection,
    initialize_individual,
)
from functions_trajectory_simulator_solve_ivp import simulate_trajectory
from functions_results_processing import (
    save_fittest_individuals_per_generation,
    update_and_save_fittest_solution,
    plot_and_save_trajectory,
    save_trajectories_and_fitness  # Import the new function
)

# Disable gradients and set default dtype for PyTorch (replaces deprecated set_default_tensor_type)
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Physical parameters (module-level constants)
AU = 1.496e11  # 1 Astronomical Unit in meters

##### Settings #####

# JGA toggle (set to False to disable Jupiter Gravity Assist)
enable_jga = True

# Fixed initial conditions (set fix_initial_conditions=True to override chromosome's initial values)
fix_initial_conditions = False
fixed_C3 = 0.0
fixed_launch_angle_rad = 0.0
fixed_mu = 0.59

# Spacecraft launch conditions
C3_interval = np.array([10, 40])
Pi_launch_angle_interval = np.array([0, -10]) * np.pi / 180
mu_propellantPlusPayload_interval = np.array([0.5, 0.6])

# Fitness function settings
fitness_settings = {
    'R_SolarOBerth_Design': 0.3,
    'R_SolarOberth_LowerBound': 0.25,
    'R_SolarOberth_UpperBound': 0.7,
    't_mission_target': 25 * 365 * 24 * 60 * 60,  # 25 years in seconds
    'm_payload_target': 1000  # kg
}

# Neurocontroller
activation_function = torch.nn.Tanh
n_hidden_layers = 0
n_neuronsPerLayer = 20

# Trajectory simulation time and tolerances
total_simulation_time_thrust_phase = 4.5 * 365 * 24 * 60 * 60  # 4.5 years in seconds
rtol, atol = 1e-5, 1e-5

# Hyperparameter grid
mutation_rates = [0.1]
mutation_std_devs = [0.1]
population_sizes = [50]
generations = 5000

##### End Settings #####

# Seed values for reproducibility
seed_values = [42]

def evaluate_individual(args):
    (
        current_generation, generations, population_size, individual_nr, individual_chromosome, n_hidden_layers, n_neuronsPerLayer, activation_function, total_simulation_time_thrust_phase, rtol, atol, fitness_settings, mu_propellantPlusPayload_interval, C3_interval, Pi_launch_angle_interval, fix_initial_conditions, fixed_C3, fixed_launch_angle_rad, fixed_mu, enable_jga
    ) = args

    print(f"Now generation {current_generation}/{generations} - individual {individual_nr}/{population_size}")

    # Only fix initial conditions if explicitly configured (not by default)
    if fix_initial_conditions:
        individual_chromosome[0] = fixed_C3
        individual_chromosome[1] = fixed_launch_angle_rad
        individual_chromosome[2] = fixed_mu

    # Simulate trajectory
    simulation_results = simulate_trajectory(
        individual_chromosome, fitness_settings, n_hidden_layers, n_neuronsPerLayer, activation_function,
        total_simulation_time_thrust_phase, rtol, atol, enable_jga=enable_jga
    )
    (
        x, y, v_magnitude, Tx_h, Ty_h, Isp, t, delta_v, m_sc, m_EPS, m_structure, m_payload, m_propellant, JGA_results, t_200AU, P_0, u_IspEff, u_PowerThrottle, theta_thrust, flight_path_angle, accumulated_true_anomaly_normalized
    ) = simulation_results

    # Assess fitness
    fitness_score, subFitnessScores = evaluate_fitness(
        x, y, JGA_results['v_mag_afterJGA'].values[0], m_payload,
        t_200AU, fitness_settings, P_0, enable_jga=enable_jga
    )

    # Resimulate with tighter tolerances if fitness is high
    if fitness_score > 3.5:
        rtol_new, atol_new = 1e-6, 1e-6  # Fixed: was swapped (atol_new, rtol_new)
        if fitness_score > 4.5:
            rtol_new, atol_new = 1e-7, 1e-7

        simulation_results = simulate_trajectory(
            individual_chromosome, fitness_settings, n_hidden_layers, n_neuronsPerLayer, activation_function,
            total_simulation_time_thrust_phase, rtol_new, atol_new, enable_jga=enable_jga  # Fixed order: rtol, atol
        )
        (
            x, y, v_magnitude, Tx_h, Ty_h, Isp, t, delta_v, m_sc, m_EPS, m_structure, m_payload, m_propellant, JGA_results,
            t_200AU, P_0, u_IspEff, u_PowerThrottle, theta_thrust, flight_path_angle, accumulated_true_anomaly_normalized
        ) = simulation_results
        fitness_score, subFitnessScores = evaluate_fitness(
            x, y, JGA_results['v_mag_afterJGA'].values[0], m_payload, t_200AU, fitness_settings, P_0, enable_jga=enable_jga
        )

    # Reinitialize if fitness is zero
    if fitness_score == 0:
        individual_chromosome = initialize_individual(
            n_hidden_layers, n_neuronsPerLayer, activation_function,
            mu_propellantPlusPayload_interval, C3_interval, Pi_launch_angle_interval
        )

    # Compile results
    result = {'individual_nr': individual_nr, 'fitness_score': fitness_score, 'subFitnessScores': subFitnessScores, 'individual_chromosome': individual_chromosome, 'JGA_results': JGA_results, 'x': x, 'y': y, 'v_magnitude': v_magnitude, 'Tx_h': Tx_h, 'Ty_h': Ty_h, 'Isp': Isp, 't': t, 'delta_v': delta_v, 'm_sc': m_sc, 'm_EPS': m_EPS, 'm_structure': m_structure, 'm_payload': m_payload, 'm_propellant': m_propellant, 't_200AU': t_200AU, 'P_0': P_0, 'u_IspEff': u_IspEff, 'u_PowerThrottle': u_PowerThrottle, 'theta_thrust': theta_thrust, 'flight_path_angle': flight_path_angle, 'accumulated_true_anomaly_normalized': accumulated_true_anomaly_normalized}
    return result

def main():
    """Main function to run the evolutionary algorithm.
    
    All initialization that should only happen once (seeding, console clearing,
    directory creation) is done here to avoid repeated execution under Windows spawn.
    """
    # Suppress FutureWarnings and clear the console (Windows)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.system('cls')

    # Timestamp and results directory setup
    timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
    file_path = os.path.abspath(__file__)
    results_dir = os.path.join(os.path.dirname(file_path), "Results")
    os.makedirs(results_dir, exist_ok=True)

    # Create a folder with the name of the current timestamp in Trajectory Figures
    trajectory_figures_dir = os.path.join(os.path.dirname(file_path), "Trajectory Figures", timestamp)
    os.makedirs(trajectory_figures_dir, exist_ok=True)

    # Prepare combined CSV file
    combined_xlsx_path = os.path.join(results_dir, f'fitness-statistics-{timestamp}.xlsx')
    with open(combined_xlsx_path, 'w') as f:
        f.write('population_size,mutation_rate,mutation_std_dev,seed_value,generation,max_fitness,min_fitness,avg_fitness,var_fitness\n')

    # Initialize best fitness
    best_fitness = -np.inf

    # Initialize list to collect trajectory data
    all_trajectories_data = []

    # Set random seeds for reproducibility
    np.random.seed(seed_values[0])
    torch.manual_seed(seed_values[0])
    random.seed(seed_values[0])

    # Initialize first generation
    chromosomes_first_gen = initialize_first_gen(
        population_sizes[0], n_hidden_layers, n_neuronsPerLayer,
        activation_function, mu_propellantPlusPayload_interval,
        C3_interval, Pi_launch_angle_interval
    )

    # Create executor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for seed_value in seed_values:
            for population_size in population_sizes:

                # Reset seeds
                np.random.seed(seed_value)
                torch.manual_seed(seed_value)
                random.seed(seed_value)

                # Use the same first generation
                chromosomes = chromosomes_first_gen.copy()

                for mutation_rate in mutation_rates:
                    for mutation_std_dev in mutation_std_devs:

                        # Initialize fitness statistics
                        max_fitness, min_fitness, avg_fitness, var_fitness = [], [], [], []

                        # Initialize list to collect fittest individuals per generation
                        fittest_individuals_per_generation = []

                        # Evolution loop
                        for current_generation in range(1, generations + 1):
                            generation_fitness_scores = []

                            # Prepare arguments for parallel evaluation
                            args_list = [
                                (
                                    current_generation, generations, population_size, individual_nr,
                                    chromosomes[individual_nr - 1], n_hidden_layers, n_neuronsPerLayer,
                                    activation_function, total_simulation_time_thrust_phase, rtol, atol,
                                    fitness_settings, mu_propellantPlusPayload_interval,
                                    C3_interval, Pi_launch_angle_interval,
                                    fix_initial_conditions, fixed_C3, fixed_launch_angle_rad, fixed_mu,
                                    enable_jga
                                )
                                for individual_nr in range(1, population_size + 1)
                            ]

                            print(f"Generation {current_generation}: Evaluating individuals...")

                            # Evaluate individuals in parallel
                            results = list(executor.map(evaluate_individual, args_list))

                            print(f"Generation {current_generation}: Evaluation complete.")

                            # Initialize variables to track the fittest individual in this generation
                            gen_best_fitness = -np.inf
                            gen_best_result = None

                            # Collect and process results
                            for res in results:
                                individual_nr = res['individual_nr']
                                fitness_score = res['fitness_score']
                                individual_chromosome = res['individual_chromosome']

                                # Update chromosomes and fitness scores
                                chromosomes[individual_nr - 1] = individual_chromosome
                                generation_fitness_scores.append(fitness_score)

                                # Keep track of the fittest individual in this generation
                                if fitness_score > gen_best_fitness:
                                    gen_best_fitness = fitness_score
                                    gen_best_result = res

                                if individual_nr == 1:
                                    plot_and_save_trajectory(res["x"], res["y"], res["v_magnitude"], res["Tx_h"], res["Ty_h"], res["Isp"], res["JGA_results"], res["individual_chromosome"], current_generation, individual_nr, res["fitness_score"], res["subFitnessScores"], res["m_sc"], res["m_EPS"], res["m_structure"], res["m_payload"], res["m_propellant"], res["delta_v"], res["t_200AU"], res["P_0"], res["u_IspEff"], res["u_PowerThrottle"], res["theta_thrust"], res["flight_path_angle"], res["accumulated_true_anomaly_normalized"], res["t"], seed_value, trajectory_figures_dir)

                                # Collect data for saving
                                trajectory_data = {
                                    'timestamp': timestamp,
                                    'generation': current_generation,
                                    'individual_nr': individual_nr,
                                    'fitness_score': fitness_score,
                                    'x': res['x'],
                                    'y': res['y'],
                                    'v_magnitude': res['v_magnitude'],
                                    't': res['t']
                                }
                                all_trajectories_data.append(trajectory_data)

                            # Append the fittest individual of the generation to the list
                            gen_best_result.update({
                                'current_generation': current_generation,
                                'rtol': rtol,
                                'atol': atol,
                                'n_hidden_layers': n_hidden_layers,
                                'n_neuronsPerLayer': n_neuronsPerLayer,
                                'activation_function': activation_function,
                                'total_simulation_time_thrust_phase': total_simulation_time_thrust_phase,
                                'mutation_rate': mutation_rate,
                                'mutation_std_dev': mutation_std_dev,
                                'seed_value': seed_value,
                                'timestamp': timestamp
                            })
                            fittest_individuals_per_generation.append(gen_best_result)

                            # Check if the fittest individual of this generation is better than the best overall
                            if gen_best_fitness > best_fitness:
                                best_fitness = gen_best_fitness
                                update_and_save_fittest_solution(
                                    gen_best_fitness, timestamp, gen_best_result['individual_nr'], current_generation,
                                    gen_best_result['subFitnessScores'], gen_best_result['individual_chromosome'],
                                    gen_best_result['JGA_results'], gen_best_result['x'], gen_best_result['y'],
                                    gen_best_result['v_magnitude'], gen_best_result['Tx_h'], gen_best_result['Ty_h'],
                                    gen_best_result['Isp'], gen_best_result['t'], gen_best_result['delta_v'],
                                    gen_best_result['m_sc'], gen_best_result['m_EPS'], gen_best_result['m_structure'],
                                    gen_best_result['m_payload'], gen_best_result['m_propellant'],
                                    gen_best_result['t_200AU'], gen_best_result['P_0'], rtol, atol, n_hidden_layers,
                                    n_neuronsPerLayer, activation_function, total_simulation_time_thrust_phase,
                                    mutation_rate, mutation_std_dev, gen_best_result['u_IspEff'],
                                    gen_best_result['u_PowerThrottle'], gen_best_result['theta_thrust'],
                                    gen_best_result['flight_path_angle'],
                                    gen_best_result['accumulated_true_anomaly_normalized'], seed_value
                                )

                            # Perform selection and create next generation
                            chromosomes = tournament_selection(
                                chromosomes, generation_fitness_scores, n_hidden_layers,
                                n_neuronsPerLayer, mutation_rate, mutation_std_dev
                            )

                            # Record fitness statistics
                            max_fitness.append(np.max(generation_fitness_scores))
                            min_fitness.append(np.min(generation_fitness_scores))
                            avg_fitness.append(np.mean(generation_fitness_scores))
                            var_fitness.append(np.var(generation_fitness_scores))

                        # After all generations are completed, save the fittest individuals per generation
                        save_fittest_individuals_per_generation(fittest_individuals_per_generation)

                        # Save fitness statistics to CSV
                        with open(combined_xlsx_path, 'a') as f:
                            for gen in range(generations):
                                f.write(f"{population_size},{mutation_rate},{mutation_std_dev},{seed_value},{gen+1},"
                                        f"{max_fitness[gen]},{min_fitness[gen]},{avg_fitness[gen]},{var_fitness[gen]}\n")

    # After all processing is done, save the trajectory data
    save_trajectories_and_fitness(all_trajectories_data)

    print("\n\n######### EVOLUTIONARY ALGORITHM FINISHED #########\n")
    print(f"Best fitness: {best_fitness} after {generations} generations.\n")
    print(f"Results saved in {results_dir} as 'results-{timestamp}.xlsx'\n")
    print("Fittest solution saved as 'fittest_solution.csv'")


# Main execution block
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Necessary for Windows
    main()
