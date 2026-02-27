#!/usr/bin/env python
"""
SOMBRERO CLI Runner

Provides command-line interface for:
- simulate: Run a single trajectory simulation with a given chromosome
- optimize: Run the evolutionary optimization algorithm

Usage:
    python run.py simulate --config configs/paper_jga.json --chromosome configs/best_chromosome.json
    python run.py optimize --config configs/paper_jga.json
    python run.py optimize --config configs/paper_nojga.json
"""

# ============================================================================
# CRITICAL: Set thread limits BEFORE importing numpy/torch/scipy to prevent
# OpenMP/MKL oversubscription in multi-process runs (fixes libiomp5md.dll errors)
# ============================================================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch

# Disable gradients and set default dtype for PyTorch
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from config_io import load_config, load_chromosome, save_json, save_chromosome
from config_schema import get_activation_function, Config
from utils_repro import set_all_seeds, generate_run_id, create_summary, save_summary
from functions_trajectory_simulator_solve_ivp import simulate_trajectory, determine_t_200AU, GM_sun
from functions_evolutionary import (
    evaluate_fitness,
    initialize_first_gen,
    tournament_selection,
    initialize_individual,
    load_parameters_into_nn
)
from functions_results_processing import plot_and_save_trajectory

# ============================================================================
# Module-level evaluation function (required for multiprocessing)
# ============================================================================
def save_fittest_to_output_dir(best_fitness, run_id, individual_nr, current_generation,
                               subFitnessScores, individual_chromosome, JGA_results,
                               x, y, v_magnitude, Tx_h, Ty_h, Isp, t, delta_v, m_sc,
                               m_EPS, m_structure, m_payload, m_propellant, t_200AU, P_0,
                               rtol, atol, n_hidden_layers, n_neuronsPerLayer,
                               activation_function, total_simulation_time_thrust_phase,
                               mutation_rate, mutation_std_dev, u_IspEff, u_PowerThrottle,
                               theta_thrust, flight_path_angle, accumulated_true_anomaly_normalized,
                               seed, output_dir, timestamp):
    """Save fittest solution to output_dir/fittest_solution_timestamp.xlsx"""
    import os
    from datetime import datetime
    AU = 1.496e11
    
    R = np.sqrt(x**2 + y**2) / AU
    fittest_solution = pd.DataFrame({
        'timestamp': [timestamp],
        'seed_value': [seed],
        'individual_nr': [individual_nr],
        'current_generation': [current_generation],
        'fitness_score': [best_fitness],
        'subFitnessScores': [subFitnessScores],
        'individual_chromosome': [individual_chromosome],
        'JGA_results': [JGA_results],
        'm_sc': [m_sc],
        'm_EPS': [m_EPS],
        'm_structure': [m_structure],
        'm_payload': [m_payload],
        'm_propellant': [m_propellant],
        'delta_v': [delta_v],
        't_200AU': [t_200AU],
        'P_0': [P_0],
        'rtol': [rtol],
        'atol': [atol],
        'n_hidden_layers': [n_hidden_layers],
        'n_neuronsPerLayer': [n_neuronsPerLayer],
        'activation_function': [activation_function],
        'total_simulation_time_thrust_phase': [total_simulation_time_thrust_phase],
        'mutation_rate': [mutation_rate],
        'mutation_std_dev': [mutation_std_dev],
        'x': [x],
        'y': [y],
        'R': [R],
        'v_magnitude': [v_magnitude/29784],
        'flight_path_angle': [flight_path_angle],
        'accumulated_true_anomaly_normalized': [accumulated_true_anomaly_normalized],
        'Tx_h': [Tx_h],
        'Ty_h': [Ty_h],
        'Isp': [Isp],
        't': [t/(365*24*60*60)],
        'u_IspEff': [u_IspEff],
        'u_PowerThrottle': [u_PowerThrottle],
        'theta_thrust': [theta_thrust]
    })
    fittest_path = output_dir / f"fittest_solution_{timestamp}.xlsx"
    fittest_solution.to_excel(fittest_path, index=False, engine='openpyxl')


def evaluate_individual(args_tuple):
    """
    Evaluate a single individual's fitness. Must be at module level for pickling.
    """
    (
        current_generation, generations, population_size, individual_nr,
        individual_chromosome, n_hidden_layers, n_neuronsPerLayer,
        activation_function, total_simulation_time_thrust_phase, rtol, atol,
        fitness_settings, mu_propellantPlusPayload_interval,
        C3_interval, Pi_launch_angle_interval,
        fix_C3, fixed_C3, fix_launch_angle, fixed_launch_angle_rad, fix_mu, fixed_mu,
        enable_jga, config_Isp, config_efficiency
    ) = args_tuple
    
    print(f"Gen {current_generation}/{generations} - Individual {individual_nr}/{population_size}")
    
    # Copy chromosome to avoid mutation issues
    individual_chromosome = np.array(individual_chromosome, dtype=np.float32)
    
    # Apply fixed initial conditions individually if configured
    if fix_C3:
        individual_chromosome[0] = fixed_C3
    if fix_launch_angle:
        individual_chromosome[1] = fixed_launch_angle_rad
    if fix_mu:
        individual_chromosome[2] = fixed_mu
    
    # Simulate trajectory
    simulation_results = simulate_trajectory(
        individual_chromosome, fitness_settings, n_hidden_layers, n_neuronsPerLayer,
        activation_function, total_simulation_time_thrust_phase, rtol, atol,
        enable_jga=enable_jga, Isp_config=config_Isp, efficiency_config=config_efficiency
    )
    
    (x, y, v_magnitude, Tx_h, Ty_h, Isp, t, delta_v, m_sc, m_EPS, m_structure,
     m_payload, m_propellant, JGA_results, t_200AU, P_0, u_IspEff, u_PowerThrottle,
     theta_thrust, flight_path_angle, accumulated_true_anomaly_normalized) = simulation_results
    
    # Assess fitness
    fitness_score, subFitnessScores = evaluate_fitness(
        x, y, JGA_results['v_mag_afterJGA'].values[0], m_payload,
        t_200AU, fitness_settings, P_0, enable_jga=enable_jga
    )
    
    # Resimulate with tighter tolerances if fitness is high
    if fitness_score > 3.5:
        rtol_new, atol_new = 1e-6, 1e-6
        if fitness_score > 4.5:
            rtol_new, atol_new = 1e-7, 1e-7
        
        simulation_results = simulate_trajectory(
            individual_chromosome, fitness_settings, n_hidden_layers, n_neuronsPerLayer,
            activation_function, total_simulation_time_thrust_phase, rtol_new, atol_new,
            enable_jga=enable_jga, Isp_config=config_Isp, efficiency_config=config_efficiency
        )
        
        (x, y, v_magnitude, Tx_h, Ty_h, Isp, t, delta_v, m_sc, m_EPS, m_structure,
         m_payload, m_propellant, JGA_results, t_200AU, P_0, u_IspEff, u_PowerThrottle,
         theta_thrust, flight_path_angle, accumulated_true_anomaly_normalized) = simulation_results
        
        fitness_score, subFitnessScores = evaluate_fitness(
            x, y, JGA_results['v_mag_afterJGA'].values[0], m_payload,
            t_200AU, fitness_settings, P_0, enable_jga=enable_jga
        )
    
    # Reinitialize if fitness is zero
    if fitness_score == 0:
        individual_chromosome = initialize_individual(
            n_hidden_layers, n_neuronsPerLayer, activation_function,
            mu_propellantPlusPayload_interval, C3_interval, Pi_launch_angle_interval
        )
    
    return {
        'individual_nr': individual_nr,
        'fitness_score': fitness_score,
        'subFitnessScores': subFitnessScores,
        'individual_chromosome': individual_chromosome,
        'JGA_results': JGA_results,
        'x': x, 'y': y, 'v_magnitude': v_magnitude,
        'Tx_h': Tx_h, 'Ty_h': Ty_h, 'Isp': Isp,
        't': t, 'delta_v': delta_v, 'm_sc': m_sc,
        'm_EPS': m_EPS, 'm_structure': m_structure,
        'm_payload': m_payload, 'm_propellant': m_propellant,
        't_200AU': t_200AU, 'P_0': P_0,
        'u_IspEff': u_IspEff, 'u_PowerThrottle': u_PowerThrottle,
        'theta_thrust': theta_thrust, 'flight_path_angle': flight_path_angle,
        'accumulated_true_anomaly_normalized': accumulated_true_anomaly_normalized
    }


def cmd_simulate(args):
    """Run a single trajectory simulation."""
    # Load config
    config = load_config(args.config)
    
    # Load chromosome
    chromosome = load_chromosome(args.chromosome)
    individual_chromosome = np.array(chromosome, dtype=np.float32)
    
    # Set up output directory
    run_id = config.output.run_id or generate_run_id()
    output_dir = Path(config.output.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seeds for reproducibility
    set_all_seeds(config.evolution.seed)
    
    # Get activation function
    activation_function = get_activation_function(config.neuro.activation_name)
    
    # Build fitness settings dict for backward compatibility
    fitness_settings = config.fitness.to_dict()
    
    # Apply fixed initial conditions individually if configured
    if config.initial_conditions.fix_C3:
        individual_chromosome[0] = config.initial_conditions.fixed_C3
    if config.initial_conditions.fix_launch_angle:
        individual_chromosome[1] = config.initial_conditions.fixed_launch_angle_rad
    if config.initial_conditions.fix_mu:
        individual_chromosome[2] = config.initial_conditions.fixed_mu
    
    print(f"Running simulation with config: {args.config}")
    print(f"Chromosome length: {len(individual_chromosome)}")
    print(f"JGA enabled: {config.simulation.enable_jga}")
    print(f"Output directory: {output_dir}")
    
    # Run simulation
    simulation_results = simulate_trajectory(
        individual_chromosome,
        fitness_settings,
        config.neuro.n_hidden_layers,
        config.neuro.n_neurons_per_layer,
        activation_function,
        config.simulation.total_simulation_time_thrust_phase,
        config.simulation.rtol,
        config.simulation.atol,
        enable_jga=config.simulation.enable_jga,
        Isp_config=config.electric_propulsion.Isp,
        efficiency_config=config.electric_propulsion.efficiency
    )
    
    (x, y, v_magnitude, Tx_h, Ty_h, Isp, t, delta_v, m_sc, m_EPS, m_structure, 
     m_payload, m_propellant, JGA_results, t_200AU, P_0, u_IspEff, u_PowerThrottle, 
     theta_thrust, flight_path_angle, accumulated_true_anomaly_normalized) = simulation_results
    
    # Fallback: if t_200AU was not computed (e.g. spacecraft did not cross 5.2 AU
    # event boundary during thrust phase) but has solar escape velocity at the
    # final integration state, compute coast-phase t_200AU from that state.
    # This is a post-processing step for cmd_simulate reporting only and does not
    # affect optimizer logic or trajectory dynamics.
    if not np.isfinite(t_200AU):
        AU_val = 1.496e11
        r_final = np.sqrt(x[-1]**2 + y[-1]**2)
        # Reconstruct velocity components from the full solve_ivp output
        # v_magnitude is already available; for determine_t_200AU we need vx, vy.
        # Re-derive from trajectory arrays (last point):
        if len(x) >= 2:
            # Use finite-difference approximation of velocity at final point
            dt_last = t[-1] - t[-2]
            if dt_last > 0:
                vx_final = (x[-1] - x[-2]) / dt_last
                vy_final = (y[-1] - y[-2]) / dt_last
            else:
                vx_final = 0.0
                vy_final = 0.0
        else:
            vx_final = 0.0
            vy_final = 0.0
        v_final_mag = np.sqrt(vx_final**2 + vy_final**2)
        v_esc_final = np.sqrt(2 * GM_sun / r_final)
        if v_final_mag > v_esc_final:
            state_final = [x[-1], y[-1], vx_final, vy_final, m_sc[-1], delta_v[-1]]
            t_200AU = determine_t_200AU(state_final, t[-1],
                                         config.simulation.rtol, config.simulation.atol)
            if np.isfinite(t_200AU):
                print(f"[simulate] Fallback coast t_200AU: {t_200AU / (365.25 * 24 * 3600):.2f} years")
    
    # Evaluate fitness
    fitness_score, subFitnessScores = evaluate_fitness(
        x, y, JGA_results['v_mag_afterJGA'].values[0], m_payload,
        t_200AU, fitness_settings, P_0, enable_jga=config.simulation.enable_jga
    )
    
    print(f"\n=== Simulation Results ===")
    print(f"Fitness Score: {fitness_score:.4f}")
    print(f"Payload Mass: {m_payload:.2f} kg")
    print(f"Time to 200 AU: {t_200AU / (365.25 * 24 * 3600):.2f} years" if np.isfinite(t_200AU) else "Time to 200 AU: Not reachable")
    print(f"Initial Power P_0: {P_0:.2f} W")
    if config.simulation.enable_jga:
        print(f"Velocity after JGA: {JGA_results['v_mag_afterJGA'].values[0]:.2f} m/s")
        print(f"JGA delta-v: {JGA_results['JGA_delta_v'].values[0]:.2f} m/s")
    else:
        print(f"(JGA disabled — JGA-specific metrics not applicable)")
    
    # Create summary
    jga_v_after = float(JGA_results['v_mag_afterJGA'].values[0]) if config.simulation.enable_jga and pd.notna(JGA_results['v_mag_afterJGA'].values[0]) else None
    jga_dv = float(JGA_results['JGA_delta_v'].values[0]) if config.simulation.enable_jga and pd.notna(JGA_results['JGA_delta_v'].values[0]) else None
    scalars = {
        'm_payload': float(m_payload),
        't_200AU_years': float(t_200AU / (365.25 * 24 * 3600)) if np.isfinite(t_200AU) else None,
        't_200AU_seconds': float(t_200AU) if np.isfinite(t_200AU) else None,
        'P_0': float(P_0),
        'v_after_jga': jga_v_after,
        'jga_delta_v': jga_dv,
        'm_sc_initial': float(m_sc[0]),
        'm_sc_final': float(m_sc[-1]),
        'm_propellant': float(m_propellant),
        'm_structure': float(m_structure),
        'm_EPS': float(m_EPS),
        'delta_v_total': float(delta_v[-1]),
        'subFitnessScores': {k: float(v) if pd.notna(v) else None 
                           for k, v in subFitnessScores.iloc[0].to_dict().items()}
    }
    
    summary = create_summary(
        config=config,
        chromosome=individual_chromosome.tolist(),
        fitness_score=fitness_score,
        scalars=scalars,
        seed=config.evolution.seed
    )
    
    # Save summary
    summary_path = save_summary(summary, output_dir, "summary.json")
    print(f"\nSummary saved to: {summary_path}")
    
    # Save plot if configured
    if config.output.save_plots:
        try:
            plot_and_save_trajectory(
                x, y, v_magnitude, Tx_h, Ty_h, Isp, JGA_results, 
                individual_chromosome, 0, 1, fitness_score, subFitnessScores,
                m_sc, m_EPS, m_structure, m_payload, m_propellant, delta_v, 
                t_200AU, P_0, u_IspEff, u_PowerThrottle, theta_thrust, 
                flight_path_angle, accumulated_true_anomaly_normalized, t,
                config.evolution.seed, str(output_dir)
            )
            print(f"Trajectory plot saved to: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not save plot: {e}")
    
    return 0


def cmd_optimize(args):
    """Run evolutionary optimization with parallel evaluation."""
    import concurrent.futures
    import multiprocessing
    
    # Set spawn mode for Windows compatibility (must be done before creating pool)
    multiprocessing.set_start_method('spawn', force=True)
    
    # Load config
    config = load_config(args.config)
    
    # Set up output directory with JGA status in folder name
    run_id = config.output.run_id or generate_run_id()
    jga_status = "JGA" if config.simulation.enable_jga else "noJGA"
    run_id_with_jga = f"{run_id}-{jga_status}"
    output_dir = Path(config.output.output_dir) / run_id_with_jga
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seeds for reproducibility
    seed = config.evolution.seed
    set_all_seeds(seed)
    
    # Get activation function
    activation_function = get_activation_function(config.neuro.activation_name)
    
    # Build fitness settings dict
    fitness_settings = config.fitness.to_dict()
    
    # Get intervals
    C3_interval = config.launch_intervals.get_C3_interval()
    Pi_launch_angle_interval = config.launch_intervals.get_launch_angle_interval_rad()
    mu_propellantPlusPayload_interval = config.launch_intervals.get_mu_interval()
    
    # Load warm start chromosome if configured
    warm_start_chromosome = None
    if config.init.init_mode == "warm" and config.init.warm_start_chromosome_path:
        warm_start_chromosome = load_chromosome(config.init.warm_start_chromosome_path)
        print(f"Loaded warm start chromosome from: {config.init.warm_start_chromosome_path}")
    
    print(f"Starting optimization with config: {args.config}")
    print(f"Population size: {config.evolution.population_size}")
    print(f"Generations: {config.evolution.generations}")
    print(f"JGA enabled: {config.simulation.enable_jga}")
    print(f"Init mode: {config.init.init_mode}")
    print(f"Output directory: {output_dir}")
    
    # Suppress FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
    
    # Trajectory figures directory (only subfolder)
    trajectory_figures_dir = output_dir / "figures"
    trajectory_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare fitness statistics CSV file (saved directly in output_dir)
    if config.output.save_excel:
        combined_xlsx_path = output_dir / f'fitness-statistics-{timestamp}.csv'
        with open(combined_xlsx_path, 'w') as f:
            f.write('generation,max_fitness,min_fitness,avg_fitness,var_fitness\n')
    
    # Initialize best fitness
    best_fitness = -np.inf
    best_chromosome = None
    
    # Initialize first generation
    chromosomes = initialize_first_gen(
        config.evolution.population_size, 
        config.neuro.n_hidden_layers, 
        config.neuro.n_neurons_per_layer,
        activation_function, 
        mu_propellantPlusPayload_interval,
        C3_interval, 
        Pi_launch_angle_interval,
        init_mode=config.init.init_mode,
        warm_start_chromosome=warm_start_chromosome,
        warm_start_mix_fraction=config.init.warm_start_mix_fraction
    )
    
    # Evolution parameters
    population_size = config.evolution.population_size
    generations = config.evolution.generations
    n_hidden_layers = config.neuro.n_hidden_layers
    n_neuronsPerLayer = config.neuro.n_neurons_per_layer
    mutation_rate = config.evolution.mutation_rate
    mutation_std_dev = config.evolution.mutation_std_dev
    
    # Simulation parameters
    rtol = config.simulation.rtol
    atol = config.simulation.atol
    total_simulation_time_thrust_phase = config.simulation.total_simulation_time_thrust_phase
    enable_jga = config.simulation.enable_jga
    
    # Fixed initial conditions (individual flags)
    fix_C3 = config.initial_conditions.fix_C3
    fixed_C3 = config.initial_conditions.fixed_C3
    fix_launch_angle = config.initial_conditions.fix_launch_angle
    fixed_launch_angle_rad = config.initial_conditions.fixed_launch_angle_rad
    fix_mu = config.initial_conditions.fix_mu
    fixed_mu = config.initial_conditions.fixed_mu
    
    # Electric propulsion parameters
    Isp = config.electric_propulsion.Isp
    efficiency = config.electric_propulsion.efficiency
    
    # Number of parallel workers
    n_workers = config.evolution.n_workers if hasattr(config.evolution, 'n_workers') else None
    if n_workers is None or n_workers <= 0:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Using {n_workers} parallel workers")
    
    # Fitness statistics
    max_fitness_list, min_fitness_list, avg_fitness_list, var_fitness_list = [], [], [], []
    
    # Evolution loop with parallel evaluation using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for current_generation in range(1, generations + 1):
            generation_fitness_scores = []
            
            print(f"\n=== Generation {current_generation}/{generations} ===")
            
            # Prepare arguments for parallel evaluation
            args_list = [
                (
                    current_generation, generations, population_size, individual_nr,
                    chromosomes[individual_nr - 1], n_hidden_layers, n_neuronsPerLayer,
                    activation_function, total_simulation_time_thrust_phase, rtol, atol,
                    fitness_settings, mu_propellantPlusPayload_interval,
                    C3_interval, Pi_launch_angle_interval,
                    fix_C3, fixed_C3, fix_launch_angle, fixed_launch_angle_rad, fix_mu, fixed_mu,
                    enable_jga, Isp, efficiency
                )
                for individual_nr in range(1, population_size + 1)
            ]
            
            # Evaluate individuals in parallel
            print(f"Generation {current_generation}: Evaluating {population_size} individuals in parallel...")
            results = list(executor.map(evaluate_individual, args_list))
            print(f"Generation {current_generation}: Evaluation complete.")
        
            # Track best in generation
            gen_best_fitness = -np.inf
            gen_best_result = None
            
            # Process results
            for res in results:
                individual_nr = res['individual_nr']
                fitness_score = res['fitness_score']
                individual_chromosome = res['individual_chromosome']
                
                chromosomes[individual_nr - 1] = individual_chromosome
                generation_fitness_scores.append(fitness_score)
                
                if fitness_score > gen_best_fitness:
                    gen_best_fitness = fitness_score
                    gen_best_result = res
            
            # Update best overall
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_chromosome = gen_best_result['individual_chromosome']
                
                # Save best chromosome
                save_chromosome(
                    list(best_chromosome), 
                    output_dir / "best_chromosome.json",
                    metadata={
                        'generation': current_generation,
                        'fitness_score': float(best_fitness)
                    }
                )
                print(f"New best fitness: {best_fitness:.4f} at generation {current_generation}")
                
                # Save fittest_solution to output_dir with timestamp
                try:
                    save_fittest_to_output_dir(
                        gen_best_fitness, run_id, gen_best_result['individual_nr'], current_generation,
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
                        gen_best_result['accumulated_true_anomaly_normalized'], seed, output_dir, timestamp
                    )
                except Exception as e:
                    print(f"Warning: Could not save fittest_solution.xlsx: {e}")
            
            # Save trajectory plot for the fittest individual (individual #1) if configured
            if config.output.save_plots:
                # Find individual #1's results (the fittest from previous generation)
                individual_1_result = next((res for res in results if res['individual_nr'] == 1), None)
                if individual_1_result:
                    try:
                        plot_and_save_trajectory(
                            individual_1_result['x'], individual_1_result['y'], individual_1_result['v_magnitude'],
                            individual_1_result['Tx_h'], individual_1_result['Ty_h'], individual_1_result['Isp'],
                            individual_1_result['JGA_results'], individual_1_result['individual_chromosome'],
                            current_generation, 1, individual_1_result['fitness_score'], individual_1_result['subFitnessScores'],
                            individual_1_result['m_sc'], individual_1_result['m_EPS'], individual_1_result['m_structure'],
                            individual_1_result['m_payload'], individual_1_result['m_propellant'], individual_1_result['delta_v'],
                            individual_1_result['t_200AU'], individual_1_result['P_0'], individual_1_result['u_IspEff'],
                            individual_1_result['u_PowerThrottle'], individual_1_result['theta_thrust'],
                            individual_1_result['flight_path_angle'], individual_1_result['accumulated_true_anomaly_normalized'],
                            individual_1_result['t'], seed, str(trajectory_figures_dir)
                        )
                        print(f"Saved trajectory plot for fittest individual (Gen {current_generation}, Ind 1, Fitness {individual_1_result['fitness_score']:.4f})")
                    except Exception as e:
                        print(f"Warning: Could not save trajectory plot for Gen {current_generation}: {e}")
            
            # Update fittest_solution.xlsx every 10 generations
            if current_generation % 10 == 0 and gen_best_result:
                try:
                    save_fittest_to_output_dir(
                        gen_best_fitness, run_id, gen_best_result['individual_nr'], current_generation,
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
                        gen_best_result['accumulated_true_anomaly_normalized'], seed, output_dir, timestamp
                    )
                    print(f"Updated fittest_solution_{timestamp}.xlsx at generation {current_generation}")
                except Exception as e:
                    print(f"Warning: Could not update fittest_solution.xlsx: {e}")
            
            # Perform selection
            chromosomes = tournament_selection(
                chromosomes, generation_fitness_scores, n_hidden_layers,
                n_neuronsPerLayer, mutation_rate, mutation_std_dev
            )
        
            # Record statistics
            max_fitness_list.append(np.max(generation_fitness_scores))
            min_fitness_list.append(np.min(generation_fitness_scores))
            avg_fitness_list.append(np.mean(generation_fitness_scores))
            var_fitness_list.append(np.var(generation_fitness_scores))
            
            print(f"Generation {current_generation}: Max={max_fitness_list[-1]:.4f}, "
                  f"Avg={avg_fitness_list[-1]:.4f}, Min={min_fitness_list[-1]:.4f}")
            
            # Save statistics
            if config.output.save_excel:
                with open(combined_xlsx_path, 'a') as f:
                    f.write(f"{current_generation},{max_fitness_list[-1]},{min_fitness_list[-1]},"
                            f"{avg_fitness_list[-1]},{var_fitness_list[-1]}\n")
    
    # Final summary
    print(f"\n=== Optimization Complete ===")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"  - best_chromosome.json")
    print(f"  - fitness-statistics-{timestamp}.csv")
    print(f"  - fittest_solution.csv")
    print(f"  - optimization_summary.json")
    print(f"  - figures/ (trajectory plots)")
    
    # Save final summary
    final_summary = {
        'run_id': run_id,
        'config_path': str(args.config),
        'best_fitness': float(best_fitness),
        'generations_completed': generations,
        'population_size': population_size,
        'seed': seed,
        'enable_jga': enable_jga
    }
    save_json(output_dir / "optimization_summary.json", final_summary)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="SOMBRERO Trajectory Optimization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py simulate --config configs/paper_jga.json --chromosome configs/best_chromosome.json
  python run.py optimize --config configs/paper_jga.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simulate subcommand
    simulate_parser = subparsers.add_parser(
        'simulate', 
        help='Run a single trajectory simulation with a given chromosome'
    )
    simulate_parser.add_argument(
        '--config', '-c', 
        required=True, 
        help='Path to configuration JSON file'
    )
    simulate_parser.add_argument(
        '--chromosome', '-chr', 
        required=True, 
        help='Path to chromosome JSON file'
    )
    
    # Optimize subcommand
    optimize_parser = subparsers.add_parser(
        'optimize', 
        help='Run evolutionary optimization'
    )
    optimize_parser.add_argument(
        '--config', '-c', 
        required=True, 
        help='Path to configuration JSON file'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == 'simulate':
        return cmd_simulate(args)
    elif args.command == 'optimize':
        return cmd_optimize(args)
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
