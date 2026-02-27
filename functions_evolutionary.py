# ============================================================================
# CRITICAL: Set thread limits BEFORE importing numpy/torch/scipy to prevent
# OpenMP/MKL oversubscription in multi-process runs (fixes libiomp5md.dll errors)
# ============================================================================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # LAST RESORT ONLY - unsafe workaround

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import random
from typing import Optional, List


# Disable gradients globally and set default dtype (replaces deprecated set_default_tensor_type)
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)

AU = 1.496e11  # 1 Astronomical Unit in meters
GM_sun = 1.327e20  # gravitational parameter of the Sun in m^3/s^2


# define neural network class
class initialize_neural_network(nn.Module):
    def __init__(self, n_hidden_layers, n_neuronsPerLayer, activation_function):
        super().__init__()
        activation_fn = activation_function

        self.inputLayer = nn.Sequential(
            nn.Linear(4, n_neuronsPerLayer),
            activation_fn()
        )

        self.hiddenLayers = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(n_neuronsPerLayer, n_neuronsPerLayer),
                activation_fn()
            ) for _ in range(n_hidden_layers)]
        )

        self.outputLayer = nn.Sequential(
            nn.Linear(n_neuronsPerLayer, 2)
        )

    def forward(self, R, true_anomaly, flight_path_angle, V_magnitude):
        input_tensor = torch.tensor([R, true_anomaly, flight_path_angle, V_magnitude], dtype=torch.float32)
        out_inputLayer = self.inputLayer(input_tensor)
        out_hiddenLayers = self.hiddenLayers(out_inputLayer)
        out_outputLayer = self.outputLayer(out_hiddenLayers)

        out_outputLayer[0] = -torch.exp(-out_outputLayer[0]**2) # activation for theta_thrust
        out_outputLayer[1:] = 1-torch.exp(-5*out_outputLayer[1:]**2)# activation for u_PowerThrottle || NECESSITY for same input, activation function of phi must: -->0 and activation of u_P must -->1

        return out_outputLayer.detach().numpy()

# create genome from neural network
def create_genome(C3, gamma, mu, neural_net):
    genome = [C3, gamma, mu]

    for layer in neural_net.children():
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Linear):
                    genome.extend(sub_layer.bias.detach().numpy().astype(np.float32).tolist())
                    genome.extend(sub_layer.weight.detach().numpy().flatten().astype(np.float32).tolist())

    return np.array(genome)


def initialize_individual(hidden_layers, hidden_nodes, activation_function, mu_propellantPlusPayload_interval, C3_interval, Pi_launch_angle_interval):
    """
    Initialize a single individual for the evolutionary algorithm.

    Args:
        hidden_layers (int): Number of hidden layers in the neural network.
        hidden_nodes (int): Number of nodes in each hidden layer.
        activation_function (function): Activation function for the neural network.
        mu_propellantPlusPayload_interval (np.array): Interval for the combined mass fraction (mu).
        C3_interval (np.array): Interval for the C3 parameter.
        Pi_launch_angle_interval (np.array): Interval for the launch angle.

    Returns:
        chromosome: Initialized chromosome.
    """
    # Initialize neural network using the provided class
    NN = initialize_neural_network(n_hidden_layers=hidden_layers, n_neuronsPerLayer=hidden_nodes, activation_function=activation_function)

    # Initialize spacecraft parameters: mu, C3, launch_angle
    C3 = np.random.uniform(*C3_interval)
    launch_angle = np.random.uniform(*Pi_launch_angle_interval)
    mu_PayloadPropellantInitial = np.random.uniform(*mu_propellantPlusPayload_interval)

    # Use create_genome function to create a genome
    chromosome = create_genome(C3, launch_angle, mu_PayloadPropellantInitial, NN)

    return chromosome

def initialize_first_gen(population_size, hidden_layers, hidden_nodes, activation_function, mu_propellantPlusPayload_interval, C3_interval, Pi_launch_angle_interval,
                         init_mode: str = "cold", warm_start_chromosome: Optional[List[float]] = None, 
                         warm_start_mix_fraction: float = 0.5):
    """
    Initialize the first generation of chromosomes for the evolutionary algorithm.

    Args:
        population_size (int): The size of the population.
        hidden_layers (int): Number of hidden layers in the neural network.
        hidden_nodes (int): Number of nodes in each hidden layer.
        activation_function (function): Activation function for the neural network.
        mu_propellantPlusPayload_interval (np.array): Interval for the combined mass fraction (mu).
        C3_interval (np.array): Interval for the C3 parameter.
        Pi_launch_angle_interval (np.array): Interval for the launch angle.
        init_mode (str): "cold" for all random, "warm" to seed some individuals from chromosome.
        warm_start_chromosome (list): Chromosome to use for warm start (if init_mode="warm").
        warm_start_mix_fraction (float): Fraction of population to warm-start (0..1).

    Returns:
        list: List of initialized chromosomes.
    """
    np.set_printoptions(edgeitems=np.inf, linewidth=np.inf, formatter={'float': '{: .3e}'.format})

    chromosomes = []

    if init_mode == "warm" and warm_start_chromosome is not None:
        # Warm start: a fraction of the population is based on the provided chromosome
        n_warm = int(population_size * warm_start_mix_fraction)
        n_cold = population_size - n_warm
        
        warm_chromosome = np.array(warm_start_chromosome, dtype=np.float32)
        
        # Validate chromosome length
        expected_length = _compute_expected_genome_length(hidden_layers, hidden_nodes)
        if len(warm_chromosome) != expected_length:
            raise ValueError(f"Warm start chromosome length {len(warm_chromosome)} does not match "
                           f"expected {expected_length} for architecture.")
        
        # Add warm-started individuals (with small noise to create diversity)
        for i in range(n_warm):
            if i == 0:
                # Keep one exact copy
                chromosomes.append(warm_chromosome.copy())
            else:
                # Add small gaussian noise to create diversity
                noisy_chromosome = warm_chromosome.copy()
                noise = np.random.normal(0, 0.01, size=len(noisy_chromosome))
                # Don't add noise to first 3 params (C3, gamma, mu) to keep them valid
                noisy_chromosome[3:] += noisy_chromosome[3:] * noise[3:]
                chromosomes.append(noisy_chromosome)
        
        # Fill the rest with random individuals
        for _ in range(n_cold):
            chromosome = initialize_individual(hidden_layers, hidden_nodes, activation_function,
                                              mu_propellantPlusPayload_interval, C3_interval, 
                                              Pi_launch_angle_interval)
            chromosomes.append(chromosome)
    else:
        # Cold start: all random initialization
        for _ in range(population_size):
            chromosome = initialize_individual(hidden_layers, hidden_nodes, activation_function,
                                              mu_propellantPlusPayload_interval, C3_interval, 
                                              Pi_launch_angle_interval)
            chromosomes.append(chromosome)

    return chromosomes


def _compute_expected_genome_length(n_hidden_layers: int, n_neurons_per_layer: int) -> int:
    """Compute expected chromosome length for given architecture."""
    input_size = 4
    output_size = 2
    initial_params = 3  # C3, gamma, mu
    input_layer_size = n_neurons_per_layer * (1 + input_size)
    hidden_layer_size = n_hidden_layers * n_neurons_per_layer * (1 + n_neurons_per_layer)
    output_layer_size = output_size * (1 + n_neurons_per_layer)
    return initial_params + input_layer_size + hidden_layer_size + output_layer_size


def load_parameters_into_nn(genome, n_hidden_layers, n_neuronsPerLayer, activation_function):
    C3, gamma, mu = genome[:3]
    nn_params = genome[3:]
    nn_model = initialize_neural_network(n_hidden_layers, n_neuronsPerLayer, activation_function)
    
    index = 0
    for layer in nn_model.children():
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Linear):
                    bias_length = sub_layer.bias.shape[0]
                    weight_length = sub_layer.weight.numel()

                    sub_layer.bias.data = torch.tensor(nn_params[index:index + bias_length], dtype=torch.float32)
                    index += bias_length
                    sub_layer.weight.data = torch.tensor(nn_params[index:index + weight_length], dtype=torch.float32).reshape(sub_layer.weight.shape)
                    index += weight_length
    
    return nn_model

def evaluate_fitness(x, y, v_mag_AfterJGA, m_payload, t_200AU, fitness_settings, P_0, enable_jga=True):
    """
    Evaluate fitness of a trajectory.
    
    Args:
        x, y: Position arrays
        v_mag_AfterJGA: Velocity magnitude after JGA (or current velocity if JGA disabled)
        m_payload: Payload mass in kg
        t_200AU: Time to reach 200 AU in seconds
        fitness_settings: Dict with fitness parameters
        P_0: Initial power
        enable_jga: Whether JGA is enabled (affects fitness calculation stages)
    
    Returns:
        Tuple of (fitness_score, subFitnessScores DataFrame)
    """

    # initialize scores
    fitness_score = 0
    fitness_score_SolarOberth = 0
    fitness_score_ApproachJupiter = 0
    fitness_score_SolarEscape = 0
    fitness_score_Time200AU = 0
    fitness_score_Payload = 0

    # load settings
    R_SolarOBerth_Design = fitness_settings['R_SolarOBerth_Design']
    R_SolarOberth_LowerBound = fitness_settings['R_SolarOberth_LowerBound']
    R_SolarOberth_UpperBound = fitness_settings['R_SolarOberth_UpperBound']
    t_mission_target = fitness_settings['t_mission_target']
    m_payload_target = fitness_settings['m_payload_target']

    R = np.sqrt(x**2 + y**2) / AU
    v_escape_heliocentric_from_JupiterOrbit = np.sqrt(2 * GM_sun / (5.2 * AU))
    
    # Calculate the average of R
    R_avg = np.mean(R)

    # Find all local maxima in R
    local_max_indices = []
    for i in range(10, len(R) - 1):
        if R[i] > R[i-1] and R[i] > R[i+1]:
            local_max_indices.append(i)

    # Check the combined conditions for a Solar Oberth Candidate (Filter)
    if len(local_max_indices) == 1 and 1.2 < R_avg and 2 <= R[local_max_indices[0]] and 1 < R[-1] and P_0 > 0:
        local_max_index = local_max_indices[0]
        # Determine R_min as the minimum value of R after the first local maximum
        R_min = np.min(R[local_max_index + 1:])
        R_min_index = np.argmin(R[local_max_index + 1:]) + local_max_index + 1
        R_max = np.max(R[R_min_index:])

        # Evolution Stage 1: Solar Oberth Maneuver (same for both JGA and no-JGA)
        if R_min < R_SolarOBerth_Design:
            fitness_score_SolarOberth = np.exp(-1000 * (R_min - R_SolarOBerth_Design)**2)
        else:
            fitness_score_SolarOberth = np.exp(-50 * (R_min - R_SolarOBerth_Design)**2)
        fitness_score += fitness_score_SolarOberth

        if enable_jga:
            # ===== JGA ENABLED PATH =====
            # Evolution Stage 2: Approaching Jupiter
            if R_SolarOberth_LowerBound < R_min < R_SolarOberth_UpperBound:
                
                if R_max < 5.2:
                    fitness_score_ApproachJupiter = R_max / 5.2
                    fitness_score += fitness_score_ApproachJupiter
                if R_max >= 5.2:
                    fitness_score_ApproachJupiter = 1
                    fitness_score += fitness_score_ApproachJupiter       

                    # Evolution Stage 3: Reaching Solar Escape Velocity after JGA
                    if v_mag_AfterJGA < v_escape_heliocentric_from_JupiterOrbit:
                        energy_needed_for_escape = v_escape_heliocentric_from_JupiterOrbit**2 / 2
                        energy_actual = v_mag_AfterJGA**2 / 2
                        fitness_score_SolarEscape = energy_actual / energy_needed_for_escape
                        fitness_score += fitness_score_SolarEscape
                    if v_mag_AfterJGA > 18470:  # reached solar escape velocity
                        fitness_score_SolarEscape = 1
                        fitness_score += fitness_score_SolarEscape

                        # Evolution Stage 4: Reach 200 AU within given timeframe
                        if t_200AU > t_mission_target:
                            fitness_score_Time200AU = -np.tanh(0.5 * (t_200AU / t_mission_target - 1)) + 1
                            fitness_score += fitness_score_Time200AU

                        if t_200AU <= t_mission_target:
                            fitness_score_Time200AU = 1
                            fitness_score += fitness_score_Time200AU

                            # Evolution Stage 5: Maximize Payload Mass
                            if t_200AU <= 1.02 * t_mission_target:
                                fitness_score_Payload = m_payload / m_payload_target
                                fitness_score += fitness_score_Payload
        else:
            # ===== JGA DISABLED PATH =====
            # Still approach Jupiter, but without gravity assist rotation
            # Stage 2: Approaching Jupiter (same as JGA case)
            if R_SolarOberth_LowerBound < R_min < R_SolarOberth_UpperBound:
                
                if R_max < 5.2:
                    fitness_score_ApproachJupiter = R_max / 5.2
                    fitness_score += fitness_score_ApproachJupiter
                if R_max >= 5.2:
                    fitness_score_ApproachJupiter = 1
                    fitness_score += fitness_score_ApproachJupiter
                    
                    # Stage 3: Solar Escape (using final velocity without JGA rotation)
                    # Use final heliocentric velocity vs escape speed from final position
                    r_final = R[-1] * AU  # Final radius in meters
                    v_escape_at_final = np.sqrt(2 * GM_sun / r_final)
                    
                    # v_mag_AfterJGA is actually the final velocity when JGA is disabled
                    if v_mag_AfterJGA >= v_escape_at_final:
                        # Already hyperbolic
                        fitness_score_SolarEscape = 1.0
                        fitness_score += fitness_score_SolarEscape
                        
                        # Stage 4: Time to 200 AU (if hyperbolic)
                        if np.isfinite(t_200AU):
                            if t_200AU > t_mission_target:
                                fitness_score_Time200AU = -np.tanh(0.5 * (t_200AU / t_mission_target - 1)) + 1
                            else:
                                fitness_score_Time200AU = 1.0
                            fitness_score += fitness_score_Time200AU
                            
                            # Stage 5: Payload (only if reasonable time to 200AU)
                            if t_200AU <= 1 * t_mission_target:
                                fitness_score_Payload = m_payload / m_payload_target
                                fitness_score += fitness_score_Payload
                    else:
                        # Not hyperbolic - partial score
                        energy_actual = v_mag_AfterJGA**2 / 2
                        energy_needed = v_escape_at_final**2 / 2
                        fitness_score_SolarEscape = min(energy_actual / energy_needed, 1.0)
                        fitness_score += fitness_score_SolarEscape

    # Create subFitnessScores DataFrame
    # In both JGA and noJGA modes, all stages are scored (Approach Jupiter is relevant in both)
    subFitnessScores = pd.DataFrame({
        'Solar Oberth': [fitness_score_SolarOberth], 
        'Approach Jupiter': [fitness_score_ApproachJupiter], 
        'Solar Escape': [fitness_score_SolarEscape], 
        'Time to 200 AU': [fitness_score_Time200AU], 
        'Payload': [fitness_score_Payload]
    })
    
    # Ensure fitness_score is never NaN - replace with 0 if needed
    if not np.isfinite(fitness_score):
        fitness_score = 0.0
    
    print(f"Total Fitness score: {fitness_score}, subFitnessScores:\n{subFitnessScores}")

    return fitness_score, subFitnessScores

def one_point_crossover(parent1_genome, parent2_genome, n_hidden_layers, n_neuronsPerLayer):
    assert len(parent1_genome) == len(parent2_genome), "Genomes must be of the same length"
    
    input_size = 4  # fixed input size as per the neural network definition
    output_size = 2  # fixed output size as per the neural network definition

    # Compute the size of each node's parameters (1 bias + input_size weights for input layer, 
    # 1 bias + n_neuronsPerLayer weights for hidden layers, 1 bias + n_neuronsPerLayer weights for output layer)
    input_node_size = 1 + input_size
    hidden_node_size = 1 + n_neuronsPerLayer
    output_node_size = 1 + n_neuronsPerLayer

    # Calculate the total number of parameters in the genome excluding C3, gamma, mu
    genome_start_index = 3  # C3, gamma, mu

    input_layer_size = n_neuronsPerLayer * input_node_size
    hidden_layer_size = n_hidden_layers * n_neuronsPerLayer * hidden_node_size
    output_layer_size = output_size * output_node_size

    total_genome_length = input_layer_size + hidden_layer_size + output_layer_size

    assert len(parent1_genome) == total_genome_length + genome_start_index, "Genome length does not match the expected length"

    # Determine the valid crossover points
    possible_crossover_points = []
    current_index = genome_start_index

    # Input layer nodes
    for _ in range(n_neuronsPerLayer):
        possible_crossover_points.append(current_index)
        current_index += input_node_size

    # Hidden layer nodes
    for _ in range(n_hidden_layers):
        for _ in range(n_neuronsPerLayer):
            possible_crossover_points.append(current_index)
            current_index += hidden_node_size

    # Output layer nodes
    for _ in range(output_size):
        possible_crossover_points.append(current_index)
        current_index += output_node_size

    # Add the end of the genome as a possible crossover point
    possible_crossover_points.append(len(parent1_genome))

    # Select a random crossover point
    crossover_point = np.random.choice(possible_crossover_points)

    print(f"Performing one-point crossover at index {crossover_point}")

    # Perform crossover
    child1_genome = np.concatenate((parent1_genome[:crossover_point], parent2_genome[crossover_point:]))
    child2_genome = np.concatenate((parent2_genome[:crossover_point], parent1_genome[crossover_point:]))

    return child1_genome, child2_genome

def uniform_crossover(parent1_genome, parent2_genome, n_hidden_layers, n_neuronsPerLayer):
    print(f"len(parent1_genome): {len(parent1_genome)} | len(parent2_genome): {len(parent2_genome)}")

    assert len(parent1_genome) == len(parent2_genome), "Genomes must be of the same length"
    
    input_size = 4  # fixed input size as per the neural network definition
    output_size = 2  # fixed output size as per the neural network definition

    # Compute the size of each node's parameters (1 bias + input_size weights for input layer, 
    # 1 bias + n_neuronsPerLayer weights for hidden layers, 1 bias + n_neuronsPerLayer weights for output layer)
    input_node_size = 1 + input_size
    hidden_node_size = 1 + n_neuronsPerLayer
    output_node_size = 1 + n_neuronsPerLayer

    # Calculate the total number of parameters in the genome excluding C3, gamma, mu
    genome_start_index = 3  # C3, gamma, mu

    input_layer_size = n_neuronsPerLayer * input_node_size
    hidden_layer_size = n_hidden_layers * n_neuronsPerLayer * hidden_node_size
    output_layer_size = output_size * output_node_size

    total_genome_length = input_layer_size + hidden_layer_size + output_layer_size

    assert len(parent1_genome) == total_genome_length + genome_start_index, "Genome length does not match the expected length"

    # Initialize the child genomes with the initial parameters
    child1_genome = np.empty_like(parent1_genome)
    child2_genome = np.empty_like(parent2_genome)

    # Perform uniform crossover for the initial parameters
    for i in range(genome_start_index):
        if np.random.rand() > 0.5:
            child1_genome[i] = parent1_genome[i]
            child2_genome[i] = parent2_genome[i]
        else:
            child1_genome[i] = parent2_genome[i]
            child2_genome[i] = parent1_genome[i]

    current_index = genome_start_index

    # Uniform crossover for input layer nodes
    for _ in range(n_neuronsPerLayer):
        if np.random.rand() > 0.5:
            child1_genome[current_index:current_index + input_node_size] = parent1_genome[current_index:current_index + input_node_size]
            child2_genome[current_index:current_index + input_node_size] = parent2_genome[current_index:current_index + input_node_size]
        else:
            child1_genome[current_index:current_index + input_node_size] = parent2_genome[current_index:current_index + input_node_size]
            child2_genome[current_index:current_index + input_node_size] = parent1_genome[current_index:current_index + input_node_size]
        current_index += input_node_size

    # Uniform crossover for hidden layer nodes
    for _ in range(n_hidden_layers):
        for _ in range(n_neuronsPerLayer):
            if np.random.rand() > 0.5:
                child1_genome[current_index:current_index + hidden_node_size] = parent1_genome[current_index:current_index + hidden_node_size]
                child2_genome[current_index:current_index + hidden_node_size] = parent2_genome[current_index:current_index + hidden_node_size]
            else:
                child1_genome[current_index:current_index + hidden_node_size] = parent2_genome[current_index:current_index + hidden_node_size]
                child2_genome[current_index:current_index + hidden_node_size] = parent1_genome[current_index:current_index + hidden_node_size]
            current_index += hidden_node_size

    # Uniform crossover for output layer nodes
    for _ in range(output_size):
        if np.random.rand() > 0.5:
            child1_genome[current_index:current_index + output_node_size] = parent1_genome[current_index:current_index + output_node_size]
            child2_genome[current_index:current_index + output_node_size] = parent2_genome[current_index:current_index + output_node_size]
        else:
            child1_genome[current_index:current_index + output_node_size] = parent2_genome[current_index:current_index + output_node_size]
            child2_genome[current_index:current_index + output_node_size] = parent1_genome[current_index:current_index + output_node_size]
        current_index += output_node_size

    return child1_genome, child2_genome

def node_crossover(parent1_genome, parent2_genome, n_hidden_layers, n_neuronsPerLayer):
    print(f"len(parent1_genome): {len(parent1_genome)} | len(parent2_genome): {len(parent2_genome)}")
    assert len(parent1_genome) == len(parent2_genome), "Genomes must be of the same length"
    
    input_size = 4  # fixed input size as per the neural network definition
    output_size = 2  # fixed output size as per the neural network definition

    # Compute the size of each node's parameters
    input_node_size = 1 + input_size
    hidden_node_size = 1 + n_neuronsPerLayer
    output_node_size = 1 + n_neuronsPerLayer

    # Calculate the total number of parameters in the genome excluding C3, gamma, mu
    genome_start_index = 3  # C3, gamma, mu

    input_layer_size = n_neuronsPerLayer * input_node_size
    hidden_layer_size = n_hidden_layers * n_neuronsPerLayer * hidden_node_size
    output_layer_size = output_size * output_node_size

    total_genome_length = input_layer_size + hidden_layer_size + output_layer_size

    assert len(parent1_genome) == total_genome_length + genome_start_index, "Genome length does not match the expected length"

    # Calculate the indices where each node starts
    node_indices = [0, 1, 2]  # Start indices for C3, gamma, mu
    current_index = genome_start_index

    # Input layer nodes
    for _ in range(n_neuronsPerLayer):
        node_indices.append(current_index)
        current_index += input_node_size

    # Hidden layer nodes
    for _ in range(n_hidden_layers):
        for _ in range(n_neuronsPerLayer):
            node_indices.append(current_index)
            current_index += hidden_node_size

    # Output layer nodes
    for _ in range(output_size):
        node_indices.append(current_index)
        current_index += output_node_size

    # Choose random start and end nodes for the crossover
    start_node = np.random.randint(0, len(node_indices))
    end_node = np.random.randint(start_node, len(node_indices))

    # Determine the start and end points for the crossover
    start_point = node_indices[start_node]
    if end_node < len(node_indices) - 1:
        end_point = node_indices[end_node + 1]
    else:
        end_point = len(parent1_genome)

    print(f"Performing node crossover from node {start_node} [index: {start_point}] to node {end_node} [index: {end_point}]")

    # Perform crossover
    child1_genome = np.concatenate((parent1_genome[:start_point], parent2_genome[start_point:end_point], parent1_genome[end_point:]))
    child2_genome = np.concatenate((parent2_genome[:start_point], parent1_genome[start_point:end_point], parent2_genome[end_point:]))

    return child1_genome, child2_genome

def arithmetic_crossover(parent1_genome, parent2_genome, n_hidden_layers, n_neuronsPerLayer):
    assert len(parent1_genome) == len(parent2_genome), "Genomes must be of the same length"

    input_size = 4  # fixed input size as per the neural network definition
    output_size = 2  # fixed output size as per the neural network definition

    # Compute the size of each node's parameters (1 bias + input_size weights for input layer, 
    # 1 bias + n_neuronsPerLayer weights for hidden layers, 1 bias + n_neuronsPerLayer weights for output layer)
    input_node_size = 1 + input_size
    hidden_node_size = 1 + n_neuronsPerLayer
    output_node_size = 1 + n_neuronsPerLayer

    # Calculate the total number of parameters in the genome excluding C3, gamma, mu
    genome_start_index = 3  # C3, gamma, mu

    input_layer_size = n_neuronsPerLayer * input_node_size
    hidden_layer_size = n_hidden_layers * n_neuronsPerLayer * hidden_node_size
    output_layer_size = output_node_size * output_size

    total_genome_length = genome_start_index + input_layer_size + hidden_layer_size + output_layer_size

    print(f"genome_start_index: {genome_start_index}, input_layer_size: {input_layer_size}, hidden_layer_size: {hidden_layer_size}, output_layer_size: {output_layer_size}, total_genome_length: {total_genome_length}, len(parent1_genome): {len(parent1_genome)}")
    assert len(parent1_genome) == total_genome_length, "Genome length does not match the expected length"

    # Calculate the indices where each node starts
    node_indices = [0, 1, 2]  # Start indices for C3, gamma, mu
    current_index = genome_start_index

    # Input layer nodes
    for _ in range(n_neuronsPerLayer):
        node_indices.append(current_index)
        current_index += input_node_size

    # Hidden layer nodes
    for _ in range(n_hidden_layers):
        for _ in range(n_neuronsPerLayer):
            node_indices.append(current_index)
            current_index += hidden_node_size

    # Output layer nodes
    for _ in range(output_size):
        node_indices.append(current_index)
        current_index += output_node_size

    # Perform arithmetic crossover for each segment
    a = np.random.rand()  # Random weighting factor
    print(f"Performing arithmetic crossover with a = {a}")
    child1_genome = np.empty_like(parent1_genome)
    child2_genome = np.empty_like(parent2_genome)

    for i in range(len(node_indices) - 1):
        start_point = node_indices[i]
        end_point = node_indices[i + 1]

        child1_genome[start_point:end_point] = a * parent1_genome[start_point:end_point] + (1 - a) * parent2_genome[start_point:end_point]
        child2_genome[start_point:end_point] = (1 - a) * parent1_genome[start_point:end_point] + a * parent2_genome[start_point:end_point]

    # Handle the last segment
    start_point = node_indices[-1]
    child1_genome[start_point:] = a * parent1_genome[start_point:] + (1 - a) * parent2_genome[start_point:]
    child2_genome[start_point:] = (1 - a) * parent1_genome[start_point:] + a * parent2_genome[start_point:]

    return child1_genome, child2_genome


def mutate_genome(genome, mutation_rate, mutation_std_dev):
    """
    Apply Gaussian mutation to the genome.

    Args:
        genome (np.array): The genome to be mutated.
        mutation_rate (float): Probability of each gene being mutated.
        mutation_std_dev (float): Standard deviation for the percentage Gaussian mutation.

    Returns:
        np.array: The mutated genome.
    """
    mutated_genome = genome.copy()
    for i in range(len(mutated_genome)):
        if np.random.rand() < mutation_rate:
            if i in [0, 1, 2]: # initial conditions need smaller mutation
                mutation_std_dev = mutation_std_dev / 10
            mutation = np.random.normal(0, mutation_std_dev)
            mutated_genome[i] += mutated_genome[i] * mutation
    return mutated_genome

def tournament_selection(chromosomes, generation_fitness_scores, n_hidden_layers, n_neuronsPerLayer, mutation_rate, mutation_std_dev):

    # Modify numpy print options
    np.set_printoptions(edgeitems=np.inf, linewidth=np.inf, formatter={'float': '{: .3e}'.format})

    """
    print("Current chromosomes and their fitness scores:")
    for i in range(len(chromosomes)):
        print(f"Chromosome {i+1} -  Fitness: {generation_fitness_scores[i]} \n{chromosomes[i]}")
    """

    next_generation = []
    population_size = len(chromosomes)

    # Check if all_fitness_scores and chromosomes have the same length
    if len(generation_fitness_scores) != population_size:
        raise ValueError("The length of all_fitness_scores must match the length of chromosomes")

    # Replace NaN and inf values with -inf for selection purposes
    fitness_scores_clean = np.array(generation_fitness_scores, dtype=float)
    fitness_scores_clean[~np.isfinite(fitness_scores_clean)] = -np.inf
    
    # Identify the fittest individual
    fittest_indices = np.where(fitness_scores_clean == np.max(fitness_scores_clean))[0]
    if len(fittest_indices) == 0:
        # Fallback: if all are NaN/inf, pick first one
        fittest_index = 0
    else:
        fittest_index = fittest_indices[0]
    fittest_individual = chromosomes[fittest_index]

    while len(next_generation) < population_size:
        
        
        # Ensure the fittest individual is always in the first round
        if len(next_generation) == 0:
            individual1 = fittest_individual
            idx1 = fittest_index
            print(f"Passing fittest individual {idx1+1} with fitness {generation_fitness_scores[fittest_index]} to the next generation")
        else:
            idx1 = np.random.choice(range(population_size))
            individual1 = chromosomes[idx1]
        
        
        # Select the second individual randomly
        idx2 = np.random.choice(range(population_size), size=1)[0]
        individual2 = chromosomes[idx2]
        fitness1, fitness2 = generation_fitness_scores[idx1], generation_fitness_scores[idx2]

        
        if fitness1 > fitness2:
            fitter, unfitter = individual1, individual2
            print(f"Passing individual {idx1+1} to the next generation")
        else:
            fitter, unfitter = individual2, individual1
            print(f"Passing individual {idx2+1} to the next generation")


        # Add the fitter individual to the next generation
        next_generation.append(fitter)

        # Choose a crossover method randomly
        crossover_method = random.choices(
            [one_point_crossover, uniform_crossover, node_crossover, arithmetic_crossover], 
            k=1
        )[0]
        

         # Print the chosen crossover method and the parent chromosomes
        print(f"Performing {crossover_method.__name__} on chromosomes {idx1+1} and {idx2+1}")


        # Perform the crossover to create a new individual
        child1, _ = crossover_method(fitter, unfitter, n_hidden_layers, n_neuronsPerLayer)

        # Apply mutation to the child genome
        child1 = mutate_genome(child1, mutation_rate, mutation_std_dev)

        next_generation.append(child1)

        

    
    # Ensure the next generation has exactly the population size
    if len(next_generation) > population_size:
        raise error("The next generation has more individuals than the population size")


    return next_generation



