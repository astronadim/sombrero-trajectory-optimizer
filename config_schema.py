"""
Configuration schema for SOMBRERO trajectory optimization.
Defines dataclasses for all configuration settings.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class SimulationConfig:
    """Simulation parameters for trajectory integration."""
    total_simulation_time_thrust_phase: float = 4.5 * 365 * 24 * 60 * 60  # 4.5 years in seconds
    rtol: float = 1e-5
    atol: float = 1e-5
    enable_jga: bool = True


@dataclass
class NeuroConfig:
    """Neurocontroller architecture settings."""
    activation_name: str = "tanh"
    n_hidden_layers: int = 0
    n_neurons_per_layer: int = 20


@dataclass
class EvolutionConfig:
    """Evolutionary algorithm hyperparameters."""
    population_size: int = 50
    generations: int = 5000
    mutation_rate: float = 0.1
    mutation_std_dev: float = 0.1
    tournament_k: int = 2
    seed: int = 42
    n_workers: Optional[int] = None  # Number of parallel workers (None = auto-detect)


@dataclass
class InitConfig:
    """Initialization mode for population."""
    init_mode: str = "cold"  # "cold" or "warm"
    warm_start_chromosome_path: Optional[str] = None
    warm_start_mix_fraction: float = 0.5  # fraction of population to warm-start (0..1)


@dataclass
class ElectricPropulsionConfig:
    """Electric propulsion system parameters."""
    Isp: float = 6000.0  # Specific impulse in seconds
    efficiency: float = 0.75  # Thruster efficiency (0-1)


@dataclass
class InitialConditionConfig:
    """Fixed initial conditions for spacecraft (individually toggleable)."""
    fix_C3: bool = False
    fixed_C3: float = 0.0  # km^2/s^2
    fix_launch_angle: bool = False
    fixed_launch_angle_rad: float = 0.0  # radians
    fix_mu: bool = False
    fixed_mu: float = 0.59  # mass fraction


@dataclass
class LaunchIntervalConfig:
    """Intervals for random initialization of launch parameters."""
    C3_interval: List[float] = field(default_factory=lambda: [10.0, 40.0])
    launch_angle_interval_deg: List[float] = field(default_factory=lambda: [0.0, -10.0])
    mu_propellant_plus_payload_interval: List[float] = field(default_factory=lambda: [0.5, 0.6])
    
    def get_C3_interval(self) -> np.ndarray:
        return np.array(self.C3_interval)
    
    def get_launch_angle_interval_rad(self) -> np.ndarray:
        return np.array(self.launch_angle_interval_deg) * np.pi / 180
    
    def get_mu_interval(self) -> np.ndarray:
        return np.array(self.mu_propellant_plus_payload_interval)


@dataclass
class FitnessSettings:
    """Fitness function parameters."""
    R_SolarOBerth_Design: float = 0.3
    R_SolarOberth_LowerBound: float = 0.25
    R_SolarOberth_UpperBound: float = 0.7
    t_mission_target: float = 25 * 365 * 24 * 60 * 60  # 25 years in seconds
    m_payload_target: float = 1000.0  # kg
    
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            'R_SolarOBerth_Design': self.R_SolarOBerth_Design,
            'R_SolarOberth_LowerBound': self.R_SolarOberth_LowerBound,
            'R_SolarOberth_UpperBound': self.R_SolarOberth_UpperBound,
            't_mission_target': self.t_mission_target,
            'm_payload_target': self.m_payload_target
        }


@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "outputs"
    save_plots: bool = True
    save_excel: bool = True
    run_id: Optional[str] = None  # If None, will be auto-generated from timestamp


@dataclass
class Config:
    """Top-level configuration container."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    neuro: NeuroConfig = field(default_factory=NeuroConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    init: InitConfig = field(default_factory=InitConfig)
    electric_propulsion: ElectricPropulsionConfig = field(default_factory=ElectricPropulsionConfig)
    initial_conditions: InitialConditionConfig = field(default_factory=InitialConditionConfig)
    launch_intervals: LaunchIntervalConfig = field(default_factory=LaunchIntervalConfig)
    fitness: FitnessSettings = field(default_factory=FitnessSettings)
    output: OutputConfig = field(default_factory=OutputConfig)


def get_activation_function(activation_name: str):
    """Get PyTorch activation function class from name."""
    import torch.nn as nn
    
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
    }
    
    name_lower = activation_name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation function: {activation_name}. "
                         f"Available: {list(activations.keys())}")
    return activations[name_lower]
