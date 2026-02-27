"""
Configuration I/O utilities for SOMBRERO.
Handles loading/saving JSON configs and chromosomes.
"""
import json
from pathlib import Path
from typing import Union, List, Any, Optional
from dataclasses import asdict

from config_schema import (
    Config, SimulationConfig, NeuroConfig, EvolutionConfig,
    InitConfig, ElectricPropulsionConfig, InitialConditionConfig, LaunchIntervalConfig,
    FitnessSettings, OutputConfig
)


def load_config(path: Union[str, Path]) -> Config:
    """
    Load configuration from a JSON file.
    
    Args:
        path: Path to the JSON configuration file.
        
    Returns:
        Config object with all settings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Build config from nested dictionaries
    config = Config(
        simulation=SimulationConfig(**data.get('simulation', {})),
        neuro=NeuroConfig(**data.get('neuro', {})),
        evolution=EvolutionConfig(**data.get('evolution', {})),
        init=InitConfig(**data.get('init', {})),
        electric_propulsion=ElectricPropulsionConfig(**data.get('electric_propulsion', {})),
        initial_conditions=InitialConditionConfig(**data.get('initial_conditions', {})),
        launch_intervals=LaunchIntervalConfig(**data.get('launch_intervals', {})),
        fitness=FitnessSettings(**data.get('fitness', {})),
        output=OutputConfig(**data.get('output', {}))
    )
    
    return config


def save_config(config: Config, path: Union[str, Path]) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Config object to save.
        path: Output path for the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = asdict(config)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_json(path: Union[str, Path], data: Any) -> None:
    """
    Save arbitrary data to a JSON file.
    
    Args:
        path: Output path for the JSON file.
        data: Data to serialize (must be JSON-serializable).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load JSON data from a file.
    
    Args:
        path: Path to the JSON file.
        
    Returns:
        Parsed JSON data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def load_chromosome(path: Union[str, Path]) -> List[float]:
    """
    Load a chromosome from a JSON file.
    
    Args:
        path: Path to the chromosome JSON file.
        
    Returns:
        List of floats representing the chromosome.
    """
    data = load_json(path)
    
    # Handle both direct list format and dict format
    if isinstance(data, list):
        return [float(x) for x in data]
    elif isinstance(data, dict) and 'chromosome' in data:
        return [float(x) for x in data['chromosome']]
    else:
        raise ValueError(f"Invalid chromosome format in {path}. "
                         "Expected a list or dict with 'chromosome' key.")


def save_chromosome(chromosome: List[float], path: Union[str, Path], 
                    metadata: Optional[dict] = None) -> None:
    """
    Save a chromosome to a JSON file.
    
    Args:
        chromosome: List of floats representing the chromosome.
        path: Output path for the JSON file.
        metadata: Optional metadata to include (fitness, generation, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if metadata:
        data = {
            'chromosome': [float(x) for x in chromosome],
            **metadata
        }
    else:
        data = {'chromosome': [float(x) for x in chromosome]}
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def validate_chromosome_length(chromosome: List[float], n_hidden_layers: int, 
                                n_neurons_per_layer: int) -> bool:
    """
    Validate that chromosome length matches expected neural network architecture.
    
    Args:
        chromosome: The chromosome to validate.
        n_hidden_layers: Number of hidden layers.
        n_neurons_per_layer: Neurons per layer.
        
    Returns:
        True if valid, raises ValueError otherwise.
    """
    input_size = 4
    output_size = 2
    
    # 3 initial params (C3, gamma, mu)
    initial_params = 3
    
    # Input layer: n_neurons_per_layer * (1 bias + input_size weights)
    input_layer_size = n_neurons_per_layer * (1 + input_size)
    
    # Hidden layers: n_hidden_layers * n_neurons_per_layer * (1 bias + n_neurons_per_layer weights)
    hidden_layer_size = n_hidden_layers * n_neurons_per_layer * (1 + n_neurons_per_layer)
    
    # Output layer: output_size * (1 bias + n_neurons_per_layer weights)
    output_layer_size = output_size * (1 + n_neurons_per_layer)
    
    expected_length = initial_params + input_layer_size + hidden_layer_size + output_layer_size
    
    if len(chromosome) != expected_length:
        raise ValueError(
            f"Chromosome length {len(chromosome)} does not match expected {expected_length} "
            f"for architecture: n_hidden_layers={n_hidden_layers}, "
            f"n_neurons_per_layer={n_neurons_per_layer}"
        )
    
    return True
