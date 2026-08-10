"""
Reproducibility utilities for SOMBRERO.
Provides deterministic seeding and output tracking.
"""
import random
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for all relevant libraries to ensure reproducibility.
    
    Args:
        seed: Integer seed value for random number generators.
    """
    import numpy as np
    import torch
    
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # PyTorch determinism settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_run_id() -> str:
    """
    Generate a unique run ID based on current timestamp.
    
    Returns:
        String run ID in format 'YYYY-MM-DD-HHhMMmSSs'.
    """
    return datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")


def create_summary(
    config: Any,
    chromosome: Optional[list] = None,
    fitness_score: Optional[float] = None,
    scalars: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a summary dictionary for output.
    
    Args:
        config: Configuration object (will be converted to dict).
        chromosome: Optional chromosome list.
        fitness_score: Optional fitness score.
        scalars: Optional dict of scalar outputs (payload, t_200AU, etc.).
        seed: Seed used for this run.
        
    Returns:
        Dictionary suitable for JSON serialization.
    """
    from dataclasses import asdict
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
    }
    
    # Add config (handle both dataclass and dict)
    try:
        summary['config'] = asdict(config)
    except TypeError:
        summary['config'] = dict(config) if hasattr(config, '__iter__') else str(config)
    
    if chromosome is not None:
        summary['chromosome'] = [float(x) for x in chromosome]
    
    if fitness_score is not None:
        summary['fitness_score'] = float(fitness_score)
    
    if scalars is not None:
        # Convert numpy types to Python types for JSON serialization
        clean_scalars = {}
        for k, v in scalars.items():
            try:
                import numpy as np
                if isinstance(v, (np.integer, np.floating)):
                    clean_scalars[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean_scalars[k] = v.tolist()
                else:
                    clean_scalars[k] = v
            except (ImportError, TypeError):
                clean_scalars[k] = v
        summary['scalars'] = clean_scalars
    
    return summary


def save_summary(summary: Dict[str, Any], output_dir: Path, filename: str = "summary.json") -> Path:
    """
    Save a summary dictionary to JSON.
    
    Args:
        summary: Summary dictionary to save.
        output_dir: Directory to save to.
        filename: Name of the output file.
        
    Returns:
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return output_path
