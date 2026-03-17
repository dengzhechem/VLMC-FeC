import os
import json

def read_params(file_path='params.json'):
    required_keys = {
        'energy_C', 'miu_C', 'model', 'init_stru', 'max_iterations',
        'temperature', 'prob_mig_C', 'prob_add_C', 'prob_rmv_C', 'prob_rtt'
    }
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file {file_path} not found")
    
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file {file_path}: {e}") from e
    
    if not isinstance(params, dict):
        raise TypeError(f"Input file {file_path} must contain a JSON object {{...}}.")
    
    missing = required_keys - params.keys()
    if missing:
        raise KeyError(f"Missing parameter(s): {', '.join(sorted(missing))}")

    return params