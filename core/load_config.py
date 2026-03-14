import yaml
import os

def load_config(config_path="config/config.yaml"):
    """
    Loads configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # Test loading
    try:
        cfg = load_config()
        print("Configuration successfully loaded:")
        print(cfg)
    except Exception as e:
        print(f"Error: {e}")
