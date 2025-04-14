# src/utils/config_loader.py
import yaml
import os
import sys

DEFAULT_CONFIG_PATH = "../config/config.yaml"

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If there's an error parsing the YAML.
        Exception: For other potential loading errors.
    """
    absolute_path = os.path.abspath(config_path)
    if not os.path.exists(absolute_path):
        print(f"Error: Configuration file not found at expected absolute path: {absolute_path}", file=sys.stderr)
        # Try relative path as fallback, though absolute is preferred for clarity
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Configuration file not found at: {config_path} (checked relative and absolute paths)")
        else:
            absolute_path = config_path # Use relative path if absolute failed but relative exists

    try:
        with open(absolute_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty config file case
            print(f"Warning: Configuration file '{absolute_path}' is empty.")
            return {}
        print(f"Configuration loaded successfully from: {absolute_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{absolute_path}': {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error loading configuration from '{absolute_path}': {e}", file=sys.stderr)
        raise