import configparser
import os

def load_config(config_file='config.ini'):
    """Loads the configuration file."""
    config = configparser.ConfigParser()
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    config.read(config_file)

    def create_dir_for_file(file_path):
        """Helper function to create a directory for a file, if needed."""
        dir_name = os.path.dirname(file_path)
        # Only create if dir_name is not an empty string
        if dir_name: 
            os.makedirs(dir_name, exist_ok=True)

    # Create necessary directories
    os.makedirs(config.get('Paths', 'dataset_path'), exist_ok=True)
    os.makedirs(config.get('Paths', 'unknown_faces_path'), exist_ok=True)
    os.makedirs(config.get('Paths', 'debug_aligned_faces_path'), exist_ok=True)
    
    # Use the new helper function for file paths
    create_dir_for_file(config.get('Paths', 'log_file'))
    create_dir_for_file(config.get('Paths', 'attendance_file'))
    
    return config