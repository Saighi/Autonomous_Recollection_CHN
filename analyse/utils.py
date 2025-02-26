import os 
import numpy as np
from collections import defaultdict

def load_simulation_trajectories(myDir, name_file):
    data = []
    max_index = -1
    
    # First pass to determine the size of data list
    for root, dirs, files in os.walk(myDir):
        for dir in dirs:
            if dir.startswith('sim_nb_'):
                sim_dir = os.path.join(root, dir)
                for file in os.listdir(sim_dir):
                    if file.startswith(name_file):
                        # Extract number from filename (assuming format: name_file + number + ".data")
                        number = int(file[len(name_file):-5])  # Remove prefix and ".data" suffix
                        max_index = max(max_index, number)
    
    # Initialize data list with None values
    data = [None] * (max_index + 1)
    
    # Second pass to fill the data
    for root, dirs, files in os.walk(myDir):
        for dir in dirs:
            if dir.startswith('sim_nb_'):
                sim_dir = os.path.join(root, dir)
                for file in os.listdir(sim_dir):
                    if file.startswith(name_file):
                        # Extract number from filename
                        number = int(file[len(name_file):-5])
                        file_path = os.path.join(sim_dir, file)
                        # Load matrix and store at correct index
                        data[number] = np.loadtxt(file_path, ndmin=2)
    
    return data

def parse_config_file(file_name):
    config = {}
    
    try:
        with open(file_name, 'r') as file:
            for line in file:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Check if the line contains '='
                if '=' not in line:
                    continue
                
                # Split each line into key and value
                key, value = line.split('=', 1)  # Split only on the first '='
                
                # Strip whitespace from key and value
                key = key.strip()
                value = value.strip()
                
                # Try to convert the value to a float or int if possible
                try:
                    # First, try converting to int
                    value = int(value)
                except ValueError:
                    try:
                        # If int fails, try converting to float
                        value = float(value)
                    except ValueError:
                        # If both fail, keep it as a string
                        pass
                
                # Add the key-value pair to the dictionary
                config[key] = value
        
        return config
    
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return None
    except IOError:
        print(f"Error: Unable to read file '{file_name}'.")
        return None