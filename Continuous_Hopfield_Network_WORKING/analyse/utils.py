import os 
import numpy as np
from collections import defaultdict

def load_simulation_trajectories(myDir, name_file):
    data = []
    
    for root, dirs, files in os.walk(myDir):
        for dir in dirs:
            if dir.startswith('sim_nb_'):
                data.append([])
                x = int(dir.split('_')[-1])  # Extract the number x from sim_nb_x
                
                sim_dir = os.path.join(root, dir)
                cpt = 0
                for file in os.listdir(sim_dir):
                    if file.startswith(name_file):
                        cpt+=1
                        file_path = os.path.join(sim_dir, file)
                        print(file_path)
                        matrix = np.loadtxt(file_path) 
                        data[-1].append(matrix)
    
    return np.array(data)  # Convert defaultdict to regular dict before returning

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