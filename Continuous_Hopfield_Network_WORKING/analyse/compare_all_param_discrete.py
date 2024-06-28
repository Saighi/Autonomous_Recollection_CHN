#%%
import os
import pandas as pd
from collections import defaultdict
import numpy as np

def custom_read_data_parameters(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        recording = False
        for line in lines:
            line = line.strip()
            if line.find("="):
                recording = True
            if recording:
                parts = line.split('=')
                if len(parts) >= 2:
                    parameter = parts[0]
                    value = parts[1]
                    data.append([parameter, value])
    return pd.DataFrame(data, columns=['parameter', 'value'])


def read_data(folder_path):
    data = {}
    file_name = 'parameters.data'
    file_path= os.path.join(folder_path,file_name)
    if os.path.exists(file_path):
        data[file_name.split('.')[0]] = custom_read_data_parameters(file_path)
    else:
        print("error no file found")

    file_name = 'results.data'
    file_path= os.path.join(folder_path,file_name)
    if os.path.exists(file_path):
        data[file_name.split('.')[0]] = custom_read_data_parameters(file_path)
    else:
        print("error no file found")
       
    return data

def collect_data(base_folder,nb_sim):
    data_collection = []
    for i in range(nb_sim):
        folder_name = f'sim_nb_{i}'
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            data = read_data(folder_path)
            data_collection.append(data)
    return data_collection


#%% Function to structure the data for analysis
def network_size_max_retrieve_pattern(all_data):
    structured_data = defaultdict(list)

    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            network_size = int(parameters_df.loc[parameters_df['parameter'] == 'network_size', 'value'].values[0])
            nb_found_pattern = int(results_df.loc[results_df['parameter'] == 'nb_found_patterns', 'value'].values[0])
            structured_data[network_size].append(nb_found_pattern)          

    structured_list = [[],[]]
    for key,value in structured_data.items() :
        structured_list[0].append(key)
        structured_list[1].append(max(value)) # Maximum stored pattern

    return structured_list

#%% Collect data from all subfolders
base_folder = "..\\..\\..\\data\\all_data_discret\\network_size_num_pattern"
all_data = collect_data(base_folder,200)
#%% Structure the collected data
structured_data = network_size_max_retrieve_pattern(all_data)
#%% Plotting examples
import matplotlib.pyplot as plt
plt.plot(structured_data[0],structured_data[1])
plt.ylabel("number of retrieved patterns")
plt.xlabel("network size")
plt.show()

# %%
