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

def custom_read_data_results(file_path):
    data = []
    actual_iter = ""
    actual_nb_spurious = 0
    actual_nb_found = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()
        recording = False
        nb_line= 0
        for line in lines:
            line = line.strip()
            if line.find("="):
                recording = True
            if recording:
                parts = line.split('=')
                if len(parts) >= 2:
                    if parts[0] == "iter" and nb_line !=0:
                        data.append([actual_iter,actual_nb_found,actual_nb_spurious])
                        actual_iter = parts[1]
                    if parts[0] == "nb_found_patterns":
                        actual_nb_found = parts[1]
                    if parts[0] == "nb_spurious_patterns":
                        actual_nb_spurious = parts[1]
            nb_line+=1
    return pd.DataFrame(data, columns=['iter', 'nb_found_patterns', 'nb_spurious_patterns'])

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
        data[file_name.split('.')[0]] = custom_read_data_results(file_path)
    else:
        print("error no file found")
       
    return data

def collect_data(base_folder):
    data_collection = []
    for i in range(1000):
        folder_name = f'sim_nb_{i}'
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            data = read_data(folder_path)
            data_collection.append(data)
    return data_collection


#%% Function to structure the data for analysis
def structure_data_max_retrieve_pattern(all_data):
    structured_data = defaultdict(list)

    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            #single values
            network_size = int(parameters_df.loc[parameters_df['parameter'] == 'network_size', 'value'].values[0])
            # beta = float(parameters_df.loc[parameters_df['Parameter'] == 'beta', 'Value'].values[0])
            #Many values
            #nb_iter = results_df.loc[results_df['Parameter'] == 'iter', 'Value'].values
            nb_found_pattern = results_df["nb_found_patterns"].tolist()
            for index in range(len(nb_found_pattern)):
                nb_found_pattern[index] = int(nb_found_pattern[index])
            structured_data[network_size].append(max(nb_found_pattern) )          

            # max_nb_found_pattern = max(nb_found_pattern)

            # structured_data['network_size'].append(network_size)
            # structured_data['beta'].append(beta)
            # structured_data['max_nb_found_pattern'].append(max_nb_found_pattern)
    structured_list = [[],[]]
    for key,value in structured_data.items() :
        structured_list[0].append(key)
        structured_list[1].append(max(value)) # Maximum stored pattern

    # structured_df = pd.DataFrame(structured_data)
    return structured_list

#%% Function to structure the data for analysis
def structure_data_max_retrieve_pattern_no_spurious(all_data):
    structured_data = defaultdict(list)

    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            #single values
            network_size = int(parameters_df.loc[parameters_df['parameter'] == 'network_size', 'value'].values[0])
            # beta = float(parameters_df.loc[parameters_df['Parameter'] == 'beta', 'Value'].values[0])
            #Many values
            nb_found_pattern_no_spurious = results_df.loc[results_df['nb_spurious_patterns'] == '0', 'nb_found_patterns'].values
            print(nb_found_pattern_no_spurious)
            for index in range(len(nb_found_pattern_no_spurious)):
                nb_found_pattern_no_spurious[index] = int(nb_found_pattern_no_spurious[index])
            nb_found_pattern_no_spurious= np.append(nb_found_pattern_no_spurious,[0])
            structured_data[network_size].append(max(nb_found_pattern_no_spurious) )          

    structured_list = [[],[]]
    print(structured_data)
    for key,value in structured_data.items() :
        structured_list[0].append(key)
        structured_list[1].append(max(value)) # Maximum stored pattern

    # structured_df = pd.DataFrame(structured_data)
    return structured_list
#%% Collect data from all subfolders
base_folder = "..\\..\\..\\data\\all_data_splited\\sleep_simulations\\sleep_parameter_test"
all_data = collect_data(base_folder)
#%% Structure the collected data
structured_data = structure_data_max_retrieve_pattern(all_data)
structured_data_no_spurious = structure_data_max_retrieve_pattern_no_spurious(all_data)
 
#%% Plotting examples
import matplotlib.pyplot as plt
plt.plot(structured_data[0],structured_data[1])
plt.xlabel("network size")
plt.ylabel("number of retrieve patterns")
plt.show()
# %%
plt.plot(structured_data_no_spurious[0],structured_data_no_spurious[1])
plt.xlabel("network size")
plt.ylabel("number of retrieve patterns")
plt.show()