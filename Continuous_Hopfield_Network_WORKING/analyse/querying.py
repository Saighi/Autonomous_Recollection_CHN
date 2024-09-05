#%%
import os
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def get_param_list(data,name_param):
    param_list = []
    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            param_list.append(parameters_df.loc[parameters_df['parameter'] == name_param, 'value'].values[0])
    return param_list

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
    return pd.DataFrame(data, columns=['variable', 'value'])

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

#%%
def ratio_found_pattern(all_data,net_size_target, num_pat_targes):
    structured_data = [] 
    nb_found = 0
    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            net_size = float(parameters_df.loc[parameters_df['parameter'] == 'network_size', 'value'].values[0])
            relative_nb_pat = float(parameters_df.loc[parameters_df['parameter'] == 'relative_num_patterns', 'value'].values[0])

            if net_size_target==net_size and num_pat_targes == relative_nb_pat:
                ratio_flips = float(parameters_df.loc[parameters_df['parameter'] == 'ratio_flips_querying', 'value'].values[0])
                structured_data.append([ratio_flips,float(results_df.loc[results_df['variable']=='ratio_found_patterns', 'value'].values[0])])          

    return structured_data

def ratio_found_pattern_drive(all_data,target_drive_target, num_pat_targes):
    structured_data = [] 
    nb_found = 0
    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            target_drive = float(parameters_df.loc[parameters_df['parameter'] == 'drive_target', 'value'].values[0])
            relative_nb_pat = float(parameters_df.loc[parameters_df['parameter'] == 'relative_num_patterns', 'value'].values[0])

            if target_drive==target_drive_target and num_pat_targes == relative_nb_pat:
                ratio_flips = float(parameters_df.loc[parameters_df['parameter'] == 'ratio_flips_querying', 'value'].values[0])
                structured_data.append([ratio_flips,float(results_df.loc[results_df['variable']=='ratio_found_patterns', 'value'].values[0])])          

    return structured_data
#%% Collect data from all subfolders
# base_folder = "..\\..\\..\\data\\all_data_splited\\query_simulations\\quere_many_sizes_relative_num_patterns_3"
base_folder = "..\\..\\..\\data\\all_data_splited\\query_simulations\\quere_many_drives_relative_num_patterns_new_writing_new_convergence"
all_data = collect_data(base_folder)
# correlation_values = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
# Number of stored pattern relative to the network size data
relative_num_pat_string = get_param_list(all_data,"relative_num_patterns")
relative_number_patterns = np.sort(list(set(map(float,relative_num_pat_string))))
relative_number_patterns = relative_number_patterns[0::2]
#Network sizes data
net_sizes_string = get_param_list(all_data,"network_size")
net_sizes = np.sort(list(set(map(int,net_sizes_string))))
# net_sizes = net_sizes[0::2]
# Target drive data
target_drives_string = get_param_list(all_data,"drive_target")
target_drives = np.sort(list(set(map(float,target_drives_string))))
target_drives = target_drives
#%%
net_sizes
#%%
# # FIG FOR VARIOUS NET SIZES
# fig = plt.figure(layout='constrained', figsize=(15, 20))
# subfig = fig.subfigures(len(net_sizes), 1, wspace=0.07)
# for idx, size in enumerate(net_sizes):
    
#     ax = subfig[idx].subplots(1,len(relative_number_patterns),sharey=True)

#     for idx_2,nb_pat in enumerate(relative_number_patterns):
#         structured_data = np.array(ratio_found_pattern(all_data, size, nb_pat))
#         legend = []
#         rat_flips = structured_data[:,0]
#         fnd_pat = structured_data[:,1]
#         ax[idx_2].plot(rat_flips, fnd_pat)
#         ax[idx_2].set_xlabel("ratio flips")
#         ax[idx_2].set_ylabel("ratio found patterns")
#         ax[idx_2].set_title("number of pattern = "+ str(nb_pat))
    
#     # Add a shared title for the two plots in the same row
#     subfig[idx].suptitle(f'network size: {size}%', fontsize=20)

# # plt.tight_layout()
# plt.show()
# %%
# FIG FOR VARIOUS TARGET DRIVES
fig = plt.figure(layout='constrained', figsize=(15, 30))
subfig = fig.subfigures(len(target_drives), 1, wspace=0.07)
for idx, drive in enumerate(target_drives):
    
    ax = subfig[idx].subplots(1,len(relative_number_patterns),sharey=True)

    for idx_2,nb_pat in enumerate(relative_number_patterns):
        structured_data = np.array(ratio_found_pattern_drive(all_data, drive, nb_pat))
        legend = []
        rat_flips = structured_data[:,0]
        fnd_pat = structured_data[:,1]
        ax[idx_2].plot(rat_flips, fnd_pat)
        ax[idx_2].set_xlabel("ratio flips")
        ax[idx_2].set_ylabel("ratio found patterns")
        ax[idx_2].set_title("number of pattern = "+ str(nb_pat))
    
    # Add a shared title for the two plots in the same row
    subfig[idx].suptitle(f'target drive: {drive}', fontsize=20)

# plt.tight_layout()
plt.show()
# %%
