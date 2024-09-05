#%%
import os
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

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
                    if parts[0] == "iter" :
                        if nb_line!=0:
                            data.append([actual_iter,actual_nb_found,actual_nb_spurious])
                        actual_iter = parts[1]
                    if parts[0] == "nb_found_patterns":
                        actual_nb_found = parts[1]
                    if parts[0] == "nb_spurious_patterns":
                        actual_nb_spurious = parts[1]
            nb_line+=1
        data.append([actual_iter,actual_nb_found,actual_nb_spurious])
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
            beta = float(parameters_df.loc[parameters_df['Parameter'] == 'beta', 'Value'].values[0])
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
def structure_data_max_retrieve_pattern(all_data):
    structured_data = defaultdict(list)

    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            #single values
            network_size = int(parameters_df.loc[parameters_df['parameter'] == 'network_size', 'value'].values[0])
            beta = float(parameters_df.loc[parameters_df['Parameter'] == 'beta', 'Value'].values[0])
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
def relative_nb_found_iter_beta_fixe_num_pat(all_data,corr,num_pat):
    structured_data = defaultdict(list)

    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            nb_pat = int(parameters_df.loc[parameters_df['parameter'] == 'num_patterns', 'value'].values[0])
            correl = float(parameters_df.loc[parameters_df['parameter'] == 'noise_level', 'value'].values[0])

            if nb_pat==num_pat and corr == correl:
                beta = float(parameters_df.loc[parameters_df['parameter'] == 'beta', 'value'].values[0])
                nb_iter_sim = int(parameters_df.loc[parameters_df['parameter'] == 'nb_iter_mult', 'value'].values[0])*nb_pat
                nb_found_pat_iter = []
                for i in range(nb_iter_sim):
                    nb_found_pat_iter.append([i,int(results_df.loc[results_df['iter'] == str(i), 'nb_found_patterns'].values[0])/nb_pat])
                structured_data[beta]= np.array(nb_found_pat_iter)          

    return structured_data

def relative_nb_found_iter_repetition_fixe_num_pat(all_data,corr,num_pat):
    structured_data = defaultdict(list)

    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            nb_pat = int(parameters_df.loc[parameters_df['parameter'] == 'num_patterns', 'value'].values[0])
            correl = float(parameters_df.loc[parameters_df['parameter'] == 'noise_level', 'value'].values[0])

            if nb_pat==num_pat and corr == correl:
                beta = float(parameters_df.loc[parameters_df['parameter'] == 'beta', 'value'].values[0])
                repetition = float(parameters_df.loc[parameters_df['parameter'] == 'repetition', 'value'].values[0])
                nb_iter_sim = int(parameters_df.loc[parameters_df['parameter'] == 'nb_iter_mult', 'value'].values[0])*nb_pat
                nb_found_pat_iter = []
                for i in range(nb_iter_sim):
                    nb_found_pat_iter.append([i,int(results_df.loc[results_df['iter'] == str(i), 'nb_found_patterns'].values[0])/nb_pat])
                structured_data[repetition]= np.array(nb_found_pat_iter)          

    return structured_data

def relative_nb_found_iter_num_pat_fixe_beta(all_data,corr,beta_target):
    structured_data = defaultdict(list)
    nb_found = 0
    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            beta = float(parameters_df.loc[parameters_df['parameter'] == 'beta', 'value'].values[0])
            correl = float(parameters_df.loc[parameters_df['parameter'] == 'noise_level', 'value'].values[0])

            if beta==beta_target and corr == correl:
                nb_found+=1
                nb_found_pat_iter = []
                nb_pat = int(parameters_df.loc[parameters_df['parameter'] == 'num_patterns', 'value'].values[0])
                nb_iter_sim = int(parameters_df.loc[parameters_df['parameter'] == 'nb_iter_mult', 'value'].values[0])*nb_pat
                for i in range(nb_iter_sim):
                    nb_found_pat_iter.append([i,int(results_df.loc[results_df['iter'] == str(i), 'nb_found_patterns'].values[0])/nb_pat])
                structured_data[nb_pat]= np.array(nb_found_pat_iter)          

    return structured_data
#%% Collect data from all subfolders
base_folder = "..\\..\\..\\data\\all_data_splited\\sleep_simulations\\sleep_parameter_fixed_beta_auto_nb_iter"
all_data = collect_data(base_folder)
network_size = 50
correlation_values = [0.1,0.3,0.6,1]
# betas = [0.001,0.023,0.067,0.1]
betas = [0.05]
network_sizes = []
# number_patterns = [5,7,10,13,16,18,21,24,27,30]
number_patterns = [5,10,16,21,27,30]

# Create subplots for each correlation value
# fig, axs = plt.subplots(len(correlation_values), 2, figsize=(15, 25))

# for idx, correlation_pat in enumerate(correlation_values):
#     structured_data = relative_nb_found_iter_beta_fixe_num_pat(all_data, correlation_pat, 13)
#     structured_data_nb_pat = relative_nb_found_iter_num_pat_fixe_beta(all_data, correlation_pat, 0.045)
    
#     # Plotting for beta fixed
#     legend = []
#     it = 0
#     for key, value in structured_data.items():
#         if it % 2 == 0:
#             axs[idx, 0].plot(value[:,0], value[:,1])
#             axs[idx, 0].set_xlabel("number of iter")
#             axs[idx, 0].set_ylabel("percentage of retrieve patterns")
#             legend.append(str(key))
#         it += 1
#     axs[idx, 0].legend(legend, title="beta value:")
#     # axs[idx, 0].set_title(f'percentage flips: {correlation_pat}')
    
#     # Plotting for number of patterns fixed
#     legend = []
#     it = 0
#     for key, value in structured_data_nb_pat.items():
#         if it % 2 == 0:
#             axs[idx, 1].plot(value[:,0], value[:,1])
#             axs[idx, 1].set_xlabel("number of iter")
#             axs[idx, 1].set_ylabel("percentage of retrieve patterns")
#             legend.append(str(key))
#         it += 1
#     axs[idx, 1].legend(legend, title="nb pattern:")
#     # axs[idx, 1].set_title(f'Correlation: {correlation_pat}')

#     fig.suptitle(f'random flip: {correlation_pat*100}%', y=idx*0.2, fontsize=16)

# plt.tight_layout()
# plt.show()
# %%
# %%
# fig = plt.figure(layout='constrained', figsize=(15, 20))
# subfig = fig.subfigures(len(correlation_values), 1, wspace=0.07)
# for idx, correlation_pat in enumerate(correlation_values):
    
#     ax = subfig[idx].subplots(1,len(betas),sharey=True)

#     for idx_2,beta in enumerate(betas):
#         structured_data_nb_pat = relative_nb_found_iter_num_pat_fixe_beta(all_data, correlation_pat, beta)
#         # Plotting for number of patterns fixed
#         legend = []
#         it = 0
#         for key in number_patterns:
#             value = structured_data_nb_pat[key]
#             # if it % 2 == 0:
#             ax[idx_2].plot(value[:,0], value[:,1])
#             legend.append(str(key))
#             it += 1
#         ax[idx_2].legend(legend, title="nb pattern:")
#         ax[idx_2].set_xlabel("number of iter")
#         ax[idx_2].set_ylabel("percentage of retrieve patterns")
#         ax[idx_2].set_title("Beta = "+ str(beta))
    
#     # Add a shared title for the two plots in the same row
#     subfig[idx].suptitle(f'random flips: {correlation_pat*100}%', fontsize=20)

# # plt.tight_layout()
# plt.show()
# %%
repetitions = [0,1,2,3,4,5,6,7,8,9]
fig = plt.figure(layout='constrained', figsize=(15, 20))
subfig = fig.subfigures(len(correlation_values), 1, wspace=0.07)
for idx, correlation_pat in enumerate(correlation_values):
    
    ax = subfig[idx].subplots(1,len(number_patterns),sharey=True)

    for idx_2,nb_pat in enumerate(number_patterns):
        structured_data = relative_nb_found_iter_repetition_fixe_num_pat(all_data, correlation_pat, nb_pat)
        # Plotting for number of patterns fixed
        legend = []
        it = 0
        value = structured_data[0]
        ax[idx_2].plot(value[:,0], value[:,1])
        it += 1
        ax[idx_2].set_xlabel("number of iter")
        ax[idx_2].set_ylabel("percentage of retrieve patterns")
        ax[idx_2].set_title("number of pattern = "+ str(nb_pat))
    
    # Add a shared title for the two plots in the same row
    subfig[idx].suptitle(f'random flips: {correlation_pat*100}%', fontsize=20)

# plt.tight_layout()
plt.show()
# %%
structured_data = relative_nb_found_iter_repetition_fixe_num_pat(all_data, 0.3, 21)
# %%

plt.plot(structured_data[0][:,0],structured_data[0][:,1])
# %%
