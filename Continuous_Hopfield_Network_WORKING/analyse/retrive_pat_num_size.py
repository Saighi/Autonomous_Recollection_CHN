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
            beta = float(parameters_df.loc[parameters_df['parameter'] == 'beta', 'value'].values[0])
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

def relative_nb_found_iter_relative_num_pat_fixe_beta_network_size(data,net_size_target, beta_target):

    structured_data = defaultdict(list)
    nb_found = 0
    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            beta = float(parameters_df.loc[parameters_df['parameter'] == 'beta', 'value'].values[0])
            net_size = float(parameters_df.loc[parameters_df['parameter'] == 'network_size', 'value'].values[0])

            if beta==beta_target and net_size == net_size_target:
                nb_found+=1
                nb_found_pat_iter = []
                relative_nb_pat = float(parameters_df.loc[parameters_df['parameter'] == 'relative_num_patterns', 'value'].values[0])
                nb_pat = int(parameters_df.loc[parameters_df['parameter'] == 'num_patterns', 'value'].values[0])
                nb_iter_sim = int(parameters_df.loc[parameters_df['parameter'] == 'nb_iter_mult', 'value'].values[0])*nb_pat
                for i in range(nb_iter_sim):
                    nb_found_pat_iter.append([i,int(results_df.loc[results_df['iter'] == str(i), 'nb_found_patterns'].values[0])/nb_pat])
                structured_data[relative_nb_pat]= np.array(nb_found_pat_iter)          

    return structured_data

def relative_nb_found_iter_beta_fixe_relative_num_pat_network_size(data,net_size_target, num_pat_target):

    structured_data = defaultdict(list)
    nb_found = 0
    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            results_df = data['results']
            net_size = float(parameters_df.loc[parameters_df['parameter'] == 'network_size', 'value'].values[0])
            relative_nb_pat = float(parameters_df.loc[parameters_df['parameter'] == 'relative_num_patterns', 'value'].values[0])

            if relative_nb_pat==num_pat_target and net_size == net_size_target:
                nb_found+=1
                nb_found_pat_iter = []
                beta = float(parameters_df.loc[parameters_df['parameter'] == 'beta', 'value'].values[0])
                nb_pat = int(parameters_df.loc[parameters_df['parameter'] == 'num_patterns', 'value'].values[0])
                nb_iter_sim = int(parameters_df.loc[parameters_df['parameter'] == 'nb_iter_mult', 'value'].values[0])*nb_pat
                for i in range(nb_iter_sim):
                    nb_found_pat_iter.append([i,int(results_df.loc[results_df['iter'] == str(i), 'nb_found_patterns'].values[0])/nb_pat])
                structured_data[beta]= np.array(nb_found_pat_iter)          

    return structured_data
#%% Collect data from all subfolders
base_folder = "..\\..\\..\\data\\all_data_splited\\sleep_simulations\\sleep_parameter_test"
all_data = collect_data(base_folder)
# betas = [0.001,0.023,0.067,0.1]
# network_sizes = []
# # number_patterns = [5,7,10,13,16,18,21,24,27,30]
# number_patterns = [5,10,16,21,27,30]
#%%
betas_string = get_param_list(all_data,"beta")
betas = np.sort(list(set(map(float,betas_string))))
betas = betas[0::2]
relative_num_pat_string = get_param_list(all_data,"relative_num_patterns")
relative_number_patterns = np.sort(list(set(map(float,relative_num_pat_string))))
relative_number_patterns = relative_number_patterns[0::2]
net_sizes_string = get_param_list(all_data,"network_size")
net_sizes = np.sort(list(set(map(int,net_sizes_string))))
net_sizes = net_sizes[0::2]
#%%
fig = plt.figure(layout='constrained', figsize=(20, 20))
subfig = fig.subfigures(len(net_sizes), 1, wspace=0.07)
for idx, net_size in enumerate(net_sizes):
    
    ax = subfig[idx].subplots(1,len(betas),sharey=True)

    for idx_2,beta in enumerate(betas):
        structured_data_nb_pat = relative_nb_found_iter_relative_num_pat_fixe_beta_network_size(all_data,net_size , beta)
        # Plotting for number of patterns fixed
        legend = []
        it = 0
        for key in relative_number_patterns:
            value = structured_data_nb_pat[key]
            # if it % 2 == 0:
            ax[idx_2].plot(value[:,0], value[:,1])
            legend.append(str(round(key,2)))
            it += 1
        ax[idx_2].legend(legend, title="rel nb pat:")
        ax[idx_2].set_xlabel("number of iter")
        ax[idx_2].set_ylabel("percentage of retrieve patterns")
        ax[idx_2].set_title("Beta = " +str(beta))
    
    # Add a shared title for the two plots in the same row
    subfig[idx].suptitle(f'network size: {net_size}', fontsize=20)

plt.show()
#%%
fig = plt.figure(layout='constrained', figsize=(20, 20))
subfig = fig.subfigures(len(net_sizes), 1, wspace=0.07)
for idx, net_size in enumerate(net_sizes):
    
    ax = subfig[idx].subplots(1,len(relative_number_patterns),sharey=True)

    for idx_2,relative_nb_pat in enumerate(relative_number_patterns):
        structured_data_nb_pat = relative_nb_found_iter_beta_fixe_relative_num_pat_network_size(all_data,net_size, relative_nb_pat)

        # Plotting for number of patterns fixed
        legend = []
        it = 0
        for key in betas:
            value = structured_data_nb_pat[key]
            ax[idx_2].plot(value[:,0], value[:,1])
            legend.append(str(round(key,2)))
            it += 1
        ax[idx_2].legend(legend, title="beta :")
        ax[idx_2].set_xlabel("number of iter")
        ax[idx_2].set_ylabel("percentage of retrieve patterns")
        ax[idx_2].set_title("relative number  of pattern = "+ str(round(relative_nb_pat,2)))
    
    # Add a shared title for the two plots in the same row
    subfig[idx].suptitle(f'network size: {net_size}', fontsize=20)

plt.show()
# %%
