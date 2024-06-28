#%%
import os
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.stats.stats import pearsonr   
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
    data["path"] = folder_path
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
# def structure_data_max_retrieve_pattern(all_data):
#     structured_data = defaultdict(list)

#     for data in all_data:
#         if data['parameters'] is not None and data['results'] is not None:
#             parameters_df = data['parameters']
#             results_df = data['results']
#             #single values
#             network_size = int(parameters_df.loc[parameters_df['parameter'] == 'network_size', 'value'].values[0])
#             # beta = float(parameters_df.loc[parameters_df['Parameter'] == 'beta', 'Value'].values[0])
#             #Many values
#             #nb_iter = results_df.loc[results_df['Parameter'] == 'iter', 'Value'].values
#             nb_found_pattern = results_df["nb_found_patterns"].tolist()
#             for index in range(len(nb_found_pattern)):
#                 nb_found_pattern[index] = int(nb_found_pattern[index])
#             structured_data[network_size].append(max(nb_found_pattern) )          

#             # max_nb_found_pattern = max(nb_found_pattern)

#             # structured_data['network_size'].append(network_size)
#             # structured_data['beta'].append(beta)
#             # structured_data['max_nb_found_pattern'].append(max_nb_found_pattern)
#     structured_list = [[],[]]
#     for key,value in structured_data.items() :
#         structured_list[0].append(key)
#         structured_list[1].append(max(value)) # Maximum stored pattern

#     # structured_df = pd.DataFrame(structured_data)
#     return structured_list

# #%% Function to structure the data for analysis
# def structure_data_max_retrieve_pattern_no_spurious(all_data):
#     structured_data = defaultdict(list)

#     for data in all_data:
#         if data['parameters'] is not None and data['results'] is not None:
#             parameters_df = data['parameters']
#             results_df = data['results']
#             #single values
#             network_size = int(parameters_df.loc[parameters_df['parameter'] == 'network_size', 'value'].values[0])
#             # beta = float(parameters_df.loc[parameters_df['Parameter'] == 'beta', 'Value'].values[0])
#             #Many values
#             nb_found_pattern_no_spurious = results_df.loc[results_df['nb_spurious_patterns'] == '0', 'nb_found_patterns'].values
#             for index in range(len(nb_found_pattern_no_spurious)):
#                 nb_found_pattern_no_spurious[index] = int(nb_found_pattern_no_spurious[index])
#             nb_found_pattern_no_spurious= np.append(nb_found_pattern_no_spurious,[0])
#             structured_data[network_size].append(max(nb_found_pattern_no_spurious) )          

#     structured_list = [[],[]]
#     print(structured_data)
#     for key,value in structured_data.items() :
#         structured_list[0].append(key)
#         structured_list[1].append(max(value)) # Maximum stored pattern

    # structured_df = pd.DataFrame(structured_data)
    # return structured_list

# Down sample by 10 !
def get_traj(path): 
    print(path)
    if os.path.exists(path):
        return np.loadtxt(path)[::30,:]
    else:
        print("error no file found")
    pass

#Get the trajectories of the same network size and relative_num_pattern but mooving betas
def get_trajectories(all_data,net_size_target,relative_num_pattern_target):

    trajectories = defaultdict(list)
    patterns = defaultdict(list)
    nb_found = 0 

    for data in all_data:
        if data['parameters'] is not None and data['results'] is not None:
            parameters_df = data['parameters']
            nb_iter_mult = float(parameters_df.loc[parameters_df['parameter'] == 'nb_iter_mult', 'value'].values[0])
            beta = float(parameters_df.loc[parameters_df['parameter'] == 'beta', 'value'].values[0])
            relative_num_patterns = float(parameters_df.loc[parameters_df['parameter'] == 'relative_num_patterns', 'value'].values[0])
            network_size = int(parameters_df.loc[parameters_df['parameter'] == 'network_size', 'value'].values[0])
            num_patterns = int(parameters_df.loc[parameters_df['parameter'] == 'num_patterns', 'value'].values[0])

            nb_traj = int(nb_iter_mult*num_patterns)
            # condition true only once
            if network_size == net_size_target and relative_num_patterns == relative_num_pattern_target:
                nb_found+=1
                print(nb_found)
                for i in range(nb_traj):
                    trajectories[beta].append(get_traj(data["path"]+"\\results_"+str(i)+".data"))
                patterns[beta] = np.loadtxt(data["path"]+"\\patterns.data")
 
        
    return trajectories,patterns

#%% Collect data from all subfolders
base_folder = "..\\..\\..\\data\\all_data_splited\\sleep_simulations\\sleep_parameter_test"
file_to_query = ""
nb_sim = 1000
all_data = collect_data(base_folder,nb_sim)
#%% Structure the collected data
all_data_traj,patterns = get_trajectories(all_data,25,0.544444)
#%%
print(patterns)
#%%
# as there is the same number of pattern we can have a fixe number of phases
nb_phases = len(all_data_traj[0.1])
#%%
# data_sleep_phases = [] 
# for betas,traj in all_data_traj.items():
#     data_sleep_phases.append([])
#     for i in range(nb_phases): #take one randome beta
#         print(betas)
#         data_sleep_phases[-1].append(all_data_traj[betas][i])
# data_sleep_phases = np.array(data_sleep_phases)
# %%
correlations_sleep_all = []
for beta,traj in all_data_traj.items():
    print("actual beta")
    print(beta)
    correlations_sleep_all.append([])
    for i in range(len(all_data_traj[beta])):
        correlations_sleep_all[-1].append([])
        for j in range(len(all_data_traj[beta][i])):
            correlations_sleep_all[-1][-1].append([])
            for k in range(len(patterns[beta])):
                correlations_sleep_all[-1][-1][-1].append(pearsonr(all_data_traj[beta][i][j],patterns[beta][k])[0])
  
#%%
nb_betas = len(all_data_traj)
nb_patterns = len(patterns[list(all_data_traj.keys())[0]])
#%%
for h in range(nb_betas):
    correlations_sleep_all[h] = np.array(correlations_sleep_all[h])
#%%
concatenated_trajs_all = []
xcoords_all = []
divide_nb_iter = 3
for h in range(nb_betas):
    xcoords_all.append([])
    concatenated_trajs_all.append([[] for _ in range(nb_patterns)])
    added = 0
    for i in range(int(len(correlations_sleep_all[h])/divide_nb_iter)):
        xcoords_all[-1].append(len(correlations_sleep_all[h][i])+added)
        added += len(correlations_sleep_all[h][i])
        for j in range(len(correlations_sleep_all[h][i])):
            for k in range(len(correlations_sleep_all[h][i][j])):
                concatenated_trajs_all[h][k].append(correlations_sleep_all[h][i][j][k])

#%%
from matplotlib.ticker import ScalarFormatter
patterns_legend = []
for i in range(nb_patterns):
    patterns_legend.append("pattern "+str(i))

for h in range(nb_betas):
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    plt.gca().yaxis.get_offset_text().set_visible(False)
    plt.plot(np.array(concatenated_trajs_all[h]).T)
    plt.ylabel("pearson coefficient")
    plt.xlabel("nombre d'itérations")
    # plt.legend(["pattern 1","pattern 2","pattern 3","pattern 4","pattern 5","..."], loc="lower right")
    # plt.legend(patterns_legend, loc="lower right")

    # plt.title("SR for " + str(h+1)+" pattern(s)")
    # plt.savefig("figs\SR for " + str(h+1)+" pattern(s)")
    # xcoords = xcoords_all[h]
    # for xc in xcoords:
    #     plt.axvline(x=xc,c = "red", linestyle = "--")
    plt.show()
# %%
