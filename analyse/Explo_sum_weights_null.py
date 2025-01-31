#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Update the styling
sns.set_style("darkgrid")  # or "whitegrid", "dark", "white", "ticks"
sns.set_context("paper", font_scale=1.5)  # or "paper", "talk", "poster"
def equally_spaced_from_array(arr, n, ratio_taken):
    return arr[np.linspace(0, (len(arr)*ratio_taken)-1, n, dtype=int)]

def relative_iter(row,eta):
    # Replace this with your specific condition
    return row['query_iter']==int(eta*row['num_patterns'])

def get_spaced_indices(n, num_ticks=4):
    return np.linspace(0, n - 1, num_ticks, dtype=int)

def read_parameters_file(filepath):
    parameters = {}
    with open(filepath, 'r') as file:
        for line in file:
            # Split each line at '=' and strip any whitespace
            key, value = line.strip().split('=')
            
            # Try to convert the value to float if possible
            try:
                # Handle scientific notation (like 1e-10)
                value = float(value)
            except ValueError:
                # If conversion fails, keep it as string
                pass
            
            parameters[key] = value
    
    return parameters


# plt.rcParams.update({'font.size': 15})

#%%
# Read the CSV file
# Fig_load_SR_average_new_inh_plas_many_betta_larger_networks
# myDir = "../../data/all_data_splited/sleep_simulations/Fig_load_SR_average_new_inh_plas_many_betta_larger_networks"
myDir = "../../data/all_data_splited/sleep_simulations/Explo_null_w_sum"
data = pd.read_csv(myDir+'/all_simulation_last_iter_data.csv')
is_null_w_sum = True
# data = data[data['delta'] == 0.1]
#%%
# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_fnd_pat'] / (data['num_patterns'])
data['num_patterns'] = data['num_patterns'].astype(int)
data['error_ratio'] = 1
data['is_error_before_all_fnd'] = False
#%%
all_beta = np.sort(data['beta'].unique())
all_correlation = np.sort(data['noise_level'].unique())
all_repetitions= np.sort(data['repetitions'].unique())
all_nb_pat= np.sort(data['num_patterns'].unique())
nb_sim_one_parameter = len(all_repetitions)
null_w_sum=np.sort(data['null_w_sum'].unique())
data_betas_nb_pat_null_w_sum = dict()
#%%
#%%
# data for normalisation of the weights
for i,beta in enumerate(all_beta):
    for j,nb_pat in enumerate(all_nb_pat):
        data_one_beta_one_nb_pat = data[(data['beta']==beta) & (data['num_patterns']==nb_pat)& (data['null_w_sum']==is_null_w_sum)]
        data_betas_nb_pat_null_w_sum[(beta,nb_pat)]=data_one_beta_one_nb_pat # store the specific data for later

# %%
# #for beta in all_beta:
# for beta in all_beta:
#     list_avg_nb_pat_found_beta = []
#     list_std_nb_pat_found_beta = []
#     # beta = all_beta[-1]
#     for nb_pat in all_nb_pat:
#         for k,d in data_betas_nb_pat_null_w_sum.items():
#             if k[0]== beta and k[1]== nb_pat:
#                 list_nb_pattern_found = d["nb_fnd_pat"].values
#                 list_avg_nb_pat_found_beta.append(np.average(list_nb_pattern_found))
#                 list_std_nb_pat_found_beta.append(np.std(list_nb_pattern_found))
#     list_avg_nb_pat_found_beta = np.array(list_avg_nb_pat_found_beta)
#     list_std_nb_pat_found_beta = np.array(list_std_nb_pat_found_beta)
#     plt.plot(all_nb_pat,list_avg_nb_pat_found_beta,label=str(beta))
#     plt.fill_between(all_nb_pat,list_avg_nb_pat_found_beta, list_avg_nb_pat_found_beta - list_std_nb_pat_found_beta, list_avg_nb_pat_found_beta + list_std_nb_pat_found_beta, 
#                  color='blue', alpha=0.05)


# plt.legend()
# plt.show()
# %%
nb_sim = 1
for i in range(nb_sim):
    sim_dir_name = 'sim_nb_'
    folder_sim = myDir + "/" + sim_dir_name + "" + str(i)
    weights_file = folder_sim+"/weights.data"
    parameter_file = folder_sim+"/parameters.data"
    weights = np.loadtxt(weights_file)
    parameters = read_parameters_file(parameter_file)
    plt.hist(weights.flatten())
    plt.show()
    print(parameters["null_w_sum"])
    print(parameters["num_patterns"])
# %%
plt.hist(weights.flatten())
# %%
for i in range(nb_sim):
    sim_dir_name = 'sim_nb_'
    folder_sim = myDir + "/" + sim_dir_name + "" + str(i)
    weights_file = folder_sim+"/weights.data"
    parameter_file = folder_sim+"/parameters.data"
    weights = np.loadtxt(weights_file)  
    sums = np.sum(weights,axis=1)
    parameters = read_parameters_file(parameter_file)
    plt.hist(weights.flatten())
    plt.title("w hist")
    plt.show()
    plt.hist(sums.flatten())
    plt.title("sum row hist")
    plt.show()
    print(parameters["null_w_sum"])
    print(parameters["num_patterns"])
#%%
np.sum(weights[0])
# %%
np.sum(weights,axis=1)
# %%
