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

plt.rcParams.update({'font.size': 15})
#%%
# Read the CSV file
# Fig_load_SR_average_new_inh_plas_many_betta_larger_networks
# myDir = "../../data/all_data_splited/sleep_simulations/Fig_load_SR_average_new_inh_plas_many_betta_larger_networks"
myDir = "../../data/all_data_splited/sleep_simulations/load_SR_explore_new_many_correlations"
data = pd.read_csv(myDir+'/all_simulation_data.csv')
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
nb_sim_one_parameter = len(all_repetitions)
#%%
# data_betas_correlations_1 = dict()
# for i,beta in enumerate(all_beta):
#     for j,correlation in enumerate(all_correlation):
#         data_one_beta_one_correlation = data[(data['beta']==beta) & (data['noise_level']==correlation)]
#         sim_IDs=np.sort(data['sim_ID'].unique())
#         for sim_id in sim_IDs:
#             is_error = False
#             mask = data_one_beta_one_correlation['sim_ID']==sim_id
#             sim_data = data_one_beta_one_correlation[mask]
#             sim_data_all_patterns_fnd = sim_data[sim_data['nb_fnd_pat']==sim_data['num_patterns']]
#             first_iter_all_fnd= 0
#             if len(sim_data_all_patterns_fnd)==0:
#                 is_error = True
#             first_iter_all_fnd_data=sim_data_all_patterns_fnd[sim_data_all_patterns_fnd['query_iter']==sim_data_all_patterns_fnd["query_iter"].min()]
#             first_iter_all_fnd = first_iter_all_fnd_data["query_iter"]+1

#             data_one_beta_one_correlation.loc[mask, 'is_error_before_all_fnd'] = is_error
#             data_one_beta_one_correlation.loc[mask, 'first_iter_all_fnd'] = first_iter_all_fnd
#             data_one_beta_one_correlation.loc[mask, 'noned'] = 1
#             data_one_beta_one_correlation.loc[mask, 'more_than_one_noned'] = 1
#         data_betas_correlations_1[(beta,correlation)]=data_one_beta_one_correlation
data_betas_correlations = dict()
#%%
for i,beta in enumerate(all_beta):
    for j,correlation in enumerate(all_correlation):
        data_one_beta_one_correlation = data[(data['beta']==beta) & (data['noise_level']==correlation)]
        # data_one_beta_one_correlation=data_betas_correlations_1[(beta,correlation)]
        # Keep only the last iteration as representent of the simulation
        data_one_beta_one_correlation = data_one_beta_one_correlation.loc[data_one_beta_one_correlation.groupby("sim_ID")["query_iter"].idxmax()]

        # is_error = data_one_beta_one_correlation["is_error_before_all_fnd"]
        # any_error = np.any(list(is_error))
        # # more_than_one_error = len(list(is_error))>((nb_sim_one_parameter*80)/100)
        # more_than_one_error = np.sum(list(is_error))>2
        # print(np.sum(list(is_error)))
        # if any_error:
        #     data_one_beta_one_correlation['noned']=None
        # if more_than_one_error:
        #     data_one_beta_one_correlation['more_than_one_noned']=None

        # data_one_beta_one_correlation["first_iter_all_fnd_normalized"] = data_one_beta_one_correlation['first_iter_all_fnd']/data_one_beta_one_correlation["num_patterns"]

        data_betas_correlations[(beta,correlation)]=data_one_beta_one_correlation # store the specific data for later

# %%
#for beta in all_beta:
for beta in all_beta:
    list_avg_nb_pat_found_beta = []
    list_std_nb_pat_found_beta = []
    # beta = all_beta[-1]
    for noise in all_correlation:
        corr = 1-noise
        for k,d in data_betas_correlations.items():
            if k[0]== beta and 1-k[1]== corr:
                list_nb_pattern_found = d["nb_fnd_pat"].values
                list_avg_nb_pat_found_beta.append(np.average(list_nb_pattern_found))
                list_std_nb_pat_found_beta.append(np.std(list_nb_pattern_found))
    list_avg_nb_pat_found_beta = np.array(list_avg_nb_pat_found_beta)
    list_std_nb_pat_found_beta = np.array(list_std_nb_pat_found_beta)
    plt.plot(all_correlation,list_avg_nb_pat_found_beta,label=str(beta))
    plt.fill_between(all_correlation,list_avg_nb_pat_found_beta, list_avg_nb_pat_found_beta - list_std_nb_pat_found_beta, list_avg_nb_pat_found_beta + list_std_nb_pat_found_beta, 
                 color='blue', alpha=0.05)


plt.legend()
plt.show()

# %%
