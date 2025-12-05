#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 25})
#%%
# Read the CSV file
data = pd.read_csv('correlation_random_query_perceptron_spurious/all_simulation_data.csv')
#%%

#%%
plt.rcParams.update({'font.size': 25})
plt.figure(figsize=(12, 8))
unique_nb_netsize = sorted(data['net_size'].unique())
for net_size in unique_nb_netsize:
    subset = data[data["net_size"]==net_size]
    max_query_iter_subset = subset[subset['query_iter'] == subset['query_iter'].max()].copy()
    mean_spurious_subset = max_query_iter_subset.groupby("nb_pat").nb_spurious.agg('mean').reset_index()
    plt.plot(mean_spurious_subset["nb_pat"],(mean_spurious_subset["nb_spurious"]/60)*100, linewidth = 4, label=f'{net_size}')

plt.xlabel('number of stored pattern')
plt.ylabel('ratio of spurious patterns encountered', fontsize = 20)
# plt.title('Average Pattern Discovery Success Ratio vs. Query Iteration')
plt.legend(title='Network size')
plt.grid(True, linestyle='--', alpha=0.7)