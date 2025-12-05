#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Modern seaborn styling
sns.set_style("darkgrid")
sns.set_context("paper", font_scale=3)
#%%
# Read the CSV file
data_all = pd.read_csv('correlation_random_query_perceptron_spurious/all_simulation_data.csv')
data= data_all[data_all["learning_rule"]==0]
#%%
# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_fnd_pat'] / data['nb_pat']

#%%
# Get unique numbers of stored patterns
unique_ratio_rnd_bits = sorted(data['ratio_rnd_bits'].unique())
# unique_ratio_rnd_bits.remove(0.7)
#%%
# Create the plot
fig, axes = plt.subplots(2, 2, figsize=(20,15),sharex=True)
# Create the plot
for ratio_rnd_bits in unique_ratio_rnd_bits:
    # Filter data for this number of stored patterns
    subset = data[data['ratio_rnd_bits'] == ratio_rnd_bits]
    # Group by query_iter and calculate mean success_ratio
    averaged_subset = subset.groupby('query_iter')['success_ratio'].mean().reset_index()
    # Plot the averaged curve
    axes[0][0].plot(averaged_subset['query_iter']+1, averaged_subset['success_ratio'], label=f'{ratio_rnd_bits}',linewidth = 4)
axes[0][0].set( ylabel='RS')

for ratio_rnd_bits in unique_ratio_rnd_bits:
    print(ratio_rnd_bits)
    # Filter data for this number of stored patterns
    subset = data[data['ratio_rnd_bits'] == ratio_rnd_bits]
    subset["ratio_spurious_recovered"] = subset["nb_spurious"]/subset["nb_fnd_pat"]
    
    # Group by query_iter and calculate mean success_ratio
    # averaged_subset = subset.groupby(['query_iter'])["ratio_spurious_recovered"].mean().reset_index()
    averaged_subset = subset.groupby(['query_iter'])["nb_spurious"].mean().reset_index()
    # Plot the averaged curve
    # axes[1].plot(averaged_subset['query_iter']+1, averaged_subset["ratio_spurious_recovered"], label=f'{ratio_rnd_bits:.2f}',linewidth = 4)
    axes[0][1].plot(averaged_subset['query_iter']+1, averaged_subset["nb_spurious"], label=f'{ratio_rnd_bits:.2f}',linewidth = 4)
# axes[1][0].legend(title='Ratio of random bits')
axes[0][1].set( ylabel='Nb spurious patterns')

data= data_all[data_all["learning_rule"]==1]
# Calculate the ratio of successfully queried patterns
data['success_ratio'] = data['nb_fnd_pat'] / data['nb_pat']
# Get unique numbers of stored patterns
unique_ratio_rnd_bits = sorted(data['ratio_rnd_bits'].unique())
for ratio_rnd_bits in unique_ratio_rnd_bits:
    # Filter data for this number of stored patterns
    subset = data[data['ratio_rnd_bits'] == ratio_rnd_bits]
    # Group by query_iter and calculate mean success_ratio
    averaged_subset = subset.groupby('query_iter')['success_ratio'].mean().reset_index()
    # Plot the averaged curve
    axes[1][0].plot(averaged_subset['query_iter']+1, averaged_subset['success_ratio'], label=f'{ratio_rnd_bits}',linewidth = 4)
axes[1][0].set(xlabel='Nb queries',ylabel='RS')
axes[1][0].legend(title='Ratio random bits')
for ratio_rnd_bits in unique_ratio_rnd_bits:
    print(ratio_rnd_bits)
    # Filter data for this number of stored patterns
    subset = data[data['ratio_rnd_bits'] == ratio_rnd_bits]
    subset["ratio_spurious_recovered"] = subset["nb_spurious"]/subset["nb_fnd_pat"]
    
    # Group by query_iter and calculate mean success_ratio
    # averaged_subset = subset.groupby(['query_iter'])["ratio_spurious_recovered"].mean().reset_index()
    averaged_subset = subset.groupby(['query_iter'])["nb_spurious"].mean().reset_index()
    # Plot the averaged curve
    # axes[1].plot(averaged_subset['query_iter']+1, averaged_subset["ratio_spurious_recovered"], label=f'{ratio_rnd_bits:.2f}',linewidth = 4)
    axes[1][1].plot(averaged_subset['query_iter']+1, averaged_subset["nb_spurious"], label=f'{ratio_rnd_bits:.2f}',linewidth = 4)
axes[1][1].set(xlabel='Nb queries', ylabel='Nb spurious patterns')
# plt.xlim(-0.5, data['query_iter'].max())  # Set x-axis limit from 0 to max query_iter
# plt.ylim(0, 1.05)  # Set y-axis limit from 0 to 1.05 for better visualization

# plt.xlim(-0.5, data['query_iter'].max())  # Set x-axis limit from 0 to max query_iter
# plt.ylim(0, 1.05)  # Set y-axis limit from 0 to 1.05 for better visualization

plt.tight_layout()
plt.show()

# %%
