#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   

#%%
data_sleep = np.loadtxt("../data/queries_output_sleep.data")
data_no_sleep = np.loadtxt("../data/queries_output_no_sleep.data")
# %%
correlations_sleep = []
for i in data_sleep:
    correlations_sleep.append(pearsonr(data_sleep[0],i)[0])

correlations_no_sleep = []
for i in data_no_sleep:
    correlations_no_sleep.append(pearsonr(data_no_sleep[0],i)[0])
# %%
plt.plot(correlations_sleep[:100])
plt.plot(correlations_no_sleep[:100])
plt.xlabel("nombre itérations")
plt.ylabel("pearson coefficient")
plt.legend(["sleep", "no_sleep"], loc="lower right")
# %%
plt.plot(correlations_sleep)
#plt.plot(correlations_no_sleep[:100])
plt.xlabel("nombre itérations")
plt.ylabel("pearson coefficient")
plt.legend(["sleep"], loc="lower right")
# %%
