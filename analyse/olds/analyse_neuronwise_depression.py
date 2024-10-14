#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   

#%%
nb_phases = 20
nb_betas =  20
nb_patterns = 5
data_sleep_phases = [] 
for h in range(nb_betas):
    data_sleep_phases.append([])
    for i in range(nb_phases):
        data_sleep_phases[-1].append(np.loadtxt("../../data/spontaneous_recollection_neuronwise_adaptation/output_sleep_"+str(h)+"_"+str(i)+".data"))
data_sleep_phases = np.array(data_sleep_phases)
patterns = np.loadtxt("../../data/spontaneous_recollection_neuronwise_adaptation/output_sleep_patterns.data")
#%%
patterns.size
# %%
correlations_sleep_all = []
for h in range(nb_betas):
    correlations_sleep_all.append([])
    for i in range(nb_phases):
        correlations_sleep_all[-1].append([])
        for j in range(len(data_sleep_phases[h][i])):
            correlations_sleep_all[-1][-1].append([])
            for k in range(nb_patterns):
                correlations_sleep_all[-1][-1][-1].append(pearsonr(data_sleep_phases[h][i][j],patterns[k])[0])
  
#%%
for h in range(nb_betas):
    correlations_sleep_all[h] = np.array(correlations_sleep_all[h])
#%%
concatenated_trajs_all = []
xcoords_all = []
for h in range(nb_betas):
    xcoords_all.append([])
    concatenated_trajs_all.append([[] for _ in range(nb_patterns)])
    added = 0
    for i in range(len(correlations_sleep_all[h])):
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
    plt.legend(patterns_legend, loc="lower right")

    # plt.title("SR for " + str(h+1)+" pattern(s)")
    plt.savefig("figs\SR for " + str(h+1)+" pattern(s)")
    xcoords = xcoords_all[h]
    for xc in xcoords:
        plt.axvline(x=xc,c = "red", linestyle = "--")
    plt.show()
# # %%
# plt.plot(correlations_sleep[:100])
# plt.plot(correlations_no_sleep[:100])
# plt.xlabel("nombre itérations")
# plt.ylabel("pearson coefficient")
# plt.legend(["sleep", "no_sleep"], loc="lower right")
# # %%
# plt.plot(correlations_sleep)
# #plt.plot(correlations_no_sleep[:100])
# plt.xlabel("nombre itérations")
# plt.ylabel("pearson coefficient")
# plt.legend(["sleep"], loc="lower right")
# # %%
# %%
