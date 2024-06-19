#%%
import numpy as np
from random import randint
#%%
class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        num_patterns = len(patterns)
        for pattern in patterns:
            pattern = pattern.reshape(self.num_neurons, 1)
            self.weights += np.dot(pattern, pattern.T)
        self.weights -= num_patterns * np.identity(self.num_neurons)

    def train_perceptron(self,patterns):
        for pattern in patterns:
            for i in range(len(pattern)):
                sum_in = 0
                for j in range((len(pattern))):
                    if j != i :
                        sum_in += pattern[j]*self.weights[j][i]
                for j in range((len(pattern))):
                    if j != i :
                        update = ((1-(sum_in*pattern[i]))*pattern[i]*pattern[j])*(1/len(pattern))
                        self.weights[j][i]+=update
                        self.weights[i][j]+=update
                
    def activate(self, state):
        return np.where(state >= 0, 1, -1)

    # def run(self, initial_state, max_steps=100):
    #     state = initial_state.copy()
    #     for step in range(max_steps):
    #         prev_state = state.copy()
    #         indices = np.arange(self.num_neurons)
    #         np.random.shuffle(indices)
    #         for i in indices:
    #             net_input = np.dot(self.weights[i], state)
    #             state[i] = self.activate(net_input)
    #         # Check for convergence (if state does not change anymore)
    #         if np.array_equal(state, prev_state):
    #             print(f"Converged in {step + 1} steps.")
    #             break
    #     return state

    def run(self, initial_state, steps=10):
        state = initial_state.copy()
        for _ in range(steps):
            state = self.activate(np.dot(self.weights, state))
        return state


def generate_patterns(nb_patterns,nb_neurons):
    patterns = []
    for i in range(nb_patterns):
        pattern = np.random.choice([-1,1],nb_neurons)
        patterns.append(pattern)
    
    return patterns

def generate_noisy_pattern(pattern,noise,nb_neurons):
    new_pattern = pattern.copy()
    indexes_to_change = np.random.choice(list(range(nb_neurons)),int(nb_neurons*noise))
    for i in indexes_to_change:
        new_pattern[i] = -pattern[i]
    return new_pattern

#%%
# Example usages
nb_neurons = 25
nb_patterns = 6
noise = 0.2
nb_iter = 4000
# Define patterns
patterns = generate_patterns(nb_patterns,nb_neurons)
# Initialize Hopfield Network
hopfield_net = HopfieldNetwork(nb_neurons)
print("the patterns :")
print(patterns)

# Train the network with the patterns
# for i in range(nb_iter):
#     hopfield_net.train_perceptron(patterns)
hopfield_net.train(patterns)
# Test the network with a noisy pattern
for i in range(len(patterns)):
    initial_state = generate_noisy_pattern(patterns[i],noise,nb_neurons)
    final_state = np.asarray(hopfield_net.run(initial_state))

    if np.all(final_state == np.asarray(patterns[i])) or np.all(final_state == np.asarray([-j for j in patterns[i]])):
        print("well stored")
    else:
        print("error")

    print("Initial state: ", initial_state)
    print("Final state:   ", final_state)


# %%
