import numpy as np
from random import randint

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

    def activate(self, state):
        return np.where(state >= 0, 1, -1)

    def run(self, initial_state, steps=10):
        state = initial_state.copy()
        for _ in range(steps):
            state = self.activate(np.dot(self.weights, state))
        return state

# Example usages
if __name__ == "__main__":
    # Define patterns
    # patterns = [
    #     np.array([1, -1, 1, -1]),
    #     np.array([1, 1, -1, -1]),
    # ]
    filename = "input_data/correlated_patterns/patterns.data"
    patterns = np.loadtxt(filename)
    # Initialize Hopfield Network
    num_neurons = len(patterns[0])
    print(num_neurons)
    patterns = np.where(patterns == 0, -1, 1)
    hopfield_net = HopfieldNetwork(num_neurons)
    print("the patterns :")
    print(patterns)
    # Train the network with the patterns
    hopfield_net.train(patterns)

    # initial_state = np.random.choice([-1, 1], size=(num_neurons,), p=[1./2, 1./2])
    # final_state = hopfield_net.run(initial_state, steps=num_neurons)
    
    # Test the network with a noisy pattern
    for i in range(len(patterns)):
        initial_state = [j if np.random.choice([-1,1]) == 1 else np.random.choice([-1,1]) for j in patterns[i]]
        final_state = np.asarray(hopfield_net.run(initial_state, steps=num_neurons))
        if np.all(final_state == np.asarray(patterns[i])) or np.all(final_state == np.asarray([-1 if patterns[i][j] == 1 else 1 for j in patterns[i]])):
            print("well stored")
        else:
            print("error")

        # print("Initial state: ", initial_state)
        # print("Final state:   ", final_state)
            