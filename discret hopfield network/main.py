import numpy as np

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

# Example usage
if __name__ == "__main__":
    # Define patterns
    patterns = [
        np.array([1, -1, 1, -1]),
        np.array([1, 1, -1, -1]),
    ]

    # Initialize Hopfield Network
    num_neurons = len(patterns[0])
    hopfield_net = HopfieldNetwork(num_neurons)

    # Train the network with the patterns
    hopfield_net.train(patterns)

    # Test the network with a noisy pattern
    initial_state = np.array([1, -1, 1, 1])
    final_state = hopfield_net.run(initial_state, steps=10)

    print("Initial state: ", initial_state)
    print("Final state:   ", final_state)
