import numpy as np
from constants import OBSERVATION, ACTION, PLANT_ACTION

#TODO: add hidden layers
#TODO: add mutations

class Brain:
    def __init__(self, connections):
        self.connections = connections
    
    def process(self, observation, plant=False):
        action = PLANT_ACTION.copy() if plant else ACTION.copy()
        action_indices = {}

        for i, connection in enumerate(self.connections):
            a = connection[1]
            if a in action_indices:
                action_indices[a].append(i)
            else:
                action_indices[a] = [i]

        for a, indices in action_indices.items():
            sum = 0

            for index in indices:
                connection = self.connections[index]
                sum += observation[connection[0]] * connection[2]
            
            action[a] = np.tanh(sum)
        
        return action

brain = Brain([['food_distance', 'accelerate', 1.0], ['food_angle', 'turn', 1.0]])
observation = OBSERVATION.copy()
observation['food_distance'] = 2
observation['food_angle'] = 0.5
brain.process(observation)