from torch.utils.data import Dataset
from crowd_nav.utils.crowdgraph import CrowdNavGraph
import torch
import numpy as np


class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def push(self, item):
        # replace old experience with new experience
        if len(self.memory) < self.position + 1:
            self.memory.append(item)
        else:
            self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def is_full(self):
        return len(self.memory) == self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = list()

class GraphReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.state_graphs = list()
        self.next_state_graphs = list()
        self.actions = list()
        self.values = list()
        self.dones = list()
        self.rewards = list()
        self.position = 0

    def push(self, item):
        assert len(self.state_graphs) == len(self.next_state_graphs)
        assert len(self.state_graphs) == len(self.actions)
        assert len(self.state_graphs) == len(self.rewards)
        state, action, value, done, reward, next_state = item
        action = action.unsqueeze(dim=0)
        state_graph = CrowdNavGraph(state).graph
        next_state_graph = CrowdNavGraph(next_state).graph
        # replace old experience with new experience
        if len(self.state_graphs) < self.position + 1:
            self.state_graphs.append(state_graph)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.values.append(value)
            self.next_state_graphs.append(next_state_graph)
        else:
            self.state_graphs[self.position] = state_graph
            self.actions[self.position] = action
            self.values[self.position] = value
            self.dones[self.position] = done
            self.rewards[self.position] = reward
            self.next_state_graphs[self.position] = next_state_graph
        self.position = (self.position + 1) % self.capacity

    def __getitem__(self, item):
        return self.state_graphs[item], self.actions[item], self.values[item], self.dones[item], self.rewards[item], self.next_state_graphs[item]

    def __len__(self):
        return len(self.state_graphs)

    def clear(self):
        self.state_graphs = list()
        self.actions = list()
        self.rewards = list()
        self.dones = list()
        self.values = list()
        self.next_state_graphs = list()

class SafeGraphReplayMemory(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.state_graphs = list()
        self.next_state_graphs = list()
        self.actions = list()
        self.values = list()
        self.dones = list()
        self.rewards = list()
        self.constraints = list()
        self.position = 0

    def push(self, item):
        assert len(self.state_graphs) == len(self.next_state_graphs)
        assert len(self.state_graphs) == len(self.actions)
        assert len(self.state_graphs) == len(self.rewards)
        assert len(self.state_graphs) == len(self.constraints)
        state, action, value, done, reward, constraint, next_state = item
        action = action.unsqueeze(dim=0)
        state_graph = CrowdNavGraph(state).graph
        next_state_graph = CrowdNavGraph(next_state).graph
        # replace old experience with new experience
        if len(self.state_graphs) < self.position + 1:
            self.state_graphs.append(state_graph)
            self.actions.append(action)
            self.rewards.append(reward)
            self.constraints.append(constraint)
            self.dones.append(done)
            self.values.append(value)
            self.next_state_graphs.append(next_state_graph)
        else:
            self.state_graphs[self.position] = state_graph
            self.actions[self.position] = action
            self.values[self.position] = value
            self.dones[self.position] = done
            self.rewards[self.position] = reward
            self.constraints[self.position] = constraint
            self.next_state_graphs[self.position] = next_state_graph
        self.position = (self.position + 1) % self.capacity

    def __getitem__(self, item):
        return self.state_graphs[item], self.actions[item], self.values[item], self.dones[item], self.rewards[item], \
               self.constraints[item], self.next_state_graphs[item]

    def __len__(self):
        return len(self.state_graphs)

    def clear(self):
        self.state_graphs = list()
        self.actions = list()
        self.rewards = list()
        self.constraints = list()
        self.dones = list()
        self.values = list()
        self.next_state_graphs = list()