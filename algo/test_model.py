import unittest
import torch
from integratedmodel import DQN, ReplayMemory

class TestDQN(unittest.TestCase):

    def setUp(self):
        self.n_observations = 10
        self.n_actions = 5
        self.model = DQN(self.n_observations, self.n_actions)

    def test_dqn_initialization(self):
        self.assertEqual(len(list(self.model.parameters())), 6)  # 3 layers * 2 parameters per layer (weights + bias)

    def test_dqn_forward_pass(self):
        input_tensor = torch.randn(1, self.n_observations)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, torch.Size([1, self.n_actions]))

class TestReplayMemory(unittest.TestCase):

    def setUp(self):
        self.memory_capacity = 100
        self.replay_memory = ReplayMemory(self.memory_capacity)

    def test_memory_initialization(self):
        self.assertEqual(len(self.replay_memory.memory), 0)

    def test_memory_push(self):
        self.replay_memory.push(1, 2, 3, 4)  # Example Transition
        self.assertEqual(len(self.replay_memory.memory), 1)

    def test_memory_sample(self):
        for _ in range(10):
            self.replay_memory.push(1, 2, 3, 4)  # Add 10 transitions
        sample = self.replay_memory.sample(5)
        self.assertEqual(len(sample), 5)

if __name__ == '__main__':
    unittest.main()

