from ast import Tuple
import numpy as np

from numpy.ma import count

import sys



class PartitionCountingExploration:
    def __init__(
        self,
        stochastic_size: int, 
        discrete_size: int,
        num_actions: int,
        beta: float,
        start_count: float = 1.0,
    ) -> None:
        
        self.counts = np.zeros((num_actions, stochastic_size, discrete_size)) + start_count
        self.beta = beta

    def increment_counts(
        self,
        discrete_rep: np.ndarray,
        action: np.ndarray,
    ) -> None:
        action_id = np.argmax(action)
        self.counts[action_id] += discrete_rep

    def add_counts(
        self,
        counts: np.array,
    ) -> None:
        self.counts += counts
    
    def calculate_intrinsic_reward(
        self,
        discrete_rep: np.ndarray,
        action: np.ndarray,
        mid_counts: np.ndarray,
    ) -> float:
        num_action = np.argmax(action)
        counts = np.sum((self.counts + mid_counts.counts) * discrete_rep, axis=2)
        counts = np.min(counts, axis=1)
        count_s = np.sum(counts)
        count_s_a = counts[num_action]
        return self.beta * np.sqrt(2 * np.log(count_s) / count_s_a)
        

    def reset_counts(
        self
    ) -> None:
        self.counts = np.ones(self.counts.shape)
        

