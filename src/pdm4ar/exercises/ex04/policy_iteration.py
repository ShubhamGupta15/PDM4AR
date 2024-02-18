from typing import Tuple

import numpy as np

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy
from pdm4ar.exercises_def.ex04.utils import time_function


class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        # value_func = np.zeros_like(grid_mdp.grid).astype(float)
        # policy = np.zeros_like(grid_mdp.grid).astype(int)

        policy_new = np.zeros((grid_mdp.num_states,1)).astype(int)
        policy_old = np.zeros((grid_mdp.num_states,1)).astype(int)
        V_new = np.zeros((grid_mdp.num_states,1)).astype(float)
        V_old = np.zeros((grid_mdp.num_states,1)).astype(float)
        
        while True:
            V_new = PolicyIteration.policy_evaluation(policy_old, grid_mdp)                        # policy evaluation step
            
            policy_new = PolicyIteration.policy_improvement(V_new, grid_mdp)                      # policy imrovement step
            
            # print(policy_new,'\n', self.policy)
            if np.all(policy_new == policy_old) and np.all(V_new == V_old):
                break
            else:
                policy_old = policy_new
                V_old = V_new
            
         
        # self.V = self.V_from_pi(self.policy)

        value_func = V_new.reshape([grid_mdp.grid_wid, grid_mdp.grid_len])
        policy = policy_new.reshape([grid_mdp.grid_wid, grid_mdp.grid_len])

        return value_func, policy