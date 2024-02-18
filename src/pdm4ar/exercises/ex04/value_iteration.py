from typing import Tuple

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        # value_func = np.zeros_like(grid_mdp.grid).astype(float)
        # policy = np.zeros_like(grid_mdp.grid).astype(int)

        tol = 1e-3
        err = np.inf

        V_new = np.zeros((grid_mdp.num_states,1)).astype(float)
        V_old = np.zeros((grid_mdp.num_states,1)).astype(float)

        while(err>tol):
            # value_func_new = np.zeros_like(grid_mdp.grid).astype(float)
            # for i in range(grid_mdp.grid_wid):
            #     for j in range(grid_mdp.grid_len):
                    
            #         state = (i,j)
            #         next_states = grid_mdp.adjoining_states((i,j), grid_mdp.grid[state])
            #         next_states.append(state)
            #         if grid_mdp.start_cell not in next_states:
            #             next_states.append(grid_mdp.start_cell)
                    

            #         val = np.zeros((6))
            #         for action in [0,1,2,3,4,5]:
            #             temp = 0
            #             for next_st in next_states:
            #                 temp += grid_mdp.get_transition_prob(state, action, next_st)
            #                 val[action] += grid_mdp.get_transition_prob(state, action, next_st) * (grid_mdp.stage_reward(state, action, next_st) + grid_mdp.gamma * value_func[next_st])
            #             # print(temp)
            #             # if temp < 1 and temp !=0:
            #             #     print('Action:', action)
            #             #     print('State:', state)
            #                 # raise SystemExit(0)


            #         value_func_new[state] = max(val)
            #         policy[state] = np.argmax(val, axis=None)

            # err = np.max(abs(value_func_new - value_func))
            # value_func = value_func_new

            V_new = ValueIteration.bellman_optimality_operator(V_old, grid_mdp)
            err = np.max(abs(V_new-V_old))
            V_old = V_new

        policy_new = ValueIteration.policy_improvement(V_new, grid_mdp)

        value_func = V_new.reshape([grid_mdp.grid_wid, grid_mdp.grid_len])
        policy = policy_new.reshape([grid_mdp.grid_wid, grid_mdp.grid_len])

        return value_func, policy
