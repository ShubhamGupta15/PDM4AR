from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Cell, Policy, State, ValueFunc


class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""
        self.grid_wid = self.grid.shape[0]
        self.grid_len= self.grid.shape[1]
        self.num_states = self.grid_wid * self.grid_len
        self.num_actions = 6

        self.start_cell = (np.where(grid == 1)[0][0], np.where(grid == 1)[1][0])
        self.goal_cell = (np.where(grid == 0)[0][0], np.where(grid == 0)[1][0])

        self._vectorise_P_R()
        
    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""

        cell_type = self.grid[state]
        adj_states = self.adjoining_states(state, cell_type)
        des_next_st = self.desired_next_state(state, action)

        # for the goal cell
        if cell_type == 0:
            if action == Action.STAY and next_state == self.goal_cell: # only stay action allowed
                return 1
            else:
                return 0

        #for the start cell or grass cell
        elif cell_type == 1 or cell_type == 2:   #action not allowed
            if action == Action.STAY:
                return 0

            elif action == Action.ABANDON:
                if next_state == des_next_st:
                    return 1
                else:
                    return 0

            elif action in self.get_allowed_actions(state):                                      # move somewhere
                if not self.inside_map(des_next_st):    #action not allowed
                    return 0
                if next_state == des_next_st:           # moved in correct dirn
                    return 0.75
                elif next_state in adj_states:          # moved elsewhere
                    return 0.25/3
                elif next_state == self.start_cell:
                    return (0.25/3)*(4-len(adj_states))
                else:
                    return 0
            else:
                return 0

        # for swamp cell
        elif cell_type == 3:
            if action == Action.STAY:# action not allowed
                return 0

            if action == Action.ABANDON:
                if next_state == des_next_st:
                    return 1
                else:
                    return 0

            elif action in self.get_allowed_actions(state):                                       # move somewhere
                if not self.inside_map(des_next_st):      #action not allowed
                    return 0                                   
                if next_state == des_next_st:           # moved in correct dirn
                    return 0.5
                elif next_state in adj_states:          # moved elsewhere
                    return 0.25/3
                elif next_state == state:               # could not move
                    return 0.2
                elif next_state == self.start_cell:     # breakdown
                    return 0.05 + (0.25/3)*(4-len(adj_states))
                else:
                    return 0
            else:
                return 0

        else:
            print('Not a valid cell type')


    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        """Returns R(next_state | state, action)"""

        cell_type = self.grid[state]
        adj_states = self.adjoining_states(state, cell_type)
        des_next_st = self.desired_next_state(state, action)

        reward = 0

        # for the goal cell
        if cell_type == 0:
            if action == Action.STAY and next_state == self.goal_cell: # only stay action allowed
                reward += 50


        #for the start cell 
        elif cell_type == 1 :
            if action == Action.ABANDON:
                if next_state == des_next_st:
                    reward += -10
            
            else: 
                reward += -1                            #standard 1 hr to move from the cell

        # cell type grass
        elif  cell_type == 2:
            if action == Action.ABANDON:
                if next_state == des_next_st:
                    reward += -10
            
            else: 
                reward += -1                            #standard 1 hr to move from the cell
                # if next_state =   = self.goal_cell:
                #     reward += 0                                      
                if next_state in adj_states:           # moved in correct dirn
                    reward += 0
                elif next_state == self.start_cell:
                    reward += -10

        # for swamp cell
        elif cell_type == 3:
            if action == Action.ABANDON:
                if next_state == des_next_st:
                    reward += -10

            else:
                reward += -2                            # standard 2 hrs to move from that cell
                # if next_state == self.goal_cell:
                #     reward += 0                                   
                # elif next_state == des_next_st:           # moved in correct dirn
                #     reward += 0
                # elif next_state in adj_states:          # moved elsewhere
                #     reward += 0
                # elif next_state == state:               # could not move
                #     reward += 0
                if next_state == self.start_cell:     # breakdown
                    reward += -10
        else:
            print('Not a valid cell type')

        return reward

    #utility functions
    def inside_map(self, state):
        a = state[0]
        b = state[1]

        if a >= 0 and b >= 0 and b < self.grid_len and a < self.grid_wid:
            return True
        else:
            return False

    def adjoining_states(self, state, cell_type):
        adj_states = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if i == j or i+j == 0:
                    continue
                new_st = (state[0]+i , state[1]+j)
                if cell_type == 1 or self.inside_map(new_st):
                    adj_states.append(new_st)

        return adj_states

    def desired_next_state(self, state, action):
        if action == Action.WEST:
            return (state[0], state[1]-1)
        elif action == Action.EAST:
            return (state[0], state[1]+1)
        elif action == Action.SOUTH:
            return (state[0]+1, state[1])
        elif action == Action.NORTH:
            return (state[0]-1, state[1])
        elif action == Action.STAY:
            return state
        elif action == Action.ABANDON:
            return self.start_cell
        else:
            print('Not a valid action')
    
    def state_vectorise(self, state):
        return np.ravel_multi_index(state, (self.grid_wid,self.grid_len), mode='raise')

    def state_tuple(self,state_flat):
        return np.unravel_index(state_flat, (self.grid_wid,self.grid_len))

    def _vectorise_P_R(self):
        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))

        for i in range(self.num_states):
            for j in range(self.num_actions):
                for k in range(self.num_states):
                    ini_state = self.state_tuple(i)
                    nxt_state = self.state_tuple(k)
                    self.P[i,j,k] = self.get_transition_prob(ini_state, j, nxt_state)
                    self.R[i,j,k] = self.stage_reward(ini_state, j, nxt_state)

    def get_allowed_actions(self,state:State):
        state_type = self.grid[state]

        # add neighbouring cells to possible states
        cell_west = (state[0], state[1]-1)
        cell_east = (state[0], state[1]+1)
        cell_north = (state[0]-1, state[1])
        cell_south = (state[0]+1, state[1])
        cell_start = self.start_cell

        if state_type == Cell.START:
            return [Action.NORTH,Action.EAST,Action.SOUTH,Action.WEST]
        elif state_type == Cell.GRASS:
            if cell_start in [cell_west,cell_east,cell_south,cell_north]:
                return [Action.NORTH,Action.EAST,Action.SOUTH,Action.WEST]
            else:
                actions = [Action.ABANDON]
                if self.inside_map(cell_west):
                    actions.append(Action.WEST)
                if self.inside_map(cell_north):
                    actions.append(Action.NORTH)
                if self.inside_map(cell_east):
                    actions.append(Action.EAST)
                if self.inside_map(cell_south):
                    actions.append(Action.SOUTH)
                return actions
        elif state_type == Cell.SWAMP:
            actions = [Action.ABANDON]
            if self.inside_map(cell_west):
                    actions.append(Action.WEST)
            if self.inside_map(cell_north):
                actions.append(Action.NORTH)
            if self.inside_map(cell_east):
                actions.append(Action.EAST)
            if self.inside_map(cell_south):
                actions.append(Action.SOUTH)
            return actions
        else:
            return [Action.STAY]
        

class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        pass

    @staticmethod
    def policy_evaluation(policy, grid_mdp):
        A = np.zeros((grid_mdp.num_states, grid_mdp.num_states))
        b = np.zeros((grid_mdp.num_states, 1))
        # b = list(b)
        # A = list(A)
        for state in range(grid_mdp.num_states):
            A[state,:] = -grid_mdp.gamma*grid_mdp.P[state, int(policy[state]),:]
            A[state,state] += 1
            b[state] = np.sum(grid_mdp.P[state , int(policy[state]),:] * grid_mdp.R[state , int(policy[state]),:])
            
        return(np.linalg.solve(A, b))

    @staticmethod
    def policy_improvement(V, grid_mdp: GridMdp):
        Q = np.zeros((grid_mdp.num_states, grid_mdp.num_actions))
        for state in range(grid_mdp.num_states):
            # for action in range(self.A):
                Q[state,] = np.sum(grid_mdp.P[state,:,:]*(grid_mdp.R[state,:,:] + grid_mdp.gamma*V.T), axis = 1)
        
        pi = np.argmax(Q, axis = 1)
        return pi

    @staticmethod
    def bellman_optimality_operator(V, grid_mdp: GridMdp):

        V_new = np.zeros((grid_mdp.num_states,1)) 
        for state in range(grid_mdp.num_states):
                
                Ts = grid_mdp.P[state,:,:]
                Rs = grid_mdp.R[state,:,:]
                
                
                tmp = np.sum(Ts*(Rs + grid_mdp.gamma*V.T), axis = 1)
                
                V_new[state]  = np.max(tmp, axis = 0)

        # tmp = np.sum(self.P*[self.R + self.gamma*V.T], axis = 2)
        # print(np.max(tmp,axis=1))
        return V_new
