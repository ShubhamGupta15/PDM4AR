from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq    # you may find this helpful

from osmnx.distance import great_circle_vec

from pdm4ar.exercises.ex02.structures import X, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # Abstract function. Nothing to do here.
        pass

    def path_recons(self, start: X, goal: X, parent_node) -> Path:
        path = [goal]
        curr_node = goal

        while(True):
            if curr_node == start:
                break
            
            curr_node = parent_node[curr_node]
            path.append(curr_node)

        return list(reversed(path))


@dataclass
class UniformCostSearch(InformedGraphSearch):
    def path(self, start: X, goal: X) -> Path:
        # todo
        self.cost_to_reach = {start:0}
        pri_que = [(0,start)]
        heapq.heapify(pri_que)
        visited = []
        
        parent_node = {}
        while len(pri_que)>0:
            (cost,state) = heapq.heappop(pri_que)
            if cost != self.cost_to_reach[state]:
                continue

            if state == goal:
                return self.path_recons(start,goal, parent_node)

            for node in self.graph.adj_list[state]:
                new_cost_reach = self.cost_to_reach[state] + self.graph.get_weight(state, node)
                if new_cost_reach < self.cost_reach(node):
                    self.cost_to_reach[node] = new_cost_reach
                    parent_node[node] = state
                    heapq.heappush(pri_que, (new_cost_reach, node))

        return []

    def cost_reach(self, u):
        try:
            cost = self.cost_to_reach[u]
        except:
            cost = 1e20
        return cost

@dataclass
class Astar(InformedGraphSearch):

    # Keep track of how many times the heuristic is called
    heuristic_counter: int = 0
    # Allows the tester to switch between calling the students heuristic function and
    # the trivial heuristic (which always returns 0). This is a useful comparison to
    # judge how well your heuristic performs.
    use_trivial_heuristic: bool = False

    def heuristic(self, u: X, v: X) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        self.heuristic_counter += 1
        if self.use_trivial_heuristic:
            return 0
        else:
            # return the heuristic that the student implements
            return self._INTERNAL_heuristic(u, v)

    # Implement the following two functions

    def _INTERNAL_heuristic(self, u: X, v: X) -> float:
        # Implement your heuristic here. Your `path` function should NOT call
        # this function directly. Rather, it should call `heuristic`
        lat1,long1 = self.graph.get_node_coordinates(u)
        lat2,long2 = self.graph.get_node_coordinates(v)
        heuristic_value = great_circle_vec(lat1,long1,lat2,long2)/(TravelSpeed.HIGHWAY.value*2)
        return heuristic_value
        
    def path(self, start: X, goal: X) -> Path:

        self.cost_to_reach = {start:0}
        cost_to_reach_goal = {start: self.heuristic(start,goal)}
        pri_que = [(cost_to_reach_goal[start],0, start)]
        heapq.heapify(pri_que)
        parent_node = {}


        while len(pri_que)>0:
            (cost_goal,cost,state) = heapq.heappop(pri_que)

            if state == goal:
                return self.path_recons(start,goal, parent_node)   # utility funtion for back-calculating the path

            if cost_goal != cost_to_reach_goal[state]:   # to only check the instance for a node in pri queue with min cost
                continue

            for node in self.graph.adj_list[state]:
                new_cost_reach = self.cost_to_reach[state] + self.graph.get_weight(state, node) 
                if new_cost_reach < self.cost_reach(node):
                    self.cost_to_reach[node] = new_cost_reach
                    parent_node[node] = state
                    cost_to_reach_goal[node] = new_cost_reach + self.heuristic(node, goal)
                    heapq.heappush(pri_que, (cost_to_reach_goal[node], self.cost_to_reach[node] , node))
        return []

    def cost_reach(self, u):
        try:
            cost = self.cost_to_reach[u]
        except:
            cost = 1e20
        return cost


def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
