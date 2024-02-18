from abc import abstractmethod, ABC
from typing import Tuple

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
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


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        pri_queue = [start]
        opened_nodes = []
        parent_node = {}


        while len(pri_queue) != 0:
            open_node = pri_queue.pop()
            opened_nodes.append(open_node)

            if open_node == goal:
                path = self.path_recons(start, goal, parent_node)
                return (path, opened_nodes)
               
            adj_nodes = graph[open_node]
            adj_nodes = list(adj_nodes)
            adj_nodes.sort(reverse=True)    #adding the nodes to priority queue in ascending order
            
            for node in adj_nodes:
                if node in opened_nodes or node in pri_queue:
                    continue
                else:
                    parent_node[node] = open_node
                    pri_queue.append(node)

        path = []
        
        return (path, opened_nodes)



class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
        pri_queue = [start]
        opened_nodes = []
        parent_node = {}

        while len(pri_queue) != 0:
            open_node = pri_queue.pop(0)
            opened_nodes.append(open_node)

            if open_node == goal:
                path = self.path_recons(start, goal, parent_node)
                return (path, opened_nodes)
               
            adj_nodes = graph[open_node]
            adj_nodes = list(adj_nodes)

            adj_nodes.sort()    #adding the nodes to priority queue in ascending order
            for node in adj_nodes:
                if node in opened_nodes or node in pri_queue:
                    continue
                else:
                    parent_node[node] = open_node
                    pri_queue.append(node)

        path = []
        return (path, opened_nodes)


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:

        depth = 0
        depth_limit = len(graph) 
        all_explored = False
        for depth in range(0,depth_limit):
            self.parent_node = {}
            self.opened_nodes = []
            self.pri_que = []
            goal_reached = self.limited_depth_search(graph, start, goal, depth)

            if goal_reached:
                path = self.path_recons(start, goal, self.parent_node)
                return (path,self.opened_nodes)
            

        return ([],self.opened_nodes)
            
    
    def limited_depth_search(self, graph: AdjacencyList, start_node: X, goal: X, depth: int):
        if start_node in self.pri_que:
            self.pri_que.remove(start_node)
        self.opened_nodes.append(start_node)
        if depth == 0:
            if start_node == goal:
                return True
            else:
                return False

        adj_nodes = graph[start_node]
        adj_nodes = list(adj_nodes)
        adj_nodes.sort()
        
        trunc_adj_nodes = []
        for nodes in adj_nodes:
            if nodes not in self.opened_nodes and nodes not in self.pri_que:
                trunc_adj_nodes.append(nodes)
                self.pri_que.append(nodes)

        for node in trunc_adj_nodes:
            self.parent_node[node] = start_node
            
            goal_reached = self.limited_depth_search(graph, node, goal, depth-1)
            
            if goal_reached:
                return True