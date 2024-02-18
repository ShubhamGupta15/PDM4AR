from queue import PriorityQueue
import numpy as np
import heapq
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import LineString
import shapely

MANHATTAN_HEURITSTIC = "manhattan"
EUCLIDEAN_HEURISTIC = "euclidean"

#best score so far Astar with grid size = 50 and buffer size = 1.3


class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_node(self, node):
        if node not in self.adj_list:
            self.adj_list[node] = []

    def add_edge(self, node1, node2):
        if node1 not in self.adj_list:
            self.adj_list[node1] = []
        if node2 not in self.adj_list:
            self.adj_list[node2] = []
        self.adj_list[node1].append(node2)
        self.adj_list[node2].append(node1)

    def get_weight(self, node1, node2):
        return self._euclidean_distance(node1, node2)

    def _euclidean_distance(self, node1, node2):
        # Euclidean distance between two nodes
        return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


class AgentPathPlanner:
    def __init__(self, algorithm="A*", grid_size=None, grid=None, obstacles=None):
        self.algorithm = algorithm
        self.grid_size = grid_size
        self.grid_map = grid
        self.heuristic = MANHATTAN_HEURITSTIC
        self.obstacles = obstacles

    # *************************
    # ****FIND PATH****
    # *************************
    def find_path(self, start, goal, grid=None):
        if self.algorithm == "A*":
            return self._a_star_search(start, goal, grid)

        elif self.algorithm == "A*+VG":
            # get all the vertices of all the obstacles
            vertices = []
            for obstacle in self.obstacles:
                obstacle = obstacle.shape.buffer(5)
                if isinstance(obstacle, Polygon):
                    vertices.extend(list(obstacle.exterior.coords[:-1]))
                elif isinstance(obstacle, Circle):
                    x, y = obstacle.centroid.coords[0]
                    vertices.append((x, y))

            # add the start and goal points
            vertices.append(start)
            vertices.append(goal)
            vertices = [self.grid_map.map_to_index(vertex) for vertex in vertices]
            # add also 20 random points
            for i in range(100):
                vertices.append(
                    (
                        np.random.randint(0, self.grid_size),
                        np.random.randint(0, self.grid_size),
                    )
                )
            # build the visibility graph
            graph = self._build_visibility_graph(self.obstacles, vertices)
            # find the path using A* on the visibility graph
            return self.a_star_search_visibility_graph(graph, start, goal)

        elif self.algorithm == "PRM":
            prm_graph = self._build_prm_graph(
                self.grid_map, 1000, start, goal
            )  # Choose an appropriate number of samples
            return self.a_star_search_visibility_graph(prm_graph, start, goal)

        # Other algorithms can be added here
        else:
            raise ValueError("Unsupported algorithm")

    # *************************
    # ****HEURISTIC****
    # *************************
    def _manhattan_distance(self, node1, node2):
        # Manhattan distance between two nodes
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

    def _euclidean_distance(self, node1, node2):
        # Euclidean distance between two nodes
        return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

    def _heuristic(self, node1, node2):
        # Heuristic function to estimate the distance between two nodes
        if self.heuristic == MANHATTAN_HEURITSTIC:
            return self._manhattan_distance(node1, node2)
        elif self.heuristic == EUCLIDEAN_HEURISTIC:
            return self._euclidean_distance(node1, node2)
        else:
            raise ValueError("Unsupported heuristic")

    # *************************
    # ****PLANNING ALGORITHMS****
    # *************************
    def _a_star_search(self, start, goal, grid):
        # graph = self.generate_graph()
        # start = self.grid_to_graph_node(start)
        # goal = self.grid_to_graph_node(goal)
        # todo
        # print the starting point
        if grid:
            self.grid_map = grid
        print("starting point:", start, "goal point:", goal)

        if start == goal:
            return [start]

        queue = [(0, start)]
        cost_to_reach = {
            node: float("inf") for node in self._get_neighbors(start[0], start[1])
        }
        cost_to_reach[start] = 0
        parent = {}
        cost_tot = {
            node: float("inf") for node in self._get_neighbors(start[0], start[1])
        }
        cost_tot[start] = self._heuristic(start, goal)
        heuristic_cache = {}
        queue_nodes = set([start])

        while queue:
            _, current = heapq.heappop(queue)
            queue_nodes.remove(current)
            # print(current)

            if current == goal:
                # reconstruct path
                return self._reconstruct_path(parent, start, goal)

            for neighbor in self._get_neighbors(current[0], current[1]):
                new_cost_to_reach = cost_to_reach[current] + 1
                if neighbor not in cost_to_reach:
                    cost_to_reach[neighbor] = float("inf")
                if new_cost_to_reach < cost_to_reach[neighbor]:
                    parent[neighbor] = current
                    cost_to_reach[neighbor] = new_cost_to_reach
                    if (neighbor, goal) not in heuristic_cache:
                        heur = self._heuristic(neighbor, goal)
                        cost_tot[neighbor] = cost_to_reach[neighbor] + heur
                        heuristic_cache[(neighbor, goal)] = heur
                    else:
                        cost_tot[neighbor] = (
                            cost_to_reach[neighbor] + heuristic_cache[(neighbor, goal)]
                        )
                    if neighbor in queue_nodes:
                        neighbour_queue_idx = 0
                        for idx, cost_node_pair in enumerate(queue):
                            if cost_node_pair[1] == neighbor:
                                neighbour_queue_idx = idx
                                break
                        queue[neighbour_queue_idx] = (cost_tot[neighbor], neighbor)
                        heapq.heapify(queue)
                    else:
                        heapq.heappush(queue, (cost_tot[neighbor], neighbor))
                        queue_nodes.add(neighbor)
        return []

    def a_star_search_visibility_graph(self, graph, start, goal):
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {vertex: float("inf") for vertex in graph.adj_list}
        g_score[start] = 0
        f_score = {vertex: float("inf") for vertex in graph.adj_list}
        f_score[start] = self._heuristic(start, goal)

        while not open_set.empty():
            current = open_set.get()[1]
            print("Current node:", current)
            if current not in graph.adj_list:
                print("Current node not in graph.adj_list")
                continue

            if current == goal:
                return self._reconstruct_path(came_from, start, goal)

            for neighbor in graph.adj_list[current]:
                tentative_g_score = g_score[current] + graph.get_weight(
                    current, neighbor
                )

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(
                        neighbor, goal
                    )
                    open_set.put((f_score[neighbor], neighbor))

        return []

    # *************************
    # ****HELPER FUNCTIONS****
    # *************************

    def _generate_graph(self):
        # Generate a graph from the grid
        graph = Graph()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if not self.grid_map.grid[i, j]:
                    graph.add_node((i, j))
                    for neighbor in self._get_neighbors(i, j):
                        graph.add_edge((i, j), neighbor)
        return graph

    def _is_visible(self, p1, p2, obstacles):
        line = LineString([p1, p2])
        for obstacle in obstacles:
            # transform obstacles in the grid size

            if shapely.intersects(line, obstacle.shape):
                return False
        return True

    def _build_visibility_graph(self, obstacles, vertices):
        graph = Graph()
        for v1 in vertices:
            for v2 in vertices:
                if v1 != v2 and self._is_visible(v1, v2, obstacles):
                    graph.add_edge(v1, v2)
        return graph

    def _get_neighbors(self, x, y):
        # Get all adjacent cells that are not obstacles
        neighbors = []
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
            (1, 1),
        ]:  # 8-connected grid (also diagonal movements)
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < self.grid_size
                and 0 <= ny < self.grid_size
                and not self.grid_map.grid[nx, ny]
            ):
                neighbors.append((nx, ny))
        return neighbors

    def _reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def _random_free_space_samples(self, agent_grid, num_samples):
        free_space_samples = []
        while len(free_space_samples) < num_samples:
            x = np.random.randint(0, agent_grid.grid_size)
            y = np.random.randint(0, agent_grid.grid_size)
            if not agent_grid.grid[x, y]:
                free_space_samples.append((x, y))
        return free_space_samples

    def _is_path_free(self, point1, point2, agent_grid):
        line = LineString([point1, point2])
        for x in np.arange(
            min(point1[0], point2[0]),
            max(point1[0], point2[0]),
            agent_grid.res_x,
        ):
            for y in np.arange(
                min(point1[1], point2[1]),
                max(point1[1], point2[1]),
                agent_grid.res_y,
            ):
                if agent_grid.grid[int(x), int(y)]:
                    return False
        return True

    def _is_path_free2(self, point1, point2, agent_grid):
        point1 = agent_grid.index_to_map(point1[0], point1[1])
        point2 = agent_grid.index_to_map(point2[0], point2[1])
        line = LineString([point1, point2])
        for obstacle in self.obstacles:
            # transform obstacles in the grid size
            if shapely.intersects(
                line, obstacle.shape.buffer(agent_grid.sg_radius * 1.3)
            ):
                return False
        return True

    def _build_prm_graph(self, agent_grid, num_samples, start, goal):
        vertices = self._random_free_space_samples(agent_grid, num_samples)
        vertices.append(start)
        vertices.append(goal)

        prm_graph = Graph()
        for vertex in vertices:
            prm_graph.add_node(vertex)

        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if self._is_path_free2(vertices[i], vertices[j], agent_grid):
                    prm_graph.add_edge(vertices[i], vertices[j])

        return prm_graph
