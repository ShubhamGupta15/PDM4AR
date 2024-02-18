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

# best score so far Astar with grid size = 50 and buffer size = 1.3


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
        # print("starting point:", start, "goal point:", goal)

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

    # *************************
    # ****HELPER FUNCTIONS****
    # *************************

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


class AgentOccupancyGrid:
    def __init__(self, static_obstacles, grid_size, sg_radius):
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = []
        self.grid_size = grid_size
        self.sg_radius = sg_radius
        self.min_x, self.min_y = np.inf, np.inf
        self.max_x, self.max_y = -np.inf, -np.inf
        self.res_x, self.res_y = 0, 0
        self.grid = self.calculate_occupancy_grid()

    def calculate_occupancy_grid(self):
        # how can i do it from -11 to 11
        self.min_x = -11
        self.max_x = 11
        self.min_y = -11
        self.max_y = 11

        self.res_x = np.float64(self.max_x - self.min_x) / self.grid_size
        self.res_y = np.float64(self.max_y - self.min_y) / self.grid_size

        grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.mark_obstacles(grid)
        # self.plot_grid(grid)
        return grid

    def mark_obstacles(self, grid):
        for obstacle in self.static_obstacles:
            obstacle = obstacle.buffer(1)
            minx, miny, maxx, maxy = obstacle.bounds
            x_range_start = max(0, int((minx - self.min_x) / self.res_x))
            x_range_end = min(self.grid_size, int((maxx - self.min_x) / self.res_x))
            y_range_start = max(0, int((miny - self.min_y) / self.res_y))
            y_range_end = min(self.grid_size, int((maxy - self.min_y) / self.res_y))

            for x in range(x_range_start, x_range_end):
                for y in range(y_range_start, y_range_end):
                    cell = Point(
                        x * self.res_x + self.min_x, y * self.res_y + self.min_y
                    )
                    if obstacle.intersects(cell):
                        grid[x, y] = True

    def map_to_index(self, x, y):
        # function to map from x,y to i,j
        return int((x - self.min_x) / self.res_x), int((y - self.min_y) / self.res_y)

    def index_to_map(self, i, j):
        # function to map from i,j to x,y
        return i * self.res_x + self.min_x, j * self.res_y + self.min_y

    def mark_one_obstacle(self, grid, obstacle, flag=True):
        # Mark one obstacle in the grid
        minx, miny, maxx, maxy = obstacle.bounds
        x_range_start = max(0, int((minx - self.min_x) / self.res_x))
        x_range_end = min(self.grid_size, int((maxx - self.min_x) / self.res_x))
        y_range_start = max(0, int((miny - self.min_y) / self.res_y))
        y_range_end = min(self.grid_size, int((maxy - self.min_y) / self.res_y))

        for x in range(x_range_start, x_range_end):
            for y in range(y_range_start, y_range_end):
                cell = Point(x * self.res_x + self.min_x, y * self.res_y + self.min_y)
                if obstacle.intersects(cell):
                    grid[x, y] = flag

    def plot_grid(self, grid):
        # Plot the occupancy grid
        fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(grid, cmap=plt.cm.Dark2)
        ax.imshow(grid, cmap=plt.cm.gray)
        plt.close()
        plt.savefig("grid.png")
