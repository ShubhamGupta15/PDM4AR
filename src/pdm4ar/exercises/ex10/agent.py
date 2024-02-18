from dataclasses import dataclass
from typing import Sequence
from shapely import Polygon, Point, LineString

from dg_commons import PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands, DiffDriveState
from dg_commons.sim.models.diff_drive_structures import (
    DiffDriveGeometry,
    DiffDriveParameters,
)
from dg_commons.sim.models.obstacles import StaticObstacle

import numpy as np
import copy

from pdm4ar.exercises.ex10.path_planning import AgentPathPlanner
from pdm4ar.exercises.ex10.controller import Controller

@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    name: PlayerName
    goal: PlanningGoal
    planner: AgentPathPlanner
    sub_goal = (0, 0)
    
    static_obstacles: Sequence[StaticObstacle]
    sg: DiffDriveGeometry
    sp: DiffDriveParameters
    state: DiffDriveState
    start_state: DiffDriveState = None
    other_robot_locations = {}
    observed_lidar_scan = None
    robots_in_view = set()
    controller: Controller
    goal_centroid = None

    shortest_path = []
    is_replan_count = 0
    is_replan_threhold = 5
    safety_backoff_dist = 3
    
    activate_safe_point = False

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.current_state = None
        self.start_point = None
        self.goal_point = None
        self.grid_size = 100
        self.grid_map = None
        self.local_grid_map = None
        self.current_position = None
        self.previous_position = None
        self.current_time = None
        self.previous_time = None
        self.current_orientation = None
        self.previous_orientation = None
        self.robots = {}

    # ********************
    # ****UPDATE METHODS****
    # ********************

    def update_position(self, new_position, new_orientation, new_time):
        self.previous_position = self.current_position
        self.current_position = new_position
        self.previous_time = self.current_time
        self.current_time = new_time
        self.previous_orientation = self.current_orientation
        self.current_orientation = new_orientation

    def calculate_velocity(self):
        if self.previous_position is None or self.previous_time is None:
            return np.zeros(2)  # We can't calculate velocity without a previous position and time

        delta_position = self.current_position - self.previous_position
        delta_time = self.current_time - self.previous_time

        velocity = delta_position / float(delta_time)
        return velocity

    def calculate_angular_velocity(self):
        if self.previous_orientation is None or self.previous_time is None:
            return None  # We can't calculate angular velocity without a previous orientation and time

        delta_orientation = self.current_orientation - self.previous_orientation
        delta_time = self.current_time - self.previous_time

        omega = delta_orientation / float(delta_time)
        return omega

    def update_robot(self, robot_id, position, orientation, time):
        if robot_id not in self.robots:
            self.robots[robot_id] = {
                "current_position": None,
                "previous_position": None,
                "current_orientation": None,
                "previous_orientation": None,
                "current_time": None,
                "previous_time": None,
                "velocity": np.zeros(2),
                "angular_velocity": 0,
            }

        robot = self.robots[robot_id]

        # Update position and calculate velocity
        robot["previous_position"] = robot["current_position"]
        robot["current_position"] = position
        if (
            robot["previous_position"] is not None
            and robot["previous_time"] is not None
        ):
            delta_position = robot["current_position"] - robot["previous_position"]
            delta_time = time - robot["previous_time"]
            robot["velocity"] = delta_position / float(delta_time)

        # Update orientation and calculate angular velocity
        robot["previous_orientation"] = robot["current_orientation"]
        robot["current_orientation"] = orientation
        if (
            robot["previous_orientation"] is not None
            and robot["previous_time"] is not None
        ):
            delta_orientation = (
                robot["current_orientation"] - robot["previous_orientation"]
            )
            robot["angular_velocity"] = delta_orientation / float(delta_time)

        # Update time
        robot["previous_time"] = robot["current_time"]
        robot["current_time"] = time

    # ********************
    # ****MAIN METHODS****
    # ********************

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator at the beginning of each episode."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.static_obstacles = list(init_obs.dg_scenario.static_obstacles)
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params
        
        self.controller = Controller(self.name, self.sg, self.sp, self.goal, self.static_obstacles, algorithm="pure_pursuit")

        self.grid_size = 50

        self.grid_map = AgentOccupancyGrid(
            self.static_obstacles, self.grid_size, self.sg.radius
        )
        self.planner = AgentPathPlanner(
            algorithm="A*",
            grid_size=self.grid_size,
            grid=self.grid_map,
            obstacles=self.static_obstacles,
        )


    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: Diffsim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        # save the robot state
        self.state: DiffDriveState = sim_obs.players[self.name].state
        self.update_position(
            np.array([self.state.x, self.state.y]),
            self.state.psi,
            sim_obs.time,
        )
        self.controller.set_state(self.state)
        
        if self.start_state is None:
            # do A* to find path to goal
            self.initilize_path_planning()

        # find the other visible robots
        self.robots_in_view = set()
        self.observed_lidar_scan = sim_obs.players[self.name].occupancy

        curr_vel = self.calculate_velocity()
        self.controller.set_curr_measured_vel(curr_vel)
        curr_pos = np.array([self.state.x, self.state.y])
        
        marked_dynamic_obstacles = []
        is_replan = False
        self.local_grid_map = copy.deepcopy(self.grid_map)

        for obs in sim_obs.players:
            if obs != self.name:
                self.robots_in_view.add(obs)
                self.other_robot_locations[obs] = sim_obs.players[obs].state
                self.update_robot(obs, np.array([sim_obs.players[obs].state.x, sim_obs.players[obs].state.y]), sim_obs.players[obs].state.psi, sim_obs.time)
                
                # calculate distance to other robots
                distance = np.linalg.norm(
                    curr_pos - self.robots[obs]["current_position"]
                )
                if distance < 6 * self.sg.radius:
                    # add robot as obstacle in the grid map
                    obstacle = Point(self.robots[obs]["current_position"]).buffer(
                        self.sg.radius * 2
                    )
                    self.local_grid_map.mark_one_obstacle(
                        self.local_grid_map.grid, obstacle, flag=1
                    )
                    marked_dynamic_obstacles.append(obstacle)
                        
                    if self.name < obs and self.is_robot_along_path(self.robots[obs]["current_position"]):
                        # replan
                        is_replan = True
        
        # enable replanning
        if is_replan or self.activate_safe_point:
            # replan
            if (self.is_replan_count < self.is_replan_threhold):
                self.is_replan_count += 1
            else: 
                self.is_replan_count = 0
                print("replaning for agent", self.name)
                self.replan()
        else:
            self.is_replan_count = 0
            
        self.controller.set_grid_map(self.local_grid_map)
        
        self.controller.set_other_robot_info(self.robots_in_view, self.robots)

        drive_commands = self.controller.execute_action(self.state)
            
        return drive_commands

    # ********************
    # **INIT PLANNING*****
    # ********************

    def initilize_path_planning(self):
        self.start_state = self.state
        shapely_polygon = Polygon(
            self.goal.get_plottable_geometry().exterior.coords
        )
        self.goal_centroid = shapely_polygon.centroid

        self.shortest_path = self.planner.find_path(
            self.grid_map.map_to_index(self.start_state.x, self.start_state.y),
            self.grid_map.map_to_index(self.goal_centroid.x, self.goal_centroid.y),
        )

        print("path in grid map:", self.shortest_path)
        
        # convert from grid to map coordinates
        self.shortest_path = [
            (
                x * self.grid_map.res_x + self.grid_map.min_x,
                y * self.grid_map.res_y + self.grid_map.min_y,
            )
            for (x, y) in self.shortest_path
        ]
        
        if self.shortest_path:
            self.shortest_path.append([self.goal_centroid.x, self.goal_centroid.y])
        
        self.controller.set_trajectory(self.shortest_path)

        print("starting the agent", self.name)
        
    def replan(self):
        shapely_polygon = Polygon(self.goal.get_plottable_geometry().exterior.coords)

        self.goal_centroid = shapely_polygon.centroid

        self.shortest_path = self.planner.find_path(
            self.local_grid_map.map_to_index(self.state.x, self.state.y),
            self.local_grid_map.map_to_index(self.goal_centroid.x, self.goal_centroid.y), self.local_grid_map) 
        
        # convert from grid to map coordinates
        self.shortest_path = [
            (
                x * self.local_grid_map.res_x + self.local_grid_map.min_x,
                y * self.local_grid_map.res_y + self.local_grid_map.min_y,
            )
            for (x, y) in self.shortest_path
        ]
        
        if self.shortest_path:
            self.shortest_path.append([self.goal_centroid.x, self.goal_centroid.y])
        
        if not self.shortest_path:
            print("no path found for", self.name)
            self.find_safe_point()
        else:
            self.activate_safe_point = False
            self.controller.set_trajectory(self.shortest_path)
            print("replaned for agent", self.name)
            
    def find_safe_point(self):
        # search for a safe point in the local grid map
        # use the current velocity direction to find a safe point
        self.activate_safe_point = True
        print("finding a safe spot for", self.name)
        curr_vel = self.controller.get_selected_velocity()
        curr_vel_norm = curr_vel / np.linalg.norm(curr_vel)
        safe_point = np.array([self.state.x, self.state.y]) + curr_vel_norm * self.safety_backoff_dist
        self.shortest_path = self.planner.find_path(
        self.local_grid_map.map_to_index(self.state.x, self.state.y),
        self.local_grid_map.map_to_index(safe_point[0], safe_point[1]), self.local_grid_map) 
        
        # convert from grid to map coordinates
        self.shortest_path = [
            (
                x * self.local_grid_map.res_x + self.local_grid_map.min_x,
                y * self.local_grid_map.res_y + self.local_grid_map.min_y,
            )
            for (x, y) in self.shortest_path
        ]
        
        if self.shortest_path:
            self.shortest_path.append(safe_point)
        
        if not self.shortest_path:
            print("no safe point found for", self.name)
            return
        else:
            self.controller.set_trajectory(self.shortest_path)
            print("safe point found for", self.name)

    def is_robot_along_path(self, position):
        # Check if the position is along the path
        # Compute the distance from the position to each point in the path
        obstacle = Point(position).buffer(self.sg.radius * 2)

        line = LineString(self.shortest_path)
        
        if line.intersects(obstacle):
            return True

        return False


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
        self.buffer_size = 1.3

    def calculate_occupancy_grid(self):
        for obstacle in self.static_obstacles:
            ob_minx, ob_miny, ob_maxx, ob_maxy = obstacle.shape.bounds
            self.min_x = min(self.min_x, ob_minx)
            self.max_x = max(self.max_x, ob_maxx)
            self.min_y = min(self.min_y, ob_miny)
            self.max_y = max(self.max_y, ob_maxy)
        
        ## DO WE NEED TO ADD THE ROBOT RADIUS TO THE BOUNDARIES?
        self.min_x -= self.sg_radius
        self.max_x += self.sg_radius
        self.min_y -= self.sg_radius
        self.max_y += self.sg_radius
        
        self.res_x = np.float64(self.max_x - self.min_x) / self.grid_size
        self.res_y = np.float64(self.max_y - self.min_y) / self.grid_size

        grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.mark_obstacles(grid)
        return grid

    def mark_obstacles(self, grid):
        for obstacle in self.static_obstacles:
            obstacle = obstacle.shape.buffer(self.sg_radius * 1.4)
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
