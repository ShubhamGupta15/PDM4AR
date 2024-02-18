from dataclasses import dataclass
from typing import Sequence
import math
from collections import deque

import shapely
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

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

# implement algorithms: pure_pursuit, stanley, mpc
# ideas: change lookahead distance based on speed

class Controller:
    
    name: PlayerName
    sg: DiffDriveGeometry
    sp: DiffDriveParameters
    state: DiffDriveState
    wheel_base: float
    wheel_radius: float
    radius: float
    trajectory: LineString
    algorithm: str
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    is_plotting: bool = False
    
    move_base_speed = 1
    move_base_max_speed = 2
    omega_thresholds = [-10, 10]
    
    robot_modes = {"idle": 0, "move": 1, "turn": 2, "avoid": 3, "wait": 4}
    curr_mode: int = 0
    plot_ax = None
    robots_in_view = set()
    other_robot_locations = {}
    sub_goal = None
    iter_no = 0
    omega_l_history = []
    omega_r_history = []
    curr_measured_vel: np.ndarray = None
    selected_velocity: np.ndarray = None
    robots = {}
    constraints = []
    lines_Amat: np.ndarray = None
    lines_bvec: np.ndarray = None
    vo_obstacles = []
    vel_before_constraints = None
    grid_map = None
    
    # TODO: parameters to tune
    poly_obstacle_enlarge_factor = 1.2
    circle_obstacle_enlarge_factor = 1.4
    lookahead_distance = 1
    vel_multiplier = 2
    
    min_max_coords = []
    
    def __init__(self, name, sg, sp, goal, static_obstacles, algorithm="pure_pursuit"):
        self.name = name
        self.sg = sg
        self.sp = sp
        self.wheel_base = self.sg.wheelbase
        self.wheel_radius = self.sg.wheelradius
        self.radius = self.sg.radius
        self.algorithm = algorithm
        self.goal = goal
        self.static_obstacles = [obstacle for obstacle in static_obstacles if isinstance(obstacle.shape, shapely.Polygon)]
        
        self.plot_init()

    def execute_action(self, state):
        if not self.trajectory:
            return DiffDriveCommands(omega_l=0, omega_r=0)
        
        self.state = state
        
        # delay robot 1 for a bit
        if (self.name=="PDM4AR_1"):
            if self.iter_no < 0:
                drive_commands = DiffDriveCommands(omega_l=0, omega_r=0)
            else:
                drive_commands = self.pure_pursuit_control()
        else:     
            drive_commands = self.pure_pursuit_control()
        
        # plot every 20 iterations
        if self.iter_no % 30 == 0:
            self.plot(self.iter_no)
            # self.plot_omega(self.iter_no)
            pass
        self.iter_no += 1
        
        # add to omega history
        self.omega_l_history.append(drive_commands.omega_l)
        self.omega_r_history.append(drive_commands.omega_r)
            
        return drive_commands
    
    def set_other_robot_info(self, robots_in_view, robots):
        self.robots_in_view = robots_in_view
        self.robots = robots

    def set_state(self, state):
        self.state = state
    
    def get_curr_mode_txt(self):
        return list(self.robot_modes.keys())[self.curr_mode]
    
    def set_trajectory(self, trajectory):
        # convert the list to shapely linestrings
        self.trajectory = LineString(trajectory)
        
        self.plot("trajectory")
    
    def compute_desired_velocity(self, current_pos, goal_pos):
        disp_vec = (goal_pos - current_pos)[:2]
        norm = np.linalg.norm(disp_vec)
        if norm < self.radius / 5:
            return np.zeros(2)
        disp_vec = disp_vec / norm
        np.shape(disp_vec)
        desired_vel = self.move_base_speed * disp_vec
        return desired_vel
        
    def find_lookahead_point(self, reverse=False):
        # create a circle around the robot
        lookahead_circle = Point(self.state.x, self.state.y).buffer(self.lookahead_distance)
        
        # find interest point between the lookahead circle and the trajectory
        lookahead_points = lookahead_circle.intersection(self.trajectory)
        
        # fin the point that has least angle difference with the robot heading
        if (isinstance(lookahead_points, shapely.LineString)):
            if reverse:
                min_arc_length = 1000
            else:
                max_arc_length = -1
            for point in lookahead_points.coords:
                # use arclength
                arc_length = self.trajectory.project(Point(point))
                if not reverse and arc_length > max_arc_length:
                    max_arc_length = arc_length
                    self.sub_goal = (point[0], point[1])
                elif reverse and arc_length < min_arc_length:
                    min_arc_length = arc_length
                    self.sub_goal = (point[0], point[1])
    
    def velocity_to_omega(self, v_linear, v_angular)->tuple:
        omega_l = (v_linear - (self.wheel_base/2)*v_angular) / self.wheel_radius
        omega_r = (v_linear + (self.wheel_base/2)*v_angular) / self.wheel_radius
        return omega_l, omega_r

    def set_curr_measured_vel(self, measured_velocity):
        self.curr_measured_vel = measured_velocity

    def pure_pursuit_control(self):
        # get all the nearby static obstacles
        obstacles = []
        velocities = []
        is_dynamic_obstacles_nearby = False
        activate_slowdown = False
        is_reverse_lookahead = False
        
        # consider static obstacles
        for obstacle in self.static_obstacles:
            if obstacle.shape.distance(Point(self.state.x, self.state.y)) < 2*self.radius:
                obstacles.append((obstacle.shape.buffer(self.poly_obstacle_enlarge_factor*self.radius),'static'))
                velocities.append(np.array([0, 0]))
        
        # get all the nearby robots
        for robot_name in self.robots_in_view: 
            # neglect the robots with higher names
            robot_state = self.robots[robot_name]["current_position"]
            dist_between_robots = np.linalg.norm(robot_state-np.array([self.state.x,self.state.y]))
            if dist_between_robots < 8*self.radius:
                if self.name < robot_name:
                    obstacle = Point(robot_state).buffer((1+self.circle_obstacle_enlarge_factor)*self.radius)
                    obstacles.append((obstacle,'dynamic,'+robot_name))
                    velocities.append(self.robots[robot_name]["velocity"])
                    
                is_dynamic_obstacles_nearby = True # to activate using static collision avoidance
                
                if dist_between_robots < 4*self.radius:
                    activate_slowdown = True
                    
                # enable reverse lookahead for the robot
                # if self.name < robot_name and self.trajectory.intersects(obstacle):
                #     is_reverse_lookahead = True
                #     print("reverse lookahead activated for robot:", self.name, "with robot:", robot_name)
                    
        if activate_slowdown:
            self.lookahead_distance = 1
            self.move_base_speed = 0.5
        else:
            self.lookahead_distance = 1
            self.move_base_speed = 1
                    
        self.find_lookahead_point(is_reverse_lookahead)
        
        if not self.sub_goal:
            return DiffDriveCommands(omega_l=0, omega_r=0)
        
        v_vec = self.compute_desired_velocity(np.array([self.state.x, self.state.y]), np.array(self.sub_goal))
        self.vel_before_constraints = v_vec
        
        # compute the velocity based on the obstacles
        if is_dynamic_obstacles_nearby:
            v_vec = self.compute_velocity2(obstacles, velocities, v_vec)
        
        self.selected_velocity = v_vec
        
        # get the angle difference
        angle_diff = np.fmod(np.arctan2(v_vec[1], v_vec[0]) - self.state.psi, 2*np.pi)
        
        # control v_tot based on angle diff
        cos_angle_diff = np.cos(angle_diff)
        
        v_tot = np.linalg.norm(v_vec)
        v_linear = v_tot * cos_angle_diff
        
        v_perpendicular = v_tot * np.sin(angle_diff)
        
        v_angular = np.sign(v_perpendicular)*np.sqrt(np.abs(v_perpendicular)/self.lookahead_distance)
        
        omega_l, omega_r = self.velocity_to_omega(v_linear, v_angular)
        # limit the omega values
        omega_l = np.clip(omega_l, self.omega_thresholds[0], self.omega_thresholds[1])
        omega_r = np.clip(omega_r, self.omega_thresholds[0], self.omega_thresholds[1])
        
        return DiffDriveCommands(omega_l=omega_l, omega_r=omega_r)

    def compute_velocity3(self, obstacles, velocities, v_desired):
        # Robot's current position
        pA = np.array([self.state.x, self.state.y])
        # Parameters for velocity computation
        vA = self.curr_measured_vel
        
        # Compute the constraints
        number_of_obstacles = len(obstacles)
        Amat = np.empty((number_of_obstacles * 2, 2))
        bvec = np.empty((number_of_obstacles * 2))

        # Define VO triangles for each obstacle
        for i, obstacle in enumerate(obstacles):
            obstacle_enlarged = obstacle[0]
            obstacle_type = obstacle[1]
            
            pB = np.array(obstacle_enlarged.centroid.coords[0])
                
            dispBA = pB - pA
            distBA = np.linalg.norm(dispBA)
            thetaBA = np.arctan2(dispBA[1], dispBA[0])
            vB = velocities[i]
            
            # vel_shift = pA + (self.vel_multiplier*vB + vA)/2
            vel_shift = pA + self.vel_multiplier*vB
                
            if obstacle_type == 'dynamic':
                # check if pA is inside the obstacle
                if obstacle_enlarged.contains(Point(pA)):
                    phi_left = np.fmod(thetaBA + np.pi/2, 2 * np.pi)
                    phi_right = np.fmod(thetaBA - np.pi/2, 2 * np.pi)
                else:
                    radius = (1+self.circle_obstacle_enlarge_factor)*self.radius

                    delta_angle = np.abs(np.arcsin(np.clip(radius/distBA, -1, 1)))
                    
                    phi_left = np.fmod(thetaBA + delta_angle, 2 * np.pi)
                    phi_right = np.fmod(thetaBA - delta_angle, 2 * np.pi)
            else:
                # check if pA is inside the obstacle
                if obstacle_enlarged.contains(Point(pA)):
                    phi_left = np.fmod(thetaBA + np.pi/2, 2 * np.pi)
                    phi_right = np.fmod(thetaBA - np.pi/2, 2 * np.pi)
                else:
                    max_angle = - np.pi
                    min_angle = 2 * np.pi
                    
                    self.min_max_coords = []
                    max_coord = None
                    min_coord = None
                    
                    if thetaBA < 0:
                        thetaBA_n = thetaBA + 2*np.pi
                    else:
                        thetaBA_n = thetaBA
                            
                    for point in obstacle_enlarged.exterior.coords:  
                        
                        if thetaBA_n > np.pi/2 and thetaBA_n < 3*np.pi/2:
                            angle = np.arctan2(point[1] - pA[1], point[0] - pA[0])
                            if angle < 0:
                                angle += 2*np.pi
                        else:
                            angle = np.fmod(
                                np.arctan2(point[1] - pA[1], point[0] - pA[0]) - thetaBA, 2 * np.pi
                            )
                            
                        max_angle = max(max_angle, angle)
                        min_angle = min(min_angle, angle)
                            
                        if (angle > max_angle):
                            max_angle = angle
                            max_coord = np.array(point)
                        if (angle < min_angle):
                            min_angle = angle
                            min_coord = np.array(point)

                    if thetaBA_n > np.pi/2 and thetaBA_n < 3*np.pi/2:
                        phi_left = np.fmod(max_angle, 2 * np.pi)
                        phi_right = np.fmod(min_angle, 2 * np.pi)
                    else:
                        phi_left = np.fmod(thetaBA + max_angle, 2 * np.pi)
                        phi_right = np.fmod(thetaBA + min_angle, 2 * np.pi)
                        
                    if max_coord is not None and min_coord is not None:
                        self.min_max_coords.append([max_coord, min_coord])
                    
            Atemp, btemp = self.create_constraints(vel_shift, phi_left, "left")
            Amat[i*2, :] = Atemp
            bvec[i*2] = btemp
            Atemp, btemp = self.create_constraints(vel_shift, phi_right, "right")
            Amat[i*2 + 1, :] = Atemp
            bvec[i*2 + 1] = btemp  

        # Create search-space
        th = np.linspace(0, 2*np.pi, 20)
        vel = np.linspace(0, self.move_base_speed, 5)

        vv, thth = np.meshgrid(vel, th)

        vx_sample = (vv * np.cos(thth)).flatten()
        vy_sample = (vv * np.sin(thth)).flatten()

        v_sample = np.stack((vx_sample, vy_sample))

        v_satisfying_constraints = self.check_constraints(v_sample, Amat, bvec)

        # Objective function
        size = np.shape(v_satisfying_constraints)
        if (len(size) == 1):
            cmd_vel = v_desired
        else:
            size = size[1]
            diffs = v_satisfying_constraints - \
                ((v_desired).reshape(2, 1) @ np.ones(size).reshape(1, size))
            norm = np.linalg.norm(diffs, axis=0)
            min_index = np.where(norm == np.amin(norm))[0][0]
            cmd_vel = (v_satisfying_constraints[:, min_index])
        
        return cmd_vel
    
    def get_selected_velocity(self):
        return self.selected_velocity

    def check_constraints(self,v_sample, Amat, bvec):
        length = np.shape(bvec)[0]

        for i in range(int(length/2)):
            v_sample = self.check_inside(v_sample, Amat[2*i:2*i+2, :], bvec[2*i:2*i+2])

        return v_sample

    def check_inside(self,v, Amat, bvec):
        v_out = []
        for i in range(np.shape(v)[1]):
            if not ((Amat @ v[:, i] < bvec).all()):
                v_out.append(v[:, i])
        return np.array(v_out).T

    def create_constraints(self,translation, angle, side):
        # create line
        origin = np.array([0, 0, 1])
        point = np.array([np.cos(angle), np.sin(angle)])
        line = np.cross(origin, point)
        line = self.translate_line(line, translation)

        if side == "left":
            line *= -1

        A = line[:2]
        b = -line[2]

        return A, b

    def translate_line(self,line, translation):
        matrix = np.eye(3)
        matrix[2, :2] = -translation[:2]
        return matrix @ line
    
    def compute_velocity2(self, obstacles, velocities, v_desired):
        # Robot's current position
        pA = np.array([self.state.x, self.state.y])
        # Parameters for velocity computation
        self.vo_obstacles = []
        vA = self.curr_measured_vel

        # Define VO triangles for each obstacle
        for i, obstacle in enumerate(obstacles):
            obstacle_enlarged = obstacle[0]
            obstacle_type = obstacle[1]
            
            pB = np.array(obstacle_enlarged.centroid.coords[0])
                
            dispBA = pB - pA
            distBA = np.linalg.norm(dispBA)
            thetaBA = np.arctan2(dispBA[1], dispBA[0])
            vB = velocities[i]
            
            # vel_shift = pA + (self.vel_multiplier*vB + vA)/2
            # vel_shift = pA + self.vel_multiplier*vB
                
            if 'dynamic' in obstacle_type:
                obstacle_name = obstacle_type.split(',')[1]
                if obstacle_name in self.robots_in_view:
                    # if self.name < obstacle_name:
                    #     vel_shift = pA + self.vel_multiplier*vB
                    # else:
                    #     vel_shift = pA + (self.vel_multiplier*vB + vA)/2
                    vel_shift = pA + self.vel_multiplier*vB
                
                radius = (1+self.circle_obstacle_enlarge_factor)*self.radius
                
                # check if pA is inside the obstacle
                if obstacle_enlarged.contains(Point(pA)):
                    # create a rectangle around the robot
                    dirBA_perp = np.array([-dispBA[1], dispBA[0]])
                    dirBA_perp /= np.linalg.norm(dirBA_perp)
                    dirBA = dispBA / distBA
                    p1 = vel_shift + radius*dirBA_perp
                    p2 = vel_shift - radius*dirBA_perp
                    p3 = p1 + dirBA*10
                    p4 = p2 + dirBA*10
                    vo_obstacle = Polygon([p1, p2, p4, p3])
                else:
                    delta_angle = np.abs(np.arcsin(np.clip(radius/distBA, -1, 1)))
                    
                    phi_left = np.fmod(thetaBA + delta_angle, 2 * np.pi)
                    phi_right = np.fmod(thetaBA - delta_angle, 2 * np.pi)

                    vo_obstacle = self.define_vo_obstacle(vel_shift, phi_left, phi_right, distance=5)    
            else:
                vel_shift = pA + self.vel_multiplier*vB
                # check if pA is inside the obstacle
                if obstacle_enlarged.contains(Point(pA)):
                    # vo_obstacle = Point(self.vel_multiplier*vB + pB).buffer(distBA).envelope
                    vo_obstacle = obstacle_enlarged
                else:
                    max_angle = - np.pi
                    min_angle = 2 * np.pi
                    
                    self.min_max_coords = []
                    max_coord = None
                    min_coord = None
                    
                    if thetaBA < 0:
                        thetaBA_n = thetaBA + 2*np.pi
                    else:
                        thetaBA_n = thetaBA
                            
                    for point in obstacle_enlarged.exterior.coords:  
                        
                        if thetaBA_n > np.pi/2 and thetaBA_n < 3*np.pi/2:
                            angle = np.arctan2(point[1] - pA[1], point[0] - pA[0])
                            if angle < 0:
                                angle += 2*np.pi
                        else:
                            angle = np.fmod(
                                np.arctan2(point[1] - pA[1], point[0] - pA[0]) - thetaBA, 2 * np.pi
                            )
                            
                        max_angle = max(max_angle, angle)
                        min_angle = min(min_angle, angle)
                            
                        if (angle > max_angle):
                            max_angle = angle
                            max_coord = np.array(point)
                        if (angle < min_angle):
                            min_angle = angle
                            min_coord = np.array(point)

                    if thetaBA_n > np.pi/2 and thetaBA_n < 3*np.pi/2:
                        phi_left = np.fmod(max_angle, 2 * np.pi)
                        phi_right = np.fmod(min_angle, 2 * np.pi)
                    else:
                        phi_left = np.fmod(thetaBA + max_angle, 2 * np.pi)
                        phi_right = np.fmod(thetaBA + min_angle, 2 * np.pi)
                        
                    if max_coord is not None and min_coord is not None:
                        self.min_max_coords.append([max_coord, min_coord])

                    vo_obstacle = self.define_vo_obstacle(vel_shift, phi_left, phi_right, distance=5)
                
            self.vo_obstacles.append(vo_obstacle)

        # Initialize variables for finding the best velocity
        min_dist = np.inf
        best_velocity = v_desired
        
        v_abs_resolution = 0.2
        v_angle_resolution = np.pi/18
        # search in the magnetude and direction of the desired velocity
        for v_abs in np.arange(0, self.move_base_max_speed + v_abs_resolution, v_abs_resolution):
            for v_angle in np.arange(0, 2*np.pi, v_angle_resolution):
                velocity_vector = np.array([v_abs*np.cos(v_angle), v_abs*np.sin(v_angle)])
                is_inside_any_vo = any(
                    self.check_inside_vo(pA, velocity_vector, vo) for vo in self.vo_obstacles
                )

                if not is_inside_any_vo:
                    # Calculate the distance to the desired velocity
                    dist = np.linalg.norm(velocity_vector - v_desired)

                    # Update best velocity if this is the closest so far
                    if dist < min_dist:
                        min_dist = dist
                        best_velocity = velocity_vector

        # Return the best velocity found
        return best_velocity
    
    def define_vo_obstacle(self, robot_position, phi_left, phi_right, distance=20):
        pA = np.array(robot_position)
        pB = np.array(
            [pA[0] + distance * np.cos(phi_left), pA[1] + distance * np.sin(phi_left)]
        )
        pC = np.array(
            [pA[0] + distance * np.cos(phi_right), pA[1] + distance * np.sin(phi_right)]
        )
        phi_middle = np.fmod((phi_left + phi_right) / 2, 2 * np.pi)
        pD = np.array(
            [
                pA[0] + distance * np.cos(phi_middle),
                pA[1] + distance * np.sin(phi_middle),
            ]
        )
        return Polygon([pA, pB, pD, pC])
    
    def check_inside_vo(self, robot_pos, velocity_vector, vo_obstacle):
        # Check if the vector is inside the VO triangle (implement this check based on your VO definition)
        line_end = Point(np.array(robot_pos) + velocity_vector)

        # Check for intersection with the VO triangle
        return line_end.intersects(vo_obstacle)
    


    def plot_init(self):
        if (not self.is_plotting): 
            return
        plt.figure()
        plt.xlim(-12, 12)
        plt.ylim(-12, 12)
        plt.grid()
        plt.gca().set_aspect("equal", adjustable="box")
        self.plot_ax = plt.gca()

        # draw shapely obstacles
        for obs in self.static_obstacles:
            if isinstance(obs.shape, shapely.Polygon):
                x, y = obs.shape.exterior.xy
                # fill the polygon
                self.plot_ax.fill(x, y)
            elif isinstance(obs.shape, shapely.LinearRing):
                x, y = obs.shape.coords.xy
                self.plot_ax.plot(x, y)

        # draw goal
        goal_geom = self.goal.get_plottable_geometry()
        x, y = goal_geom.exterior.xy
        self.plot_ax.plot(x, y)

        # save the figure
        # plt.savefig(f"out/10/fig/{self.name}_plot.png")
        # plt.close()

    def plot(self, suffix=""):
        if (not self.is_plotting): 
            return
        # draw the robot as a circle
        plt.figure()
        plt.xlim(-12, 12)
        plt.ylim(-12, 12)
        plt.grid()
        plt.gca().set_aspect("equal", adjustable="box")
        self.plot_ax = plt.gca()

        # draw shapely obstacles
        for obs in self.static_obstacles:
            if isinstance(obs.shape, shapely.Polygon):
                x, y = obs.shape.buffer(self.poly_obstacle_enlarge_factor*self.radius).exterior.xy
                # x, y = obs.shape.exterior.xy
                # fill the polygon
                self.plot_ax.fill(x, y)
            elif isinstance(obs.shape, shapely.LinearRing):
                ## DOESN'T WORK
                x, y = obs.shape.coords.xy
                self.plot_ax.plot(x, y)

        # draw goal
        goal_geom = self.goal.get_plottable_geometry()
        x, y = goal_geom.exterior.xy
        self.plot_ax.plot(x, y)

        # draw robot
        circle = Circle(
            (self.state.x, self.state.y), radius=self.sg.radius, color="green"
        )
        self.plot_ax.add_patch(circle)

        # draw arrow for robot orientation
        x = self.state.x
        y = self.state.y
        dx = math.cos(self.state.psi)
        dy = math.sin(self.state.psi)
        self.plot_ax.arrow(x, y, dx, dy, width=0.1, color="green")
        
        # add label for the robot
        self.plot_ax.text(self.state.x, self.state.y, self.name, color="black", fontsize=6)

        # curr mode in text var
        curr_mode_text = self.get_curr_mode_txt()

        # plot the trajectory
        if isinstance(self.trajectory, shapely.LineString):
            x, y = self.trajectory.xy
            self.plot_ax.plot(x, y, color="green")
        
        # draw other robots
        for robot_name in self.robots_in_view:
            robot_state = self.robots[robot_name]["current_position"]
            robot_angle = self.robots[robot_name]["current_orientation"]
            circle = Circle(
                (robot_state[0], robot_state[1]), radius=self.sg.radius, color="red"
            )
            self.plot_ax.add_patch(circle)
            # add robot name as text
            self.plot_ax.text(robot_state[0], robot_state[1], robot_name, color="black", fontsize=6)
            # draw heading
            x = robot_state[0]
            y = robot_state[1]
            dx = math.cos(robot_angle)
            dy = math.sin(robot_angle)
            self.plot_ax.arrow(x, y, dx, dy, width=0.1, color="red")
        
        # draw the velocity traingles
        for triangle in self.vo_obstacles:
            # draw the polygon
            x, y = triangle.exterior.xy
            # use unique color for each triangle
            self.plot_ax.plot(x, y, color=np.random.rand(3,))
            # self.plot_ax.plot(x, y, color="brown")
            
        # plot the selected velocity vector
        if self.selected_velocity is not None:
            x, y = self.selected_velocity
            self.plot_ax.arrow(self.state.x, self.state.y, x, y, width=0.1, color="red")
        
        # plot the velocity before constraints
        if self.curr_measured_vel is not None:
            x, y = self.curr_measured_vel
            self.plot_ax.arrow(self.state.x, self.state.y, x, y, width=0.1, color="black")
            
        # plot a circle around the robot
        # self.plot_ax.add_patch(plt.Circle((self.state.x, self.state.y), 1, color="blue", fill=False))
        
        # plot coords
        # for coord in self.min_max_coords:
        #     max_coord, min_coord = coord
        #     self.plot_ax.plot(max_coord[0], max_coord[1], "x", color="red", markersize=3)
        #     self.plot_ax.plot(min_coord[0], min_coord[1], "x", color="black", markersize=3)
        #     print("name:", self.name, "iter:", suffix, "coords:", max_coord, min_coord)
            
        # plot the grid map
        # if self.grid_map is not None:
        #     for x in range(self.grid_map.grid.shape[0]):
        #         for y in range(self.grid_map.grid.shape[1]):
                    
        #             i,j = x * self.grid_map.res_x + self.grid_map.min_x, y * self.grid_map.res_y + self.grid_map.min_y
        #             if self.grid_map.grid[x][y] == 1:
        #                 plt.plot(i,j,"x",color="red",markersize=3, alpha = 0.5)
        #             else:
        #                 plt.plot(i,j,"o",color="green",markersize=3, alpha = 0.1)
        
        # draw sub goal if present using x
        if self.sub_goal:
            # plot lookahead circle
            # self.plot_ax.add_patch(plt.Circle((self.state.x, self.state.y), self.lookahead_distance, color="blue", fill=False))
            # plot sub goal
            self.plot_ax.plot(self.sub_goal[0], self.sub_goal[1], "x", color="purple", markersize=5)

        plt.savefig(f"out/10/fig/{self.name}_plot_{suffix}_{curr_mode_text}.png")
        plt.close()
        
    def set_grid_map(self, grid_map):
        self.grid_map = grid_map
        
    def plot_omega(self, suffix=""):
        if (not self.is_plotting): 
            return
        # plot the omega values in a line plot
        plt.figure()
        
        # plot the omega history values
        plt.plot(self.omega_l_history, label="omega_l")
        plt.plot(self.omega_r_history, label="omega_r")
        
        # add legend
        plt.legend()
        
        # set auto limit and scale
        plt.autoscale()
        plt.ylim(-10, 10)
        
        # save fig
        plt.savefig(f"out/10/fig/{self.name}_omega_history_{suffix}.png")