from dataclasses import dataclass
from typing import Sequence
import numpy as np

# import matplotlib.pyplot as plt

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.rocket import RocketCommands, RocketState
from dg_commons.sim.models.rocket_structures import RocketGeometry, RocketParameters
from pdm4ar.exercises.ex09.old_replanner_planets import OldRocketPlanner

from pdm4ar.exercises.ex09.planner import RocketPlanner
from pdm4ar.exercises.ex09.satellites_planner import SatellitesRocketPlanner
from pdm4ar.exercises_def.ex09.goal import RocketTarget, SatelliteTarget
from pdm4ar.exercises_def.ex09.utils_params import PlanetParams, SatelliteParams
from pdm4ar.exercises.ex09.moving_goal_planner import MovingGoalRocketPlanner


@dataclass(frozen=True)
class Pdm4arAgentParams:
    """
    Definition space for additional agent parameters.
    """

    pos_tol: 0.5
    dir_tol: 0.5
    vel_tol: 1.0


class RocketAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: RocketState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[RocketCommands]
    state_traj: DgSampledSequence[RocketState]
    myname: PlayerName
    planner: RocketPlanner
    planet_planner: OldRocketPlanner
    satellites_planner: SatellitesRocketPlanner
    moving_goal_planner: MovingGoalRocketPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: RocketGeometry
    sp: RocketParameters

    def __init__(
        self,
        init_state: RocketState,
        satellites: dict[PlayerName, SatelliteParams],
        planets: dict[PlayerName, PlanetParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only at the beginning of each simulation.
        Provides the RocketAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.init_state = init_state
        self.satellites = satellites
        self.planets = planets

        self.replan = False
        self.replan_time = 0.0

        self.calulcated_p = None
        self.goal_time = 0.0

        """
        if self.satellites:
            omega_fastest = 0
            for satellite in self.satellites.keys():
                omega_sat = self.satellites[satellite].omega
                if omega_sat >= omega_fastest:
                    omega_fastest = omega_sat

            self.replan_time = 0.5 / omega_fastest
        """

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        Feel free to add additional methods, objects and functions that help you to solve the task
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params

        self.planner = RocketPlanner(
            planets=self.planets, satellites=self.satellites, sg=self.sg, sp=self.sp
        )

        self.satellites_planner = SatellitesRocketPlanner(
            planets=self.planets, satellites=self.satellites, sg=self.sg, sp=self.sp
        )

        self.moving_goal_planner = MovingGoalRocketPlanner(
            planets=self.planets, satellites=self.satellites, sg=self.sg, sp=self.sp
        )

        goal_time_array = [20]
        # Get goal from Targets (either moving (SatelliteTarget) or static (RocketTarget))
        min_dist = np.inf
        self.goal_time = 25.0

        cmds_plan_all = []
        state_traj_all = []

        if isinstance(init_sim_obs.goal, SatelliteTarget):
            for i in goal_time_array:
                self.goal_state = init_sim_obs.goal.get_target_state_at(i)
                print("Goal Time: ", i)
                try:
                    # run convexification for all of the goal times
                    (
                        cmds_plan,
                        state_traj,
                        calulcated_p,
                        iter_nr,
                    ) = self.moving_goal_planner.compute_trajectory(
                        self.init_state, self.goal_state, i
                    )

                    cmds_plan_all.append(cmds_plan)
                    state_traj_all.append(state_traj)

                except Exception:
                    print("Failed to convexify")
                    continue
                # calculate the distance between the time i and the calculated_p as sqrt
                dist = np.sqrt((i - calulcated_p) ** 2)
                print("Distance: ", dist)
                # if the distance is smaller than the min_dist, update the min_dist and the goal_time
                if dist < min_dist and iter_nr < 99:
                    min_dist = dist
                    self.goal_time = i
                    self.goal_state = init_sim_obs.goal.get_target_state_at(
                        self.goal_time
                    )

                    self.cmds_plan = cmds_plan_all[goal_time_array.index(i)]
                    self.state_traj = state_traj_all[goal_time_array.index(i)]
                    break
        elif isinstance(init_sim_obs.goal, RocketTarget):
            self.goal_state = init_sim_obs.goal.target

            # self.goal_data = init_sim_obs.goal

            # if self.satellites and isinstance(init_sim_obs.goal, SatelliteTarget):
            #     (
            #         self.cmds_plan,
            #         self.state_traj,
            #         calulcated_p,
            #         iter_nr,
            #     ) = self.moving_goal_planner.compute_trajectory(
            #         self.init_state, self.goal_state, self.goal_time
            #     )
            # elif self.satellites and isinstance(init_sim_obs.goal, RocketTarget):
            (
                self.cmds_plan,
                self.state_traj,
            ) = self.satellites_planner.compute_trajectory(
                self.init_state, self.goal_state
            )
        elif not self.satellites:
            self.cmds_plan, self.state_traj = self.planner.compute_trajectory(
                self.init_state, self.goal_state
            )

    def get_commands(self, sim_obs: SimObservations) -> RocketCommands:
        """
        This is called by the simulator at every time step. (0.1 sec)
        Do not modify the signature of this method.
        """
        current_state = sim_obs.players[self.myname].state
        expected_state = self.state_traj.at_interp(sim_obs.time)

        curr_state = RocketAgent.st_as_ndarray(current_state)
        expected_state = RocketAgent.st_as_ndarray(expected_state)

        """
        Hacking the system
        if (np.sqrt((expected_state[0]-self.goal_state.x)**2 + (expected_state[1]-self.goal_state.y)**2) < 0.5):
            print("Goal Reached1")
            if (
                np.linalg.norm(
                    self.goal_data.get_target_state_at(sim_obs.time) - current_state
                )
                < 0.5
            ):
                print("Goal Reached2")
                # give a coomand so that the rocket moves with same velocity
                return RocketCommands(1.0, 1.0, 0.0)
        else:
            cmds = self.cmds_plan.at_interp(sim_obs.time)
        """
        cmds = self.cmds_plan.at_interp(sim_obs.time)
        # curr_cmds = self.cmds_plan.at_interp(sim_obs.time).as_ndarray()
        curr_cmds = np.zeros(
            [
                3,
            ]
        )
        final_t = self.state_traj.get_end()

        # if np.linalg.norm(curr_state[0:2] - expected_state[0:2], 2) > 0.3:
        #     print(
        #         "Yayyyyyyyyyyyyyyyy",
        #         np.linalg.norm(curr_state[0:2] - expected_state[0:2], 2),
        #     )
        #     print("current_State: ", current_state)
        #     print("expected_State: ", expected_state)
        #     print("______________________", sim_obs.time)
        #     self.replan = True

        # if float(sim_obs.time) >

        # if self.replan:
        #     print("--------------Replanning----------------")
        #     self.replan = False

        #     new_X_ref = []
        #     new_U_ref = []

        #     X_ref = self.state_traj.get_subsequence(sim_obs.time, final_t).values
        #     U_ref = self.cmds_plan.get_subsequence(sim_obs.time, final_t).values
        #     for x, u in zip(X_ref, U_ref):
        #         new_U_ref.append(u.as_ndarray())
        #         new_X_ref.append(x.as_ndarray())

        #     new_X_ref = np.array(new_X_ref).T
        #     new_U_ref = np.array(new_U_ref).T

        #     new_X_ref = np.insert(new_X_ref, 0, curr_state, axis=1)
        #     new_U_ref = np.insert(new_U_ref, 0, curr_cmds, axis=1)

        #     K = new_X_ref.shape[1]
        #     self.replanner = SatellitesRocketPlanner(
        #         planets=self.planets,
        #         satellites=self.satellites,
        #         sg=self.sg,
        #         sp=self.sp,
        #         K=K,
        #     )

        #     try:
        #         self.cmds_plan, self.state_traj = self.replanner.replan(
        #             curr_state,
        #             self.goal_state,
        #             new_X_ref,
        #             new_U_ref,
        #             float(sim_obs.time),
        #             final_t,
        #             self.planner.tr_radius,
        #         )
        #     except:
        #         print("failed to replan")

        # FirstOrderHold

        return cmds

    @staticmethod
    def st_as_ndarray(st):
        st = st.__dict__
        state = np.array(
            [
                st["x"],
                st["y"],
                st["psi"],
                st["vx"],
                st["vy"],
                st["dpsi"],
                st["phi"],
                st["m"],
            ]
        )
        return state
