import numpy as np
import cvxpy as cvx
import sympy as spy
from dataclasses import dataclass, field

from numpy.typing import NDArray
from shapely.geometry import Point, Polygon

from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.rocket import RocketCommands, RocketState
from dg_commons.sim.models.rocket_structures import RocketGeometry, RocketParameters

from pdm4ar.exercises_def.ex09.utils_params import PlanetParams, SatelliteParams

from pdm4ar.exercises.ex09.rocket import Rocket
from pdm4ar.exercises.ex09.discretization import (
    DiscretizationMethod,
    FirstOrderHold,
    ZeroOrderHold,
)

# Additional imports
from .path_planner import AgentOccupancyGrid, AgentPathPlanner
from dg_commons.sim.models.obstacles import StaticObstacle
import shapely
from typing import Sequence


@dataclass
class FullSatelliteParams:
    planet_center: list[float, float]
    orbit_r: float
    omega: float
    tau: float
    radius: float


@dataclass(frozen=True)
class SatellitesSolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(
        default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1))
    )  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant
    # stop_crit: float = 1e-10  # Stopping criteria constant


class SatellitesRocketPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: list[PlanetParams]
    satellites: list[SatelliteParams]
    rocket: Rocket
    sg: RocketGeometry
    sp: RocketParameters
    params: SatellitesSolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: RocketGeometry,
        sp: RocketParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp

        # Additional stuff
        self.s_prime = dict()

        # Solver Parameters
        self.params = SatellitesSolverParameters()

        # Rocket Dynamics
        self.rocket = Rocket(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.rocket, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.rocket, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # Initial Guess
        self.X_bar, self.U_bar, self.p_bar = self.intial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

        self.linear_cost = None
        self.nonlinear_cost = None
        self.last_nonlinear_cost = None

        self.init_method = "astar"

    def compute_trajectory(
        self, init_state: RocketState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[RocketCommands], DgSampledSequence[RocketState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = goal_state

        # INITIALIZATION
        # Convert the planets and satellites to static obstacles

        if self.init_method == "astar":
            self.obstacles = self.calculate_obstacles_from_planets()

            grid_size = 100
            grid_map = AgentOccupancyGrid(self.obstacles, grid_size, self.sg.l)

            path_planner = AgentPathPlanner(
                algorithm="A*",
                grid_size=grid_size,
                grid=grid_map,
                obstacles=self.obstacles,
            )

            shortest_path = path_planner.find_path(
                grid_map.map_to_index(self.init_state.x, self.init_state.y),
                grid_map.map_to_index(self.goal_state.x, self.goal_state.y),
            )

            # print(shortest_path)

            # convert from grid to map coordinates
            path_nodes = [
                (
                    x * grid_map.res_x + grid_map.min_x,
                    y * grid_map.res_y + grid_map.min_y,
                )
                for (x, y) in shortest_path
            ]

            if path_nodes:
                path_nodes.append([self.goal_state.x, self.goal_state.y])

            # Interpolate to get K points
            x_coords, y_coords = zip(*path_nodes)  # separate x and y coordinates
            x_interp = np.interp(
                np.linspace(0, len(path_nodes) - 1, self.params.K),
                np.arange(len(path_nodes)),
                x_coords,
            )
            y_interp = np.interp(
                np.linspace(0, len(path_nodes) - 1, self.params.K),
                np.arange(len(path_nodes)),
                y_coords,
            )

            self.X_bar[0, :] = x_interp
            self.X_bar[0, 0] = self.init_state.x
            self.X_bar[0, -1] = self.goal_state.x

            self.X_bar[1, :] = y_interp
            self.X_bar[1, 0] = self.init_state.y
            self.X_bar[1, -1] = self.goal_state.y

        if self.init_method == "manhattan":
            # self.X_bar[0, :] = np.linspace(self.init_state.x, self.goal_state.x, len(self.X_bar[1]))
            self.X_bar[0, 0 : (len(self.X_bar[1]) // 2)] = np.linspace(
                self.init_state.x, self.goal_state.x, len(self.X_bar[1]) // 2
            )
            self.X_bar[0, (len(self.X_bar[1]) // 2) :] = np.linspace(
                self.goal_state.x, self.goal_state.x, len(self.X_bar[1]) // 2
            )

            # self.X_bar[1, :] = np.linspace(self.init_state.y, self.goal_state.y, len(self.X_bar[1]))
            self.X_bar[1, 0 : (len(self.X_bar[1]) // 2)] = np.linspace(
                self.init_state.y, self.init_state.y, len(self.X_bar[1]) // 2
            )
            self.X_bar[1, (len(self.X_bar[1]) // 2) :] = np.linspace(
                self.init_state.y, self.goal_state.y, len(self.X_bar[1]) // 2
            )

        if self.init_method == "euclidean":
            self.X_bar[0, :] = np.linspace(
                self.init_state.x, self.goal_state.x, len(self.X_bar[1])
            )
            self.X_bar[1, :] = np.linspace(
                self.init_state.y, self.goal_state.y, len(self.X_bar[1])
            )

    

        self.X_bar[2, :] = self.init_state.psi
        self.X_bar[7, :] = self.init_state.m

        self.U_bar[:] = 0
        self.p_bar[0] = 10.0  # Guess for final time

        self.tr_radius = self.params.tr_radius
        converged = False

        for i in range(self.params.max_iterations):
            self._convexification()

            print("-" * 18 + "Iteration " + str(i) + "-" * 18)

            try:
                error = self.problem.solve(
                    verbose=self.params.verbose_solver, solver=self.params.solver
                )
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")

            print("Problem status: ", self.problem.status)

            # Extract and print the optimized control and state trajectories
            X_new = self.variables["X"].value
            U_new = self.variables["U"].value
            p_new = self.variables["p"].value

            X_nl = self.integrator.integrate_nonlinear_piecewise(
                X_new, U_new, p_new
            )  # flow map

            linear_cost_dynamics = np.linalg.norm(
                self.variables["nu"].value, 1
            )  # delta nu
            nonlinear_cost_dynamics = np.linalg.norm(X_new - X_nl, 1)  # defect delta

            linear_cost_constraints = self._calculate_linear_cost_constraint()
            nonlinear_cost_constraints = self._calculate_nonlinear_cost_constraint(
                X_new
            )

            self.linear_cost = linear_cost_dynamics + linear_cost_constraints
            self.nonlinear_cost = nonlinear_cost_dynamics + nonlinear_cost_constraints

            if self.last_nonlinear_cost is None:
                self.last_nonlinear_cost = self.nonlinear_cost
                self.X_bar = X_new
                self.U_bar = U_new
                self.p_bar = p_new
                continue

            self.actual_change = self.last_nonlinear_cost - self.nonlinear_cost
            self.predicted_change = self.last_nonlinear_cost - self.linear_cost

            print("")
            print("Actual change: ", self.actual_change)
            print("Predicted change: ", self.predicted_change)
            print("Final time (p_new): ", p_new)
            print("Final time (p_bar): ", self.p_bar)
            print("")

            # check if SCvx converged
            if abs(self.predicted_change) < self.params.stop_crit:
                converged = True
                print("SCvx converged.")
            else:
                self._update_trust_region(X_new, U_new, p_new)

            if converged:
                print(f"Converged after {i + 1} iterations.")
                break

        # slack = 0
        # for obstacle in self.s_prime:
        #     slack += cvx.sum(self.s_prime[obstacle]).value
        # print("------------Final slack-----------------: ", slack)
        # print("------------ nu -----------------: ", self.variables["nu"].value)

        mycmds, mystates = self._extract_seq_from_array()

        return mycmds, mystates

    def calculate_obstacles_from_planets(self):
        obstacles = []
        for planet in self.planets:
            planet_radius = self.planets[planet].radius
            planet_center = np.array(self.planets[planet].center)
            obstacles.append(Point(planet_center).buffer(planet_radius))
        return obstacles

    def intial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K  # Number of time steps

        # Initialize state (X) and control input (U) matrices with zeros
        X = np.zeros((self.rocket.n_x, K))
        U = np.zeros((self.rocket.n_u, K))
        p = np.zeros((self.rocket.n_p))

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter(6)

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        # Variables
        variables = {
            "X": cvx.Variable((self.rocket.n_x, self.params.K)),
            "U": cvx.Variable((self.rocket.n_u, self.params.K)),
            "p": cvx.Variable(self.rocket.n_p),
            "nu": cvx.Variable((self.rocket.n_x, self.params.K - 1)),
        }

        for planet in self.planets:
            self.s_prime[planet] = cvx.Variable((self.params.K, 1), nonneg=True)

        for satellite in self.satellites:
            self.s_prime[satellite] = cvx.Variable((self.params.K, 1), nonneg=True)

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        self._set_goal()

        problem_parameters = {
            "init_state": cvx.Parameter(self.rocket.n_x),
            "goal_state": cvx.Parameter(6, 1),
            "m_v": cvx.Parameter(nonneg=True),
            "C_T": cvx.Parameter(),
            "F_limits": cvx.Parameter(shape=(2,)),
            "phi_limits": cvx.Parameter(shape=(2,)),
            "dphi_limits": cvx.Parameter(shape=(2,)),
            "A_bar": cvx.Parameter(
                (self.rocket.n_x * self.rocket.n_x, self.params.K - 1)
            ),
            "B_plus_bar": cvx.Parameter(
                (self.rocket.n_x * self.rocket.n_u, self.params.K - 1)
            ),
            "B_minus_bar": cvx.Parameter(
                (self.rocket.n_x * self.rocket.n_u, self.params.K - 1)
            ),
            "F_bar": cvx.Parameter((self.rocket.n_x, self.params.K - 1)),
            "r_bar": cvx.Parameter((self.rocket.n_x, self.params.K - 1)),
            "X_last": cvx.Parameter((self.rocket.n_x, self.params.K)),
            "U_last": cvx.Parameter((self.rocket.n_u, self.params.K)),
            "p_last": cvx.Parameter(shape=(1,)),
            "weight_nu": cvx.Parameter(nonneg=True),  # Time factor
            "tr_radius": cvx.Parameter(nonneg=True),
            "init_mass": cvx.Parameter(nonneg=True),
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.constraints]:
        """
        Define constraints for SCvx.
        """
        constraints = []

        constraints = [
            # Initial and finalState constraints
            self.variables["X"][:, 0] == self.problem_parameters["init_state"],
            self.variables["X"][0:6, -1] == self.problem_parameters["goal_state"],
            # Mass constraint
            self.variables["X"][7, :] >= self.problem_parameters["m_v"],
            # Control inputs
            self.variables["U"][:, 0] == 0,
            self.variables["U"][:, -1] == 0,
            self.variables["U"][0:2, :] >= self.problem_parameters["F_limits"][0],
            self.variables["U"][0:2, :] <= self.problem_parameters["F_limits"][1],
            # Thrust angle
            self.variables["X"][6, :] >= self.problem_parameters["phi_limits"][0],
            self.variables["X"][6, :] <= self.problem_parameters["phi_limits"][1],
            # Rate of change of thrust angle
            self.variables["U"][2, :] >= self.problem_parameters["dphi_limits"][0],
            self.variables["U"][2, :] <= self.problem_parameters["dphi_limits"][1],
            # self.variables["p"][:] <= self.problem_parameters["p_last"][0],
            self.variables["p"][:] >= 0.0,
            # Setting minimum time constraint since we're optimizing over it
            self.variables["X"][0:2, :] >= -10,
            self.variables["X"][0:2, :] <= 10,
        ]

        # # constraint on slack variables
        # for j, planet in enumerate(self.planets):
        #     constraints += [self.s_prime[j][:] <= 0.1]
        # is there a way to visualize the contraints?

        # Trust region:
        du = self.variables["U"] - self.problem_parameters["U_last"]
        dx = self.variables["X"] - self.problem_parameters["X_last"]
        dp = self.variables["p"] - self.problem_parameters["p_last"]
        constraints += [
            cvx.norm(dx, 1) + cvx.norm(du, 1) + cvx.norm(dp, 1)
            <= self.problem_parameters["tr_radius"]
        ]

        ## Dynamics constraints

        # Planets
        for j, planet in enumerate(self.planets):
            planet_radius = (
                self.planets[planet].radius + self.sg.l
            )  # Adding length of rocket
            planet_center = np.array(self.planets[planet].center)

            for i in range(self.variables["X"].shape[1]):
                X_ref = self.problem_parameters["X_last"][0:2, i]
                X_current = self.variables["X"][0:2, i]
                H_j = 1 / planet_radius
                episilon = 1e-6  # Added so that denominator doesn't go to 0
                obstacle_constraint_norm = cvx.norm(H_j * (X_ref - planet_center), 2)

                C_current = (
                    -1
                    * (H_j * H_j * (X_ref - planet_center))
                    / (obstacle_constraint_norm + episilon)
                    @ (X_current - planet_center)
                )
                constraints.append(1 + C_current <= self.s_prime[planet])

        # Satellites
        for i, satellite in enumerate(self.satellites):
            planet, _ = satellite.split("/")

            planet_center = np.array(self.planets[planet].center)

            satellite_radius = self.satellites[satellite].radius + self.sg.l
            satellite_omega = self.satellites[satellite].omega
            satellite_tau = self.satellites[satellite].tau
            satellite_orbit_r = self.satellites[satellite].orbit_r

            satellite_full = FullSatelliteParams(
                planet_center,
                satellite_orbit_r,
                satellite_omega,
                satellite_tau,
                satellite_radius,
            )

            # Calculate the center of the satellite through time
            satellite_center_positions = self._get_sat_center(satellite_full)

            for i in range(self.variables["X"].shape[1]):
                X_ref = self.problem_parameters["X_last"][0:2, i]
                X_current = self.variables["X"][0:2, i]
                satellite_center = satellite_center_positions[:, i]
                H_j = 1 / satellite_radius
                episilon = 1e-6  # Added so that denominator doesn't go to 0
                obstacle_constraint_norm = cvx.norm(H_j * (X_ref - satellite_center), 2)

                C_current = (
                    -1
                    * (H_j * H_j * (X_ref - satellite_center))
                    / (obstacle_constraint_norm + episilon)
                    @ (X_current - satellite_center)
                )
                constraints.append(1 + C_current <= self.s_prime[satellite])

        # constraints += [
        #     self.variables["X"][:, k + 1]
        #     == cvx.reshape(
        #         self.problem_parameters["A_bar"][:, k],
        #         (self.rocket.n_x, self.rocket.n_x),
        #     )
        #     @ self.variables["X"][:, k]
        #     + cvx.reshape(
        #         self.problem_parameters["B_bar"][:, k],
        #         (self.rocket.n_x, self.rocket.n_u),
        #     )
        #     @ self.variables["U"][:, k]
        #     + self.problem_parameters["F_bar"][:, k]
        #     # cvx.reshape( self.problem_parameters["F_bar"][:, k],(self.rocket.n_x, self.rocket.n_u),)
        #     # @ self.variables["U"][:, k + 1]
        #     + self.problem_parameters["r_bar"][:, k] + self.variables["p"]
        #     for k in range(self.params.K - 1)
        # ]

        # FOH Dynamics (pg. 33)
        for k in range(self.params.K - 1):
            constraints += [
                self.variables["X"][:, k + 1]
                == cvx.reshape(
                    self.problem_parameters["A_bar"][:, k],
                    (self.rocket.n_x, self.rocket.n_x),
                )
                @ self.variables["X"][:, k]
                + cvx.reshape(
                    self.problem_parameters["B_minus_bar"][:, k],
                    (self.rocket.n_x, self.rocket.n_u),
                )
                @ self.variables["U"][:, k]
                + cvx.reshape(
                    self.problem_parameters["B_plus_bar"][:, k],
                    (self.rocket.n_x, self.rocket.n_u),
                )
                @ self.variables["U"][:, k + 1]
                + self.problem_parameters["F_bar"][:, k] * self.variables["p"]
                + self.problem_parameters["r_bar"][:, k]
                + self.variables["nu"][:, k]
            ]

        return constraints

    def _get_objective(self) -> cvx.Problem:
        """
        Define objective for SCvx.
        """

        # Example objective
        objective = self.problem_parameters["weight_nu"] * cvx.norm(
            self.variables["nu"], 1
        )
        objective += (
            self.params.weight_p @ self.variables["p"]
        )  # Optimizes the final time

        weight_fuel = 0.5
        objective += weight_fuel * cvx.sum_squares(
            self.variables["U"][0:2, :]
        )  # Minimize fuel usage

        # minimize mass loss
        objective += 10 * cvx.sum_squares(
            self.variables["X"][7, :] - self.problem_parameters["init_mass"]
        )

        # Minimize use of virtual control
        slack = 0
        for obstacle in self.s_prime:
            slack += cvx.sum(self.s_prime[obstacle])
        objective += 1e5 * slack

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        (
            A_bar,
            B_plus_bar,
            B_minus_bar,
            F_bar,
            r_bar,
        ) = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)

        # Update problem parameters with the new discretized dynamics (ZOH)
        # self.problem_parameters["A_bar"].value = A_bar
        # self.problem_parameters["B_bar"].value = B_bar
        # self.problem_parameters["F_bar"].value = F_bar
        # self.problem_parameters["r_bar"].value = r_bar

        # Update problem parameters with the new discretized dynamics (FOH)
        self.problem_parameters["A_bar"].value = A_bar
        self.problem_parameters["B_plus_bar"].value = B_plus_bar
        self.problem_parameters["B_minus_bar"].value = B_minus_bar
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar

        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        self.problem_parameters["goal_state"].value = self.goal_state.as_ndarray()

        self.problem_parameters["m_v"].value = self.sp.m_v
        self.problem_parameters["C_T"].value = self.sp.C_T
        self.problem_parameters["F_limits"].value = np.array(self.sp.F_limits)
        self.problem_parameters["phi_limits"].value = np.array(self.sp.phi_limits)
        self.problem_parameters["dphi_limits"].value = np.array(self.sp.dphi_limits)

        self.problem_parameters["X_last"].value = self.X_bar
        self.problem_parameters["U_last"].value = self.U_bar
        self.problem_parameters["p_last"].value = self.p_bar

        self.problem_parameters["weight_nu"].value = self.params.lambda_nu
        self.problem_parameters["tr_radius"].value = self.tr_radius
        self.problem_parameters["init_mass"].value = self.init_state.m

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        # check if SCvx converged
        pass

    def _update_trust_region(self, X_new, U_new, p_new) -> float:
        """
        Update trust region radius.
        """
        rho = self.actual_change / self.predicted_change
        if rho < self.params.rho_0:
            print(
                f'Trust region too large. Solving again with radius={self.problem_parameters["tr_radius"].value/ self.params.alpha}'
            )
            new_tr_radius = (
                self.problem_parameters["tr_radius"].value / self.params.alpha
            )

            if new_tr_radius < self.params.min_tr_radius:
                new_tr_radius = self.params.min_tr_radius

            self.tr_radius = new_tr_radius

        else:
            self.X_bar = X_new
            self.U_bar = U_new
            self.p_bar = p_new

            if rho < self.params.rho_1:
                new_tr_radius = (
                    self.problem_parameters["tr_radius"].value / self.params.alpha
                )

                if new_tr_radius < self.params.min_tr_radius:
                    new_tr_radius = self.params.min_tr_radius

                print("Decreasing radius to ", new_tr_radius)
                self.tr_radius = new_tr_radius

            elif rho >= self.params.rho_2:
                new_tr_radius = (
                    self.problem_parameters["tr_radius"].value * self.params.beta
                )
                print("Increasing radius to ", new_tr_radius)

                if new_tr_radius > self.params.max_tr_radius:
                    new_tr_radius = self.params.max_tr_radius

                self.tr_radius = new_tr_radius

            self.last_nonlinear_cost = self.nonlinear_cost

    def _extract_seq_from_array(
        self,
    ) -> tuple[DgSampledSequence[RocketCommands], DgSampledSequence[RocketState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        # self._plot_states()
        # self._plot_controls()

        npstates = np.array(self.variables["X"].value).T
        cmdsnp = np.array(self.variables["U"].value).T
        ts = np.linspace(0, self.variables["p"].value[0], self.params.K)
        print("Final time: ", self.variables["p"].value[0])

        cmds_list = [RocketCommands(*inp) for inp in cmdsnp]
        mycmds = DgSampledSequence[RocketCommands](timestamps=ts, values=cmds_list)

        states = [RocketState(*state) for state in npstates]
        mystates = DgSampledSequence(timestamps=ts, values=states)

        return mycmds, mystates

    def _plot_states(self) -> None:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.xlim(-12, 12)
        plt.ylim(-12, 12)
        plt.grid()
        plt.gca().set_aspect("equal", adjustable="box")
        self.plot_ax = plt.gca()

        X_states = self.variables["X"].value
        U_states = self.variables["U"].value

        x_values, y_values, psi_values = X_states[0, :], X_states[1, :], X_states[2, :]
        vx_values, vy_values = X_states[3, :], X_states[4, :]
        dphi_values, phi_values, m_values = (
            X_states[5, :],
            X_states[6, :],
            X_states[7, :],
        )

        F_l_values, F_r_values, dphi_values = (
            U_states[0, :],
            U_states[1, :],
            U_states[2, :],
        )

        radius = 0.25
        arrow_length = 0.75
        for x, y, psi in zip(x_values, y_values, psi_values):
            # Create and add circle
            circle = plt.Circle((x, y), radius=radius, fill=False, color="blue")
            plt.gca().add_patch(circle)

            # Calculate arrow's end point
            dx = arrow_length * np.cos(psi)
            dy = arrow_length * np.sin(psi)

            # Draw arrow
            plt.arrow(
                x, y, dx, dy, head_width=0.2, head_length=0.3, fc="red", ec="black"
            )

        for planet in self.planets:
            planet_center = np.array(self.planets[planet].center)
            planet_radius = self.planets[planet].radius
            circle = plt.Circle(
                planet_center, radius=planet_radius, fill=False, color="brown"
            )
            plt.gca().add_patch(circle)
            circle = plt.Circle(
                planet_center, radius=planet_radius + self.sg.l, fill=False, color="red"
            )
            plt.gca().add_patch(circle)
        plt.title(
            f"Rocket Trajectory (Initial Mass: {round(m_values[0], 3)}, Final mass: {round(m_values[-1], 3)}, Dry mass: {self.sp.m_v})"
        )
        plt.savefig("output_states.png")
        plt.close()

    def _plot_controls(self) -> None:
        import matplotlib.pyplot as plt

        X_states = self.variables["X"].value
        U_values = self.variables["U"].value
        p_value = self.variables["p"].value[0]
        F_l_values, F_r_values, dphi_values = (
            U_values[0, :],
            U_values[1, :],
            U_values[2, :],
        )

        timestamps = np.linspace(0, p_value, X_states.shape[1])

        output_controls_fig = plt.figure()

        ax1 = output_controls_fig.add_subplot(3, 1, 1)
        ax2 = output_controls_fig.add_subplot(3, 1, 2)
        ax3 = output_controls_fig.add_subplot(3, 1, 3)

        # Plot F_l over time
        ax1.plot(timestamps, F_l_values)
        ax1.set_title("RocketState F_l over time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("F_l")

        # Plot F_r over time
        ax2.plot(timestamps, F_r_values)
        ax2.set_title("RocketState F_r over time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("F_r")

        # Plot dphi over time
        ax3.plot(timestamps, dphi_values)
        ax3.set_title("RocketState dphi over time")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("dphi")

        # Adjust the layoutc
        output_controls_fig.tight_layout(pad=0.5)

        plt.savefig("output_controls.png")
        plt.close()

    def _calculate_linear_cost_constraint(self) -> float:
        """
        Calculates the linear cost of the constraints
        """
        cost = 0.0

        for obstacle in self.s_prime:
            cost += np.sum(self.s_prime[obstacle].value)

        print("Linear cost constraint: ", cost)

        return cost

    def _calculate_nonlinear_cost_constraint(self, X_new) -> float:
        """
        Calculates the nonlinear cost of the constraints
        """
        cost = 0.0

        # Planet
        for i, planet in enumerate(self.planets):
            planet_center = np.array(self.planets[planet].center)
            planet_radius = self.planets[planet].radius

            vector_to_obstacle = X_new[0:2, :].T - planet_center
            dist_to_obstacle = np.linalg.norm(vector_to_obstacle, 2, axis=1)
            is_violated = dist_to_obstacle < planet_radius + self.sg.l
            violation = planet_radius + self.sg.l - dist_to_obstacle
            cost += np.sum(is_violated * violation)

        # Satellites
        for j, satellite in enumerate(self.satellites):
            planet, _ = satellite.split("/")

            planet_center = np.array(self.planets[planet].center)

            satellite_radius = self.satellites[satellite].radius
            satellite_omega = self.satellites[satellite].omega
            satellite_tau = self.satellites[satellite].tau
            satellite_orbit_r = self.satellites[satellite].orbit_r

            satellite_full = FullSatelliteParams(
                planet_center,
                satellite_orbit_r,
                satellite_omega,
                satellite_tau,
                satellite_radius,
            )

            # Calculate the center of the satellite through time
            satellite_centers = self._get_sat_center(satellite_full)

            vector_to_obstacle = X_new[0:2, :].T - satellite_centers.T
            dist_to_obstacle = np.linalg.norm(vector_to_obstacle, 2, axis=1)
            is_violated = dist_to_obstacle < satellite_radius + self.sg.l
            violation = satellite_radius + self.sg.l - dist_to_obstacle
            cost += np.sum(is_violated * violation)

        print("Nonlinear cost obstacles: ", cost)

        return cost

    def _get_sat_center(self, satellite: FullSatelliteParams):
        """
        Gets the center of the satellites with respect to time
        """
        orbit_r = satellite.orbit_r
        omega = satellite.omega
        tau = satellite.tau
        planet_x, planet_y = satellite.planet_center[0], satellite.planet_center[1]

        x_arr, y_arr = [], []

        # ts = np.linspace(0, self.variables["p"].value[0], self.params.K)
        # print("p_bar value: ", self.p_bar)
        for t in range(self.params.K):
            t = (t / self.params.K) * self.p_bar[0]

            cos_omega_t = orbit_r * (np.cos(tau) - np.sin(tau) * (omega * t))
            sin_omega_t = orbit_r * (np.sin(tau) + np.sin(tau) * (omega * t))
            x = cos_omega_t + planet_x
            y = sin_omega_t + planet_y

            x_arr.append(x)
            y_arr.append(y)

        pos = np.array([x_arr, y_arr])
        return pos

    def __convert_planets_to_static_obstacles(
        self, planets: dict[PlayerName, PlanetParams], rocket_length: float
    ):
        """
        Converts planets to list of StaticObstacles.
        """
        obstacles = list()

        for planet in planets.keys():
            planet_center = planets[planet].center
            planet_radius = planets[planet].radius

            planet_shapely_point = shapely.Point(planet_center).buffer(planet_radius)
            obstacles.append(planet_shapely_point)

        return obstacles

    def __convert_satellites_to_static_obstacles(
        self, satellites: dict[PlayerName, SatelliteParams], rocket_length: float
    ):
        """
        Converts satellites to list of StaticObstacles.
        """
        obstacles = list()

        for satellite in satellites:
            planet, _ = satellite.split("/")

            planet_center = np.array(self.planets[planet].center)

            satellite_radius = self.satellites[satellite].radius
            satellite_tau = self.satellites[satellite].tau
            satellite_orbit_r = self.satellites[satellite].orbit_r

            satellite_center = np.array(
                [
                    planet_center[0] + satellite_orbit_r * np.cos(satellite_tau),
                    planet_center[1] + satellite_orbit_r * np.sin(satellite_tau),
                ]
            )

            satellite_shapely_point = shapely.Point(satellite_center).buffer(
                satellite_radius + rocket_length
            )
            obstacles.append(satellite_shapely_point)

        return obstacles