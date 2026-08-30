#!/usr/bin/env python3
"""
GA Planner Node

Receives a scenario (start, goal, obstacles) via the /plan_trajectory service,
runs the genetic algorithm to optimize EHPPM parameters using the existing
gahomotopy core library, and returns the optimized trajectory.
The trajectory is also saved as a .npy file.
"""
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node

from gahomotopy_msgs.srv import PlanTrajectory
from gahomotopy_msgs.msg import TrajectoryResult


class GaPlannerNode(Node):
    """ROS 2 node that wraps the gahomotopy GeneticAlgorithm for trajectory planning.

    GA parameters and EHPPM search ranges are loaded from YAML at startup.
    The scenario (start, goal, obstacles) is sent per-request via the
    /plan_trajectory service.
    """

    def __init__(self):
        super().__init__('ga_planner_node')

        # --- GA parameters ---
        self.declare_parameter('num_generations', 50)
        self.declare_parameter('population_size', 20)
        self.declare_parameter('num_parents_mating', 10)
        self.declare_parameter('max_number_of_radius', 10000)
        self.declare_parameter('matrix_module', 'diagonal_dominant')
        self.declare_parameter('parallel_mode', 'none')

        # --- Robot parameters ---
        self.declare_parameter('robot_type', 'roarm_m2')
        self.declare_parameter('num_segments', 200)

        # --- EHPPM search ranges (override arm.ga_ranges defaults) ---
        self.declare_parameter('radius_low', 0.05)
        self.declare_parameter('radius_high', 0.3)
        self.declare_parameter('obstacle_value_low', 100.0)
        self.declare_parameter('obstacle_value_high', 100000.0)

        # --- Output ---
        self.declare_parameter('output_directory', 'results')
        self.declare_parameter('save_npy', True)
        self.declare_parameter('output_filename', '')    # override filename; empty = use experiment_name

        # --- Read parameters ---
        self._num_generations = int(self.get_parameter('num_generations').value)
        self._population_size = int(self.get_parameter('population_size').value)
        self._num_parents_mating = int(self.get_parameter('num_parents_mating').value)
        self._max_number_of_radius = int(self.get_parameter('max_number_of_radius').value)
        self._matrix_module = str(self.get_parameter('matrix_module').value)
        self._parallel_mode = str(self.get_parameter('parallel_mode').value)
        self._robot_type = str(self.get_parameter('robot_type').value)
        self._num_segments = int(self.get_parameter('num_segments').value)
        self._radius_low = float(self.get_parameter('radius_low').value)
        self._radius_high = float(self.get_parameter('radius_high').value)
        self._obstacle_value_low = float(self.get_parameter('obstacle_value_low').value)
        self._obstacle_value_high = float(self.get_parameter('obstacle_value_high').value)
        self._output_directory = str(self.get_parameter('output_directory').value)
        self._save_npy = bool(self.get_parameter('save_npy').value)
        self._output_filename = str(self.get_parameter('output_filename').value)

        # --- Create service ---
        self._srv = self.create_service(
            PlanTrajectory,
            '/plan_trajectory',
            self._plan_callback,
        )

        self.get_logger().info('GA Planner Node ready')
        self.get_logger().info(f'  Robot:        {self._robot_type}')
        self.get_logger().info(f'  Generations:  {self._num_generations}')
        self.get_logger().info(f'  Population:   {self._population_size}')
        self.get_logger().info(f'  Parents:      {self._num_parents_mating}')
        self.get_logger().info(f'  Matrix:       {self._matrix_module}')
        self.get_logger().info(f'  Parallel:     {self._parallel_mode}')
        self.get_logger().info(f'  Radius:       [{self._radius_low}, {self._radius_high}]')
        self.get_logger().info(f'  Obs value:    [{self._obstacle_value_low}, {self._obstacle_value_high}]')

    # ── Service callback ──────────────────────────────────────────────

    def _plan_callback(self, request, response):
        name = request.experiment_name
        self.get_logger().info(f'=== Planning request: "{name}" ===')

        # Convert ROS msg → Python dicts for the core library
        obstacles = [
            {'center': (obs.x, obs.y, obs.z), 'radius': obs.radius}
            for obs in request.obstacles
        ]
        start = (request.start.x, request.start.y, request.start.z)
        goal = (request.goal.x, request.goal.y, request.goal.z)

        self.get_logger().info(f'  Start:     {start}')
        self.get_logger().info(f'  Goal:      {goal}')
        self.get_logger().info(f'  Obstacles: {len(obstacles)}')

        # Create arm instance
        try:
            arm = self._create_arm()
        except ValueError as e:
            self.get_logger().error(str(e))
            response.success = False
            return response

        arm.setObstaclesPos(obstacles)

        # Build and run GA
        from gahomotopy.planning.genetic_algorithm import GeneticAlgorithm

        ga = GeneticAlgorithm(
            maxNumberOfRadius=self._max_number_of_radius,
            numGenerations=self._num_generations,
            populationSize=self._population_size,
            numParentsMating=self._num_parents_mating,
            obstacles=obstacles,
            start=start,
            goal=goal,
            arm=arm,
            matrix_module=self._matrix_module,
            name=name,
            parallel_processing=self._parse_parallel_mode(),
        )

        t0 = time.time()
        path, homotopy_params, final_lambda, failed, dis, lambdas, fitness_evol = ga.optimize()
        elapsed = time.time() - t0

        # Build TrajectoryResult message
        num_var = len(start)
        path_arr = np.array(path)
        path_spatial = path_arr[:, :num_var]  # strip lambda column

        traj_msg = TrajectoryResult()
        traj_msg.trajectory = path_spatial.flatten().tolist()
        traj_msg.num_var = num_var
        traj_msg.num_points = len(path)
        traj_msg.success = not failed
        traj_msg.path_distance = float(dis)
        traj_msg.final_lambda = float(final_lambda)
        traj_msg.elapsed_time = float(elapsed)
        traj_msg.fitness_evolution = [float(f) for f in fitness_evol]
        traj_msg.experiment_name = name

        # Extract best EHPPM parameters from the GA solution
        radius, obsVal, obsSign = ga._parse_solution(ga._last_solution)
        traj_msg.radius = float(radius)
        traj_msg.obstacle_values = obsVal.tolist()
        traj_msg.obstacle_signs = [int(s) for s in obsSign.tolist()]
        a_matrix = ga.matrix_module.build_matrix(ga._last_solution, num_var)
        traj_msg.a_matrix = a_matrix.flatten().tolist()

        # Save .npy
        if self._save_npy:
            out_dir = self._output_directory
            os.makedirs(out_dir, exist_ok=True)
            filename = self._output_filename if self._output_filename else name
            filepath = os.path.join(out_dir, f'{filename}.npy')
            np.save(filepath, path_spatial)
            self.get_logger().info(f'  Saved: {filepath}')

        # Fill response
        response.success = not failed
        response.trajectory = traj_msg

        if not failed:
            self.get_logger().info(
                f'=== SUCCESS: {len(path)} points, distance={dis:.6f}, '
                f'time={elapsed:.1f}s ==='
            )
        else:
            self.get_logger().warn(
                f'=== FAILED: distance={dis:.6f}, time={elapsed:.1f}s ==='
            )

        return response

    # ── Helpers ────────────────────────────────────────────────────────

    def _create_arm(self):
        """Create robot arm instance based on robot_type parameter."""
        if self._robot_type == 'roarm_m2':
            from gahomotopy.kinematics.roarm import ROARM3DOF
            arm = ROARM3DOF()
        elif self._robot_type == 'ur3e':
            from gahomotopy.kinematics.ur3e import UR3E
            arm = UR3E()
        else:
            raise ValueError(f'Unknown robot_type: {self._robot_type}')

        arm.numSegments = self._num_segments

        # Override ga_ranges with YAML values
        arm.ga_ranges = {
            'radius': {'low': self._radius_low, 'high': self._radius_high},
            'obstacle_value': {
                'low': self._obstacle_value_low,
                'high': self._obstacle_value_high,
            },
            'off_diagonal': arm.ga_ranges['off_diagonal'],
            'diagonal': arm.ga_ranges['diagonal'],
        }
        return arm

    def _parse_parallel_mode(self):
        """Parse parallel_mode string → pygad format.

        "none"        → None
        "process 10"  → ["process", 10]
        "thread 4"    → ["thread", 4]
        """
        mode = self._parallel_mode.strip()
        if not mode or mode == 'none':
            return None
        parts = mode.split()
        if len(parts) >= 2 and parts[0] in ('process', 'thread'):
            return [parts[0], int(parts[1])]
        return None


def main(args=None):
    rclpy.init(args=args)
    node = GaPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()