#!/usr/bin/env python3
"""Test client for the /plan_trajectory service.

Calls the GA planner node with a simple scenario and prints the result.
Unlike `ros2 service call`, this has no internal timeout.
"""
import sys
import rclpy
from rclpy.node import Node
from gahomotopy_msgs.srv import PlanTrajectory
from gahomotopy_msgs.msg import Obstacle
from geometry_msgs.msg import Point


class PlanClient(Node):
    def __init__(self):
        super().__init__('plan_client')
        self._client = self.create_client(PlanTrajectory, '/plan_trajectory')
        self.get_logger().info('Waiting for /plan_trajectory service...')
        self._client.wait_for_service()
        self.get_logger().info('Service available.')

    def call(self, start, goal, obstacles, name):
        req = PlanTrajectory.Request()
        req.start = Point(x=start[0], y=start[1], z=start[2])
        req.goal = Point(x=goal[0], y=goal[1], z=goal[2])
        req.obstacles = [
            Obstacle(x=o[0], y=o[1], z=o[2], radius=o[3])
            for o in obstacles
        ]
        req.experiment_name = name

        self.get_logger().info(f'Sending request: "{name}"')
        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('Service call failed')
            return None

        return future.result()


def main():
    rclpy.init()
    node = PlanClient()

    # Simple test scenario (Roarm M2, obstacles10)
    start = (32.82, 14.52, -17.44)
    goal = (-32.82, 14.52, -17.44)
    obstacles = [
        (23.015, 28.878, 0.0, 6.0),
        (23.015, 11.878, 0.0, 6.0),
    ]
    name = 'ros_test_quick'

    result = node.call(start, goal, obstacles, name)

    if result is not None:
        print()
        print('=' * 60)
        print(f'Success:    {result.success}')
        t = result.trajectory
        print(f'Points:     {t.num_points}')
        print(f'Num vars:   {t.num_var}')
        print(f'Distance:   {t.path_distance:.6f}')
        print(f'Lambda:     {t.final_lambda:.6f}')
        print(f'Time:       {t.elapsed_time:.1f}s')
        print(f'Radius:     {t.radius:.6f}')
        print(f'Obs values: {t.obstacle_values}')
        print(f'Obs signs:  {t.obstacle_signs}')
        print(f'A matrix:   {t.a_matrix}')
        print(f'Fitness:    {t.fitness_evolution}')
        print('=' * 60)

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()