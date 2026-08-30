#!/usr/bin/env python3
"""
Joint Broadcaster Node

Reads a joint-space trajectory from a .npy file and publishes each
configuration as a JointState message at a configurable interval.

Parameters (YAML or --ros-args -p):
  trajectory_file   — path to the .npy file (N x num_var array, in degrees)
  dt                — seconds between each waypoint (default: 0.01)
  robot_type        — "roarm_m2" or "ur3e" (determines joint names)
  loop              — if true, replay the trajectory indefinitely (default: false)
"""
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


# Joint name maps per robot.
# For arms where the GA optimizes fewer joints than the robot has
# (e.g. Roarm M2 has 3 DOF + 1 gripper), the extra joints are set to 0.
JOINT_NAMES = {
    'roarm_m2': [
        'armR_base_to_L1',
        'armR_L1_to_L2',
        'armR_L2_to_L3',
        'armR_L3_to_L4',
    ],
    'ur3e': [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint',
    ],
}

# Number of actuated joints the GA optimizes (may differ from len(JOINT_NAMES))
GA_DOFS = {
    'roarm_m2': 3,
    'ur3e': 6,
}

DEG2RAD = np.pi / 180.0


class JointBroadcasterNode(Node):
    """Publishes a trajectory from a .npy file as sequential JointState messages."""

    def __init__(self):
        super().__init__('joint_broadcaster')

        # --- Parameters ---
        self.declare_parameter('trajectory_file', '')
        self.declare_parameter('dt', 0.01)
        self.declare_parameter('robot_type', 'roarm_m2')
        self.declare_parameter('loop', False)

        self._filepath = str(self.get_parameter('trajectory_file').value)
        self._dt = float(self.get_parameter('dt').value)
        self._robot_type = str(self.get_parameter('robot_type').value)
        self._loop = bool(self.get_parameter('loop').value)

        if not self._filepath:
            self.get_logger().error('Parameter "trajectory_file" is required')
            raise SystemExit(1)

        if self._robot_type not in JOINT_NAMES:
            self.get_logger().error(
                f'Unknown robot_type: {self._robot_type}. '
                f'Valid: {list(JOINT_NAMES.keys())}'
            )
            raise SystemExit(1)

        self._joint_names = JOINT_NAMES[self._robot_type]
        self._ga_dofs = GA_DOFS[self._robot_type]
        self._num_joints = len(self._joint_names)

        # --- Load trajectory ---
        try:
            data = np.load(self._filepath, allow_pickle=False).astype(np.float64)
        except Exception as e:
            self.get_logger().error(f'Failed to load trajectory: {e}')
            raise SystemExit(1)

        if data.ndim != 2:
            self.get_logger().error(
                f'Expected 2D array, got shape {data.shape}'
            )
            raise SystemExit(1)

        n_cols = data.shape[1]
        if n_cols != self._ga_dofs:
            self.get_logger().error(
                f'Trajectory has {n_cols} columns but {self._robot_type} '
                f'GA optimizes {self._ga_dofs} joints'
            )
            raise SystemExit(1)

        # Pad with zeros for joints not optimized by the GA (e.g. gripper)
        if self._num_joints > self._ga_dofs:
            padding = np.zeros((len(data), self._num_joints - self._ga_dofs))
            data = np.hstack([data, padding])

        self._trajectory = data
        self._num_points = len(data)
        self._index = 0

        self.get_logger().info(f'Loaded: {self._filepath}')
        self.get_logger().info(f'  Robot:      {self._robot_type}')
        self.get_logger().info(f'  Points:     {self._num_points}')
        self.get_logger().info(f'  Joints:     {self._joint_names}')
        self.get_logger().info(f'  dt:         {self._dt}s')
        self.get_logger().info(f'  Loop:       {self._loop}')

        # --- Publisher ---
        self._pub = self.create_publisher(JointState, '/joint_states', 10)

        # --- Timer ---
        self._timer = self.create_timer(self._dt, self._publish_next)

    def _publish_next(self):
        if self._index >= self._num_points:
            if self._loop:
                self._index = 0
                self.get_logger().info('Looping trajectory...')
            else:
                self.get_logger().info('Trajectory playback complete.')
                self._timer.cancel()
                return

        angles_deg = self._trajectory[self._index]
        angles_rad = angles_deg * DEG2RAD

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._joint_names
        msg.position = angles_rad.tolist()

        self._pub.publish(msg)

        if self._index % 100 == 0:
            self.get_logger().info(
                f'Publishing waypoint {self._index + 1}/{self._num_points}'
            )

        self._index += 1


def main(args=None):
    rclpy.init(args=args)
    node = JointBroadcasterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()