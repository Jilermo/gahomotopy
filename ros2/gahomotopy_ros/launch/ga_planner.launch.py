"""Launch file for the GA Planner node with YAML parameters."""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Default config file
    default_config = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'default.yaml'
    )

    config_arg = DeclareLaunchArgument(
        'config',
        default_value=default_config,
        description='Path to YAML parameter file',
    )

    ga_node = Node(
        package='gahomotopy_ros',
        executable='ga_planner_node',
        name='ga_planner_node',
        output='screen',
        parameters=[LaunchConfiguration('config')],
    )

    return LaunchDescription([
        config_arg,
        ga_node,
    ])