"""
diffbot_bringup.launch.py — bring up DiffBot in ROSpad simulation

Launches:
  • robot_state_publisher  (publishes /robot_description → loads DiffBot in 3D sim)
  • joint_state_publisher  (publishes zero wheel joint states)

Usage (ROSpad IDE):
  Open this file → click ⚡ Launch

Usage (real ROS2):
  ros2 launch diffbot_description diffbot_bringup.launch.py
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('diffbot_description')
    urdf_path = os.path.join(pkg, 'urdf', 'diffbot.urdf')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        # Publishes /robot_description so the sim loads the DiffBot
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'publish_frequency': 50.0,
            }]
        ),

        # Publishes zero wheel joint states
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
        ),
    ])
