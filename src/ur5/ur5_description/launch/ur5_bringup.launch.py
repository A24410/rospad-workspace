"""
ur5_bringup.launch.py — bring up UR5 robot description in ROS2

Launches:
  • robot_state_publisher  (publishes /tf and /tf_static from URDF + /joint_states)
  • joint_state_publisher_gui  (GUI sliders to drive joints interactively)
  • rviz2  (3D visualisation using the bundled config)

Usage:
  ros2 launch ur5_description ur5_bringup.launch.py
  ros2 launch ur5_description ur5_bringup.launch.py gui:=false
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('ur5_description')
    urdf_path = os.path.join(pkg, 'urdf', 'ur5.urdf')
    rviz_path = os.path.join(pkg, 'rviz', 'ur5.rviz')

    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    use_gui = LaunchConfiguration('gui', default='true')

    return LaunchDescription([
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Start joint_state_publisher_gui for interactive joint control'
        ),

        # Publishes robot TF tree from URDF + joint states
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

        # GUI sliders to set joint positions
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            condition=IfCondition(use_gui)
        ),

        # Fallback headless joint state publisher when gui:=false
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            condition=IfCondition(
                LaunchConfiguration('gui', default='true')
            )
        ),

        # RViz2 with robot model display
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_path],
            output='screen'
        ),
    ])
