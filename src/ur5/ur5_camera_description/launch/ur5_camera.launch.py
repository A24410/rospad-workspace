from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    urdf_path = os.path.join(
        os.path.dirname(__file__), '..', 'urdf', 'ur5_camera.urdf'
    )
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
    ])
