from setuptools import setup
import os
from glob import glob

package_name = 'diffbot_description'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf',   glob('urdf/*.urdf')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz',   glob('rviz/*.rviz')),
    ],
    entry_points={
        'console_scripts': [],
    },
)
