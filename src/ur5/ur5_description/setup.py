from setuptools import setup
import os
from glob import glob

package_name = 'ur5_description'

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf') + glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'meshes', 'visual'),
            glob('meshes/visual/*')),
        (os.path.join('share', package_name, 'meshes', 'collision'),
            glob('meshes/collision/*')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nirav',
    maintainer_email='nirav.robotics@gmail.com',
    description='UR5 robot description for ROSpad simulation',
    license='BSD-3-Clause',
    entry_points={},
)
