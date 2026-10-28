from setuptools import setup

package_name = 'diffbot_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'color_tracker = diffbot_vision.color_tracker:main',
        ],
    },
)
