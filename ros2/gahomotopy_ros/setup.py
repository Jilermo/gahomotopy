from setuptools import find_packages, setup

package_name = 'gahomotopy_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/default.yaml',
            'config/test_quick.yaml',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/ga_planner.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Guillermo Alfredo García Manjarrez',
    maintainer_email='zS24019403@estudiantes.uv.mx',
    description='ROS 2 nodes for homotopy path planning with genetic algorithm optimization',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'ga_planner_node = gahomotopy_ros.ga_planner_node:main',
            'joint_broadcaster_node = gahomotopy_ros.joint_broadcaster_node:main',
            'plan_client = gahomotopy_ros.plan_client:main',
        ],
    },
)