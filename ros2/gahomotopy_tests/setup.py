from setuptools import find_packages, setup

package_name = 'gahomotopy_tests'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Guillermo Alfredo García Manjarrez',
    maintainer_email='zS24019403@estudiantes.uv.mx',
    description='ROS 2 test and movement utility nodes for the UR3e',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # Utility nodes will be added here as needed
        ],
    },
)