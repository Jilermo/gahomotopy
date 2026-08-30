# gahomotopy — Homotopy Path Planning with Genetic Algorithm Optimization

ROS 2 package for collision-free trajectory planning on robotic arms using the
Enhanced Homotopy Path Planning Method (EHPPM) optimized by a genetic algorithm.

## Overview

This package implements:

- **EHPPM** — Enhanced Homotopy Path Planning Method using spherical continuation
- **Genetic Algorithm** — Optimizes EHPPM parameters (radius, obstacle repulsion,
  matrix A) to minimize trajectory length in configuration space
- **ROS 2 nodes** — Service-based interface for planning trajectories and
  broadcasting joint states

Supported robots:

- **RoArm M2-S** (3 DoF) — Waveshare robotic arm
- **UR3e** (6 DoF) — Universal Robots

## Requirements

- Ubuntu 24.04
- ROS 2 Jazzy Jalisco
- Python 3.12
- NumPy, SciPy, PyGAD

## Repository Structure

```
gahomotopy/
├── gahomotopy/                 # Core Python package (no ROS dependency)
│   ├── kinematics/             #   Forward kinematics (RoArm M2, UR3e)
│   ├── planning/               #   EHPPM + Genetic Algorithm
│   │   └── matrix_modules/     #     Matrix A construction strategies
│   ├── analysis/               #   Result analysis utilities
│   ├── visualization/          #   Arm visualization
│   └── tests/                  #   Unit tests
├── ros2/                       # ROS 2 packages
│   ├── gahomotopy_msgs/        #   Custom messages and services
│   ├── gahomotopy_ros/         #   Planning and broadcasting nodes
│   └── gahomotopy_tests/       #   Test/movement utility nodes
├── scripts/                    # Helper scripts (obstacle creation, batch runs)
├── TestScenarios/              # JSON scenario definitions (obstacles, start, goal)
└── gahomotopy.repos            # External dependency manifest
```

## Installation

### 1. Install ROS 2 Jazzy

Follow the official instructions: https://docs.ros.org/en/jazzy/Installation.html

### 2. Create a workspace and clone this repo

```bash
mkdir -p ~/gahomotopy_ws/src
cd ~/gahomotopy_ws/src
git clone https://github.com/Jilermo/gahomotopy.git .
```

### 3. Install Python dependencies

```bash
pip install pygad numpy scipy
```

### 4. Install the core Python package

The `gahomotopy` core package is pure Python (no ROS dependency). Install it
so it is importable from the system Python:

```bash
pip install -e gahomotopy/
```

### 5. (Optional) Clone external robot drivers

If you need to visualize or control real hardware, clone the driver packages:

```bash
cd ~/gahomotopy_ws
vcs import src < gahomotopy.repos
```

Or clone manually:

**UR3e** (official driver + simulation):
```bash
git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git src/Universal_Robots_ROS2_Driver
```

**RoArm M2-S** (driver + URDF description):
```bash
git clone https://github.com/waveshareteam/roarm_ws_em0.git src/roarm_ws_em0
```

See the Waveshare wiki for setup instructions:
https://www.waveshare.com/wiki/RoArm-M2-S_2._ROS2_Workspace_Description

### 6. Build

```bash
cd ~/gahomotopy_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

### 7. Source the workspace

```bash
source install/setup.bash
```

## Usage

### Plan a trajectory

Launch the GA planner node with parameters:

```bash
ros2 launch gahomotopy_ros ga_planner.launch.py config:=ros2/gahomotopy_ros/config/default.yaml
```

Call the planning service with a scenario (start, goal, obstacles):

```bash
ros2 run gahomotopy_ros plan_client
```

Or call the service directly:

```bash
ros2 service call /plan_trajectory gahomotopy_msgs/srv/PlanTrajectory \
  "{start: {x: 32.82, y: 14.52, z: -17.44},
    goal: {x: -32.82, y: 14.52, z: -17.44},
    obstacles: [{x: 23.015, y: 28.878, z: 0.0, radius: 6.0}],
    experiment_name: 'test'}"
```

The optimized trajectory is saved as a `.npy` file in `results/`.

### Broadcast a trajectory

Publish the trajectory as `sensor_msgs/JointState` messages:

```bash
ros2 run gahomotopy_ros joint_broadcaster_node \
  --ros-args \
  -p trajectory_file:="results/test.npy" \
  -p dt:=0.01 \
  -p robot_type:="roarm_m2"
```

### Configuration

GA parameters and EHPPM search ranges are configured via YAML. See
`ros2/gahomotopy_ros/config/default.yaml` for all available parameters.

## Nodes

| Node | Description |
|------|-------------|
| `ga_planner_node` | Runs the genetic algorithm to optimize EHPPM parameters and returns an optimized trajectory via the `/plan_trajectory` service |
| `joint_broadcaster_node` | Reads a `.npy` trajectory file and publishes `JointState` messages at a configurable rate |

## Messages and Services

| Type | Name | Description |
|------|------|-------------|
| `msg` | `Obstacle` | Obstacle definition (center + radius) |
| `msg` | `TrajectoryResult` | Optimized trajectory with parameters and metadata |
| `srv` | `PlanTrajectory` | Request trajectory planning (scenario → trajectory) |
| `srv` | `EvaluatePath` | Evaluate a single EHPPM configuration (params → fitness) |

## License

MIT — See [LICENSE](LICENSE) for full text.

## Author

**Guillermo Alfredo García Manjarrez**
zS24019403@estudiantes.uv.mx
Maestría en Inteligencia Artificial, IIIA — Universidad Veracruzana