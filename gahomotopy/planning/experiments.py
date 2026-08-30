"""Scenario loading and utility functions for homotopy path planning experiments.

Test scenarios are stored as JSON files under TestScenarios/<robot>/.
Each file contains obstacles, start, and goal configurations.

Usage:
    from gahomotopy.planning.experiments import LoadScenario
    obstacles, start, goal, name = LoadScenario("obstacles4")
    obstacles, start, goal, name = LoadScenario("obstacles4", robot="ur3e")
"""

import json
import os
import time

import numpy as np


# Path to the TestScenarios directory (relative to this file)
_SCENARIO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "TestScenarios"
)


def save_array(data, filename="my_array.npy"):
    """Saves a numpy array to a binary file."""
    try:
        np.save(filename, data)
        print(f"Successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")


def load_array(filename="my_array.npy"):
    """Reads a numpy array from a binary file."""
    try:
        data = np.load(filename)
        print(f"Successfully loaded {filename}")
        return data
    except FileNotFoundError:
        print("File not found. Please check the path.")
        return None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.floating, np.integer)):
            return obj.tolist()
        return super().default(obj)


def save_to_json(data, filename):
    """Saves a Python object (even with NumPy data) to a JSON file."""
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
    print(f"Successfully saved to {filename}")


def LoadScenario(name, robot="ur3e"):
    """Load a test scenario from a JSON file.

    Args:
        name: Scenario name (e.g. "obstacles4" or "obstacles4.json")
        robot: Robot type subfolder ("ur3e" or "roarm")

    Returns:
        obstacles: list of {'center': (x, y, z), 'radius': r}
        start: tuple of joint angles
        goal: tuple of joint angles
        name: scenario name string
    """
    if not name.endswith('.json'):
        name = name + '.json'

    filepath = os.path.join(_SCENARIO_ROOT, robot, name)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert center lists back to tuples (obstacle functions returned tuples)
    obstacles = []
    for obs in data['obstacles']:
        obstacles.append({
            'center': tuple(obs['center']),
            'radius': obs['radius']
        })

    start = tuple(data['start'])
    goal = tuple(data['goal'])
    scenario_name = name.replace('.json', '')

    return obstacles, start, goal, scenario_name


def list_scenarios(robot="ur3e"):
    """List all available scenarios for a given robot type."""
    folder = os.path.join(_SCENARIO_ROOT, robot)
    if not os.path.isdir(folder):
        return []
    return sorted([
        f.replace('.json', '')
        for f in os.listdir(folder)
        if f.endswith('.json')
    ])