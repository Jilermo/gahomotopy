"""Tests and demos for the homotopy path planner.

These scripts exercise the planner with hardcoded default parameters and
obstacle scenes. They are not unit tests — they are integration demos that
produce trajectory output files under results/.

Run with:
    python -m gahomotopy.tests.test_homotopy
"""

import os
import time
import json

import numpy as np

from gahomotopy.kinematics.ur3e import UR3E
from gahomotopy.planning.homotopy import HomotopyPathPlanner
from gahomotopy.planning.experiments import LoadScenario


def save_to_json(data, filename):
    """Saves a Python object (even with NumPy data) to a JSON file."""
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Successfully saved to {filename}")


def savePath(vector, filename):
    np.save(filename, vector)
    print(f"Vector saved to {filename}")


def buildParamsVector(radius, obsVal, ObsSign):
    numObs = len(obsVal)
    params = np.zeros((numObs * 2) + 1)
    params[0] = radius
    for i in range(numObs):
        params[i + 1] = obsVal[i]
    for i in range(numObs):
        params[i + numObs + 1] = ObsSign[i]
    return params


def createDataForJSON(name, failed, elapsedTime, finalDistance, signs, magnitudes, radius):
    data = []
    data.append("name,failed,elapsedTime,finalDistance,signs,magnitudes,radius")
    data.append(name)
    data.append(failed)
    data.append(elapsedTime)
    data.append(finalDistance)
    data.append(signs)
    data.append(magnitudes)
    data.append(radius)
    return data


def TrackPath(start, goal, obstacles, name):
    """Run the homotopy planner with default parameters and save results.

    This is a convenience wrapper with hardcoded defaults:
    - obsVal = 30000 for all obstacles
    - obsSign = -1.0 for all obstacles
    - maxNumRadius = 25000, radiuses = 0.05
    - a_matrix: diagonally dominant matrix built from offDiagVals=20, diagVals=20
    """
    numObs = len(obstacles)

    obsSign = np.zeros(numObs)
    for i in range(numObs):
        obsSign[i] = -1.0

    obsVal = np.zeros(numObs)
    for i in range(numObs):
        obsVal[i] = 30000

    maxNumRadius = 25
    radiuses = 0.05

    # Build dominant matrix externally (test different methods here)
    dof = len(start)
    offDiagVals = 20
    diagVals = 20
    a_matrix = np.zeros((dof, dof))
    for i in range(dof):
        for j in range(dof):
            if i == j:
                a_matrix[i, j] = (dof * abs(offDiagVals)) + diagVals
            else:
                a_matrix[i, j] = offDiagVals if (i+j) % 2 == 0 else -offDiagVals

    arm = UR3E()
    arm.setObstaclesPos(obstacles)

    planner = HomotopyPathPlanner(
        radiuses, obstacles, start, goal,
        obsVal, obsSign, a_matrix, arm
    )

    print("Starting homotopy path tracking...")
    start_time = time.time()

    path, finalLambda, failed, dis, lambdas = planner.track_path_multi(
        max_steps=maxNumRadius
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Distance to obstacles calculated in {elapsed_time:.4f} seconds")

    print(f"Path found with {len(path)} steps")
    print(f"Final angles: ")
    print(path[-1, :])
    print(f"Final distance: {dis:.4f}")

    cartesianPath = np.zeros((len(path), 3))
    index = 0
    for i in path:
        cartesianPos = arm.directKinematics(i)
        cartesianPath[index][0] = cartesianPos[0, 3]
        cartesianPath[index][1] = cartesianPos[1, 3]
        cartesianPath[index][2] = cartesianPos[2, 3]
        index += 1

    folder = "results/Manual/"
    os.makedirs(folder, exist_ok=True)
    savePath(cartesianPath, folder + name + "cartesianPath")
    savePath(lambdas, folder + name + "lambdas")
    savePath(path, folder + name)
    homotopypParams = buildParamsVector(radiuses, obsVal, obsSign)
    savePath(homotopypParams, folder + name + "params")
    data = createDataForJSON(
        name, failed, elapsed_time, dis,
        obsSign.tolist(), obsVal.tolist(), radiuses
    )
    save_to_json(data, folder + name + "data")
    return path


if __name__ == "__main__":
    obstacles, start, goal, name = LoadScenario("obstacles4")
    TrackPath(start, goal, obstacles, name)
