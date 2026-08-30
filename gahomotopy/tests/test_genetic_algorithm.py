"""Test script for the genetic algorithm parameter optimization.

Runs the GA with obstacle scenes and saves results. This is the
testing entry point for GA experiments — experiments.py no longer
contains GA running code.

Run with:
    python -m gahomotopy.tests.test_genetic_algorithm
"""

import os
from queue import Full
import time
import json

import numpy as np

from gahomotopy.planning.genetic_algorithm import GeneticAlgorithm
from gahomotopy.planning.matrix_modules import DiagonalDominantMatrix,FullMatrix,PerRowMatrix
from gahomotopy.kinematics.ur3e import UR3E
from gahomotopy.kinematics.roarm import ROARM3DOF
from gahomotopy.planning.experiments import (
    LoadScenario, save_array, save_to_json,
)


def setUpGA(obstacles, start, goal, name="obstacles", run=0):
    """Run a single GA optimization and return path, params, and data."""
    maxNumRadius = 20000 #20000
    numGenerations = 15 #15
    popSize = 20 #20
    numParents = 10 #10

    arm = ROARM3DOF()
    arm.setObstaclesPos(obstacles)

    ga = GeneticAlgorithm(maxNumRadius, numGenerations, popSize, numParents,
                          obstacles, start, goal, arm,
                          matrix_module=FullMatrix(),
                          name=name,
                          parallel_processing=["process", 10])

    data = []
    start_time = time.time()
    path, homotopyParams, finalLambda, failed, dis, lambdas, fitnessEvolution = ga.optimize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    data.append("contents: name,run,time,armType,matrixModule,startPos,goalPos,failed,pathDistance,fitnessEvolution")
    data.append(name)
    data.append(run)
    data.append(elapsed_time)
    data.append(type(arm).__name__)
    data.append(ga.matrix_module.name)
    data.append(start)
    data.append(goal)
    data.append(failed)
    data.append(dis)
    data.append(fitnessEvolution)

    return path, homotopyParams, data, fitnessEvolution


def setUpGAMulti():
    """Run GA on multiple obstacle scenes."""
    #instances = ["obstacles3","obstacles4", "obstacles5","obstacles8"]
    instances = ["obstacles1", "obstacles3", "obstacles5", "obstacles6", "obstacles8"]
    runsPerInstance = 30
    folder = "results/GAFMRoarm/"

    for scenario_name in instances:
        for n in range(runsPerInstance):
            obstacles, start, goal, name = LoadScenario(scenario_name,robot="roarm")
            name = name + "-" + str(n)
            path, homotopyParams, data, fitnessEvolution = setUpGA(obstacles, start, goal, name, run=n)
            save_array(path, folder + name + ".npy")
            save_array(homotopyParams, folder + name + "params.npy")
            save_array(np.array(fitnessEvolution), folder + name + "fitnessEvolution.npy")
            save_to_json(data, folder + name + "data")


def setUpGASimple():
    """Run GA on a single obstacle scene for quick testing."""
    folder = "results/GAT/"
    os.makedirs(folder, exist_ok=True)
    obstacles, start, goal, name = LoadScenario("obstacles3",robot="roarm")
    path, homotopyParams, data, fitnessEvolution = setUpGA(obstacles, start, goal, name)
    save_array(path, folder + name + ".npy")
    save_array(homotopyParams, folder + name + "params.npy")
    save_array(np.array(fitnessEvolution), folder + name + "fitnessEvolution.npy")
    save_to_json(data, folder + name + "data")


if __name__ == "__main__":
    setUpGAMulti()
    #setUpGASimple()
