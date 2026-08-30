"""Quick single GA run test for scenario 4 with FullMatrix.

Runs ONE GA optimization on scenario 4 and reports success/failure.
This is a quick sanity check — not a full 30-run benchmark.

Run with:
    python -m gahomotopy.tests.test_single_ga
"""

import time
import numpy as np

from gahomotopy.planning.genetic_algorithm import GeneticAlgorithm
from gahomotopy.planning.matrix_modules import FullMatrix
from gahomotopy.kinematics.ur3e import UR3E
from gahomotopy.planning.experiments import LoadScenario


def run_single_ga():
    """Run a single GA optimization on scenario 4."""
    maxNumRadius = 25000
    numGenerations = 15
    popSize = 20
    numParents = 10

    obstacles, start, goal, name = LoadScenario("obstacles4")

    arm = UR3E()
    arm.setObstaclesPos(obstacles)

    ga = GeneticAlgorithm(
        maxNumRadius, numGenerations, popSize, numParents,
        obstacles, start, goal, arm,
        matrix_module=FullMatrix(),
        name="obstacles4-test",
        parallel_processing=["process", 10],
    )

    print(f"Scenario: {name}")
    print(f"Obstacles: {len(obstacles)}")
    print(f"Num genes: {len(ga.gene_space)}")
    print(f"maxNumRadius: {maxNumRadius}")
    print(f"Generations: {numGenerations}, Pop: {popSize}, Parents: {numParents}")
    print("=" * 60)

    start_time = time.time()
    path, homotopyParams, finalLambda, failed, dis, lambdas, fitnessEvolution = ga.optimize()
    elapsed = time.time() - start_time

    print("=" * 60)
    print(f"\n=== GA RESULTS ===")
    print(f"Time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"Failed: {failed}")
    print(f"Distance: {dis}")
    print(f"Path length: {len(path)} steps")
    print(f"Final lambda: {finalLambda}")
    print(f"Fitness evolution: {fitnessEvolution}")

    if not failed:
        print("\n*** SUCCESS: GA found a valid path! ***")
    else:
        print("\n*** FAILED: GA did not find a valid path. ***")

    return failed, dis, elapsed


if __name__ == "__main__":
    run_single_ga()