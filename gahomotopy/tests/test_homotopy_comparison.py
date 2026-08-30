"""Deterministic homotopy comparison test.

Uses fixed parameters (no GA randomness) to verify the new homotopy
implementation produces the same path as the old one for the same inputs.

Run on the NEW codebase:
    python -m gahomotopy.tests.test_homotopy_comparison

Run on the OLD codebase (eureka):
    python test_homotopy_comparison.py
"""

import numpy as np
import json
import os
import time


def run_test_new():
    """Run on the new codebase."""
    from gahomotopy.kinematics.ur3e import UR3E
    from gahomotopy.planning.homotopy import HomotopyPathPlanner
    from gahomotopy.planning.experiments import LoadScenario

    obstacles, start, goal, name = LoadScenario("obstacles4")

    # Use the same hardcoded parameters from the old TrackPath demo
    numObs = len(obstacles)

    obsSign = np.array([
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,  1.0,  1.0, -1.0,
        -1.0, -1.0,
    ])

    obsVal = np.array([
        37240.29178306825, 31928.160400953093, 88231.88752014069,
        77594.01311532124, 18491.24863248509, 31194.72641289492,
        2515.109581969886, 59773.783507364125, 24945.105056025634,
        75434.98201980251, 30295.1444600053, 44524.00496485376,
        47787.24283197837, 67885.3609938703,
    ])

    radius = 0.10443143123372077
    maxNumRadius = 25000

    # Build a simple diagonally dominant matrix using fixed off-diag and diag values
    # The old TrackPath used single values: offDiagVals=-34.70902699467313, DiagVals=38.05719379814289
    # In the old code, these were scalars applied to ALL off-diagonal/diagonal entries
    dof = 6
    offDiagScalar = -34.70902699467313
    diagScalar = 38.05719379814289

    # Build matrix the way the old generate_dominant_a_matrix does with scalar values
    a_matrix = np.zeros((dof, dof))
    for i in range(dof):
        for j in range(dof):
            if i != j:
                a_matrix[i, j] = offDiagScalar
    for i in range(dof):
        a_matrix[i, i] = np.sum(np.abs(a_matrix[i, :])) + diagScalar

    arm = UR3E()
    arm.setObstaclesPos(obstacles)

    planner = HomotopyPathPlanner(
        radius, obstacles, start, goal,
        obsVal, obsSign, a_matrix, arm
    )

    print("Starting homotopy path tracking (NEW code)...")
    start_time = time.time()
    path, finalLambda, failed, dis, lambdas = planner.track_path_multi(max_steps=maxNumRadius)
    elapsed = time.time() - start_time

    print(f"\n=== NEW CODE RESULTS ===")
    print(f"Time: {elapsed:.2f}s")
    print(f"Path length: {len(path)} steps")
    print(f"Final lambda: {finalLambda}")
    print(f"Failed: {failed}")
    print(f"Distance: {dis}")
    print(f"Final point: {path[-1]}")

    return path, finalLambda, failed, dis


def run_test_old():
    """Run on the old codebase (eureka)."""
    import sys
    import pathlib

    current_dir = str(pathlib.Path(__file__).parent.resolve())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from ur3eRobotArm import UR3E
    from SphericalAlgorithmMulti import HomotopyPathPlanner
    import callHomotopy as cH

    obstacles, start, goal, name = cH.setUpObstacles4()

    numObs = len(obstacles)

    obsSign = np.array([
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,  1.0,  1.0, -1.0,
        -1.0, -1.0,
    ])

    obsVal = np.array([
        37240.29178306825, 31928.160400953093, 88231.88752014069,
        77594.01311532124, 18491.24863248509, 31194.72641289492,
        2515.109581969886, 59773.783507364125, 24945.105056025634,
        75434.98201980251, 30295.1444600053, 44524.00496485376,
        47787.24283197837, 67885.3609938703,
    ])

    radius = 0.10443143123372077
    maxNumRadius = 25000

    offDiagScalar = -34.70902699467313
    diagScalar = 38.05719379814289

    dof = 6
    offDiagVals = np.full(dof * (dof - 1), offDiagScalar)
    diagVals = np.full(dof, diagScalar)

    arm = UR3E()
    arm.setObstaclesPos(obstacles)

    planner = HomotopyPathPlanner(
        maxNumRadius, radius, obstacles, start, goal,
        obsVal, obsSign, offDiagVals, diagVals, arm,
        False, False, 2
    )

    print("Starting homotopy path tracking (OLD code)...")
    start_time = time.time()
    path, finalLambda, failed, dis, lambdas, radiusesUsed = planner.track_path_multi(max_steps=maxNumRadius)
    elapsed = time.time() - start_time

    print(f"\n=== OLD CODE RESULTS ===")
    print(f"Time: {elapsed:.2f}s")
    print(f"Path length: {len(path)} steps")
    print(f"Final lambda: {finalLambda}")
    print(f"Failed: {failed}")
    print(f"Distance: {dis}")
    print(f"Final point: {path[-1]}")

    return path, finalLambda, failed, dis


if __name__ == "__main__":
    # Detect which codebase we're in
    try:
        from gahomotopy.planning.homotopy import HomotopyPathPlanner
        print("Running on NEW codebase (gahomotopy)")
        path, finalLambda, failed, dis = run_test_new()
    except ImportError:
        print("Running on OLD codebase (ur_wsMatrixGenetic)")
        path, finalLambda, failed, dis = run_test_old()

    # Save results for comparison
    np.save("homotopy_test_path.npy", path)
    print(f"\nPath saved to homotopy_test_path.npy")
    print(f"Path shape: {path.shape}")