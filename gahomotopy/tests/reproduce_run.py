"""Reproduce a GA experiment run from saved result files.

Loads the JSON metadata and params.npy from a previous run, reconstructs
the homotopy planner with the same parameters, re-runs the path tracking,
and compares the resulting path with the saved one.

Since the homotopy continuation method is deterministic, re-running with
the same parameters should produce identical results.

Usage:
    python -m gahomotopy.tests.reproduce_run results/GAT/obstacles9

    # Or with full path to the base name (without extension):
    python -m gahomotopy.tests.reproduce_run results/GAT/obstacles9data.json

Run with:
    python -m gahomotopy.tests.reproduce_run
"""

import json
import os
import sys

import numpy as np

from gahomotopy.planning.homotopy import HomotopyPathPlanner
from gahomotopy.planning.experiments import LoadScenario
from gahomotopy.planning.matrix_modules import (
    DiagonalDominantMatrix,
    FullMatrix,
    PerRowMatrix,
)

MODULE_MAP = {
    "diagonal_dominant": DiagonalDominantMatrix,
    "full_matrix": FullMatrix,
    "per_row": PerRowMatrix,
}

ARM_MAP = {
    "UR3E": "gahomotopy.kinematics.ur3e",
    "ROARM3DOF": "gahomotopy.kinematics.roarm",
}


def load_run(base_path):
    """Load all result files for a run.

    Args:
        base_path: Path to the run files without extension, or path to
                   the data.json file. E.g. "results/GAT/obstacles9"
                   or "results/GAT/obstacles9data.json"

    Returns:
        dict with keys: data, params, path
    """
    if base_path.endswith('.json'):
        base_path = base_path.replace('data.json', '')

    # Find the data.json (base_path may or may not include "data")
    json_path = base_path if base_path.endswith('data.json') else base_path + 'data.json'
    if not os.path.exists(json_path):
        json_path = base_path + '.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    # The .npy files share the base name without "data"
    npy_base = base_path
    if npy_base.endswith('data'):
        npy_base = npy_base[:-4]

    params = np.load(npy_base + 'params.npy')
    path = np.load(npy_base + '.npy')

    return {'data': data, 'params': params, 'path': path}


def parse_params(params, num_var, num_obs, module_name):
    """Parse the params vector into radius, matrix genes, obsVal, obsSign.

    The layout depends on the matrix module:
      diagonal_dominant: [radius, offDiag, diag, obsVal..., obsSign...]
      full_matrix:       [radius, n*n genes, obsVal..., obsSign...]
      per_row:           [radius, 2n genes, obsVal..., obsSign...]
    """
    if module_name == "diagonal_dominant":
        n_matrix_genes = 2
    elif module_name == "full_matrix":
        n_matrix_genes = num_var * num_var
    elif module_name == "per_row":
        n_matrix_genes = 2 * num_var
    else:
        raise ValueError(f"Unknown matrix module: {module_name}")

    radius = params[0]
    matrix_genes = params[1:1 + n_matrix_genes]
    obs_val = params[1 + n_matrix_genes:1 + n_matrix_genes + num_obs]
    obs_sign = params[1 + n_matrix_genes + num_obs:]

    return radius, matrix_genes, obs_val, obs_sign


def reproduce(base_path):
    """Reproduce a run and compare paths.

    Args:
        base_path: Path to the run files (see load_run)
    """
    run = load_run(base_path)
    data = run['data']
    params = run['params']
    saved_path = run['path']

    # Parse metadata from JSON
    # [0]=header, [1]=name, [2]=run, [3]=time, [4]=armType,
    # [5]=matrixModule, [6]=startPos, [7]=goalPos, [8]=failed, [9]=pathDistance
    name = data[1]
    arm_type = data[4]
    module_name = data[5]
    start = tuple(data[6])
    goal = tuple(data[7])
    saved_failed = data[8]
    saved_distance = data[9]

    print(f"=== Reproducing run: {name} ===")
    print(f"  armType:       {arm_type}")
    print(f"  matrixModule:  {module_name}")
    print(f"  startPos:      {start}")
    print(f"  goalPos:       {goal}")
    print(f"  saved failed:  {saved_failed}")
    print(f"  saved distance: {saved_distance}")
    print()

    # Load arm
    if arm_type not in ARM_MAP:
        raise ValueError(f"Unknown arm type: {arm_type}")
    import importlib
    arm_module = importlib.import_module(ARM_MAP[arm_type])
    arm_class = getattr(arm_module, arm_type)
    arm = arm_class()

    # Load scenario to get obstacles
    # The name may have a run suffix like "obstacles9-0", strip it
    scenario_name = name.rsplit('-', 1)[0] if '-' in name else name
    obstacles, _, _, _ = LoadScenario(scenario_name)
    arm.setObstaclesPos(obstacles)

    num_var = len(start)
    num_obs = len(obstacles)

    # Parse params
    radius, matrix_genes, obs_val, obs_sign = parse_params(
        params, num_var, num_obs, module_name
    )

    print(f"  radius:        {radius}")
    print(f"  numObs:        {num_obs}")
    print(f"  numVar:        {num_var}")
    print(f"  matrix genes:  {matrix_genes}")
    print()

    # Rebuild matrix
    module_cls = MODULE_MAP[module_name]
    module = module_cls()
    module._num_var = num_var

    # Reconstruct a solution vector so build_matrix can extract genes
    # build_matrix expects genes at the end of the solution
    solution = np.zeros(1 + num_obs * 2 + len(matrix_genes))
    solution[0] = radius
    solution[1:1 + num_obs] = obs_val
    solution[1 + num_obs:1 + 2 * num_obs] = obs_sign
    solution[-len(matrix_genes):] = matrix_genes

    a_matrix = module.build_matrix(solution, num_var)
    print(f"  Reconstructed matrix:\n{a_matrix}")
    print()

    # Re-run homotopy
    print("  Re-running homotopy path tracking...")
    planner = HomotopyPathPlanner(
        radius, obstacles, start, goal,
        obs_val, obs_sign, a_matrix, arm
    )

    new_path, final_lambda, new_failed, new_distance, lambdas = planner.track_path_multi(
        max_steps=50000
    )

    print()
    print(f"  Original path shape:  {saved_path.shape}")
    print(f"  New path shape:       {new_path.shape}")
    print(f"  Original failed:      {saved_failed}")
    print(f"  New failed:           {new_failed}")
    print(f"  Original distance:    {saved_distance}")
    print(f"  New distance:         {new_distance}")

    # Compare paths
    if saved_path.shape == new_path.shape:
        path_diff = np.max(np.abs(saved_path - new_path))
        print(f"  Max path difference:  {path_diff}")
        if path_diff < 1e-6:
            print("\n  ✅ Paths match — reproduction successful!")
        else:
            print("\n  ⚠️  Paths differ — non-deterministic behavior detected")
    else:
        print(f"\n  ⚠️  Path shapes differ ({saved_path.shape} vs {new_path.shape})")

    return {
        'path': new_path,
        'failed': new_failed,
        'distance': new_distance,
        'final_lambda': final_lambda,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        reproduce(sys.argv[1])
    else:
        # Default to the test run
        reproduce("results/GAT/obstacles9")