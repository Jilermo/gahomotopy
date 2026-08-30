#!/usr/bin/env python3
"""
Batch experiment runner for GA-optimized homotopy path planning.

Runs multiple GA experiments across obstacle scenarios, matrix modules, and
arm types with configurable run counts. Designed for long-running server use.

Features:
  - Iterates scenarios × modules × runs
  - Structured output directory: <output_dir>/<module>/<scenario>-<run>.*
  - Skips already-completed runs (resume capability)
  - Saves: path (.npy), params (.npy), fitnessEvolution (.npy), data (.json)
  - Parallel GA workers via --workers

Usage (on the server):
    cd ~/gahomotopy_ws
    source .venv/bin/activate

    # Run all 4 scenarios, per_row module, 30 runs, 10 parallel workers
    python -m gahomotopy.planning.batch_experiments \
        --scenarios obstacles3 obstacles4 obstacles5 obstacles8 \
        --module per_row \
        --runs 30 \
        --workers 10 \
        --output-dir results/GARow

    # Or run via the script directly:
    python scripts/batch_experiments.py --scenarios obstacles3 --runs 5 --workers 4

Output files per run (matching the existing GARow naming convention):
    <output_dir>/<scenario>-<run>data.json
    <output_dir>/<scenario>-<run>.npy            (path)
    <output_dir>/<scenario>-<run>params.npy
    <output_dir>/<scenario>-<run>fitnessEvolution.npy

The data.json format is a list:
    [header, name, run, time, armType, matrixModule,
     startPos, goalPos, failed, pathDistance, fitnessEvolution]
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from gahomotopy.kinematics.ur3e import UR3E
from gahomotopy.planning.genetic_algorithm import GeneticAlgorithm
from gahomotopy.planning.experiments import LoadScenario, save_to_json, NumpyEncoder


def run_single_experiment(scenario_name, run_idx, matrix_module_name,
                          output_dir, ga_config):
    """Run a single GA experiment and save results.

    Returns True if the experiment completed (success or failure),
    False if it was skipped (already exists).
    """
    # File paths for this run
    name = f"{scenario_name}-{run_idx}"
    data_path = output_dir / f"{name}data.json"

    # Skip if already completed
    if data_path.exists():
        print(f"[SKIP] {name} — already exists")
        return False

    # Load scenario
    obstacles, start, goal, _ = LoadScenario(scenario_name, robot="ur3e")
    start = tuple(start)
    goal = tuple(goal)

    # Create arm
    arm = UR3E()

    # Create GA
    ga = GeneticAlgorithm(
        maxNumberOfRadius=ga_config['max_radius'],
        numGenerations=ga_config['num_generations'],
        populationSize=ga_config['population_size'],
        numParentsMating=ga_config['num_parents_mating'],
        obstacles=obstacles,
        start=start,
        goal=goal,
        arm=arm,
        matrix_module=matrix_module_name,
        name=name,
        parallel_processing=["process", ga_config['workers']] if ga_config['workers'] > 1 else None,
    )

    print(f"\n{'='*60}")
    print(f"[RUN] {name} — module={matrix_module_name}")
    print(f"      start={start}")
    print(f"      goal={goal}")
    print(f"      obstacles={len(obstacles)}")
    print(f"      workers={ga_config['workers']}")
    print(f"{'='*60}")

    t_start = time.time()

    try:
        path, homotopyParams, finalLambda, failed, dis, lambdas, fitness_evolution = ga.optimize()
    except Exception as e:
        print(f"[ERROR] {name} — {e}")
        traceback.print_exc()
        # Save a failure record
        elapsed = time.time() - t_start
        data_record = [
            "contents: name,run,time,armType,matrixModule,startPos,goalPos,failed,pathDistance,fitnessEvolution",
            name,
            run_idx,
            elapsed,
            "UR3E",
            matrix_module_name,
            list(start),
            list(goal),
            True,  # failed
            None,  # no path distance
            [],    # no fitness evolution
        ]
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data_record, f, indent=4, cls=NumpyEncoder)
        return True

    elapsed = time.time() - t_start

    # Save results
    np.save(output_dir / f"{name}.npy", np.array(path))
    np.save(output_dir / f"{name}params.npy", np.array(homotopyParams))
    np.save(output_dir / f"{name}fitnessEvolution.npy", np.array(fitness_evolution))

    # Build data.json (matching existing format)
    data_record = [
        "contents: name,run,time,armType,matrixModule,startPos,goalPos,failed,pathDistance,fitnessEvolution",
        name,
        run_idx,
        elapsed,
        "UR3E",
        matrix_module_name,
        list(start),
        list(goal),
        failed,
        dis,
        fitness_evolution,
    ]
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data_record, f, indent=4, cls=NumpyEncoder)

    status = "SUCCESS" if not failed else "FAILED"
    print(f"[{status}] {name} — time={elapsed:.1f}s, distance={dis:.6f}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch GA homotopy experiment runner for server use."
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=["obstacles3", "obstacles4", "obstacles5", "obstacles8"],
        help="Scenario names to run (default: obstacles3 obstacles4 obstacles5 obstacles8)"
    )
    parser.add_argument(
        "--module", default="per_row",
        choices=["diagonal_dominant", "full_matrix", "per_row"],
        help="Matrix module to use (default: per_row)"
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of GA runs per scenario (default: 30)"
    )
    parser.add_argument(
        "--start-run", type=int, default=0,
        help="Starting run index (default: 0, use to resume a partial batch)"
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of parallel GA workers per run (default: 10, set 1 for sequential)"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--generations", type=int, default=15,
        help="GA generations (default: 15)"
    )
    parser.add_argument(
        "--population", type=int, default=20,
        help="GA population size (default: 20)"
    )
    parser.add_argument(
        "--parents", type=int, default=10,
        help="GA parents mating (default: 10)"
    )
    parser.add_argument(
        "--max-radius", type=int, default=200,
        help="Max homotopy steps (default: 200)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ga_config = {
        'max_radius': args.max_radius,
        'num_generations': args.generations,
        'population_size': args.population,
        'num_parents_mating': args.parents,
        'workers': args.workers,
    }

    print(f"Batch experiment configuration:")
    print(f"  Scenarios:      {args.scenarios}")
    print(f"  Matrix module:  {args.module}")
    print(f"  Runs per scenario: {args.runs}")
    print(f"  Run index range:   {args.start_run} to {args.start_run + args.runs - 1}")
    print(f"  GA generations:    {args.generations}")
    print(f"  GA population:     {args.population}")
    print(f"  GA parents mating: {args.parents}")
    print(f"  Parallel workers:  {args.workers}")
    print(f"  Output dir:        {output_dir}")
    print()

    total = len(args.scenarios) * args.runs
    completed = 0
    skipped = 0
    failed = 0

    t_total_start = time.time()

    for scenario in args.scenarios:
        for run_idx in range(args.start_run, args.start_run + args.runs):
            try:
                did_run = run_single_experiment(
                    scenario, run_idx, args.module, output_dir, ga_config
                )
                if did_run:
                    completed += 1
                    # Check if this run failed
                    data_path = output_dir / f"{scenario}-{run_idx}data.json"
                    if data_path.exists():
                        with open(data_path) as f:
                            rec = json.load(f)
                        if rec[8]:  # failed field
                            failed += 1
                else:
                    skipped += 1
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] by user")
                _print_summary(completed, skipped, failed, total, t_total_start)
                sys.exit(130)
            except Exception as e:
                print(f"[FATAL] {scenario}-{run_idx}: {e}")
                traceback.print_exc()
                failed += 1
                completed += 1

    _print_summary(completed, skipped, failed, total, t_total_start)


def _print_summary(completed, skipped, failed, total, t_start):
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"  Total runs:      {total}")
    print(f"  Completed:       {completed} ({failed} failed)")
    print(f"  Skipped (exist): {skipped}")
    print(f"  Elapsed:         {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print()


if __name__ == "__main__":
    main()