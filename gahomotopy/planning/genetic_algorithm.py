import multiprocessing
import pygad
import numpy as np
from gahomotopy.planning.homotopy import HomotopyPathPlanner
from gahomotopy.planning.matrix_modules import MatrixModuleBase, DiagonalDominantMatrix, FullMatrix, PerRowMatrix


class GeneticAlgorithm:

    def __init__(self, maxNumberOfRadius, numGenerations, populationSize,
                 numParentsMating, obstacles, start, goal, arm,
                 matrix_module=None, name="default",
                 parallel_processing=None):
        self.gene_type = []
        self.gene_space = []

        self.num_generations = numGenerations
        self.num_parents_mating = numParentsMating
        self.sol_per_pop = populationSize

        self.maxNumRadius = maxNumberOfRadius

        self.obstacles = obstacles
        self.start = start
        self.goal = goal
        self.arm = arm
        self.name = name

        self.numObs = len(obstacles)
        self.numVar = len(start)

        # Parallel processing: None (sequential), int (threads), or
        # ["process", N] / ["thread", N] — passed directly to pygad.GA
        self.parallel_processing = parallel_processing

        # Matrix construction module (defaults to DiagonalDominantMatrix)
        if matrix_module is None:
            matrix_module = DiagonalDominantMatrix()
        elif isinstance(matrix_module, str):
            matrix_module = self._get_module_by_name(matrix_module)

        # Pull GA ranges from the arm class
        ranges = arm.ga_ranges
        self.matrix_module = matrix_module
        self.matrix_module._num_var = self.numVar

        # Configure matrix module ranges from arm
        if hasattr(self.matrix_module, 'off_diagonal'):
            self.matrix_module.off_diagonal = ranges['off_diagonal']
        if hasattr(self.matrix_module, 'diagonal'):
            self.matrix_module.diagonal = ranges['diagonal']

        # Gene layout (in order):
        #   1. radius (1 gene)
        #   2. obstacle values (numObs genes)
        #   3. obstacle signs (numObs genes)
        #   4. matrix params (defined by matrix_module)
        self._setup_radius(ranges['radius'])
        self._setup_obstacle_genes(ranges['obstacle_value'])
        self.matrix_module.setup_gene_space(self.gene_space, self.gene_type, self.numVar)

        # Extract per-gene low/high from gene_space for init_range.
        # pygad's SBX crossover uses init_range_low/high as gene bounds,
        # NOT gene_space. If these don't match, SBX produces complex values
        # (negative beta -> pow(negative, fractional) -> complex).
        self._init_range_low = []
        self._init_range_high = []
        for gs in self.gene_space:
            if isinstance(gs, dict):
                self._init_range_low.append(gs['low'])
                self._init_range_high.append(gs['high'])
            elif isinstance(gs, (list, tuple)):
                self._init_range_low.append(gs[0])
                self._init_range_high.append(gs[1])
            else:
                self._init_range_low.append(-4)
                self._init_range_high.append(4)

        self.distances = []
        self.fitness_evolution = []

        # Shared flag for early stopping across worker processes.
        # In process mode, this is a Manager proxy (picklable) so workers
        # can signal "valid path found" to the main process.
        self._failed_shared = None

    def _get_failed(self):
        """Read the failed flag (supports both shared and plain modes)."""
        if self._failed_shared is not None:
            return self._failed_shared.value
        return getattr(self, '_failed_local', True)

    def _set_failed(self, value):
        """Write the failed flag (supports both shared and plain modes)."""
        if self._failed_shared is not None:
            self._failed_shared.value = value
        else:
            self._failed_local = value

    def _get_module_by_name(self, name):
        """Look up a matrix module by name string."""
        from gahomotopy.planning.matrix_modules import DiagonalDominantMatrix, FullMatrix, PerRowMatrix
        modules = {
            "diagonal_dominant": DiagonalDominantMatrix,
            "full_matrix": FullMatrix,
            "per_row": PerRowMatrix,
        }
        if name not in modules:
            raise ValueError(f"Unknown matrix module: {name}. "
                             f"Available: {list(modules.keys())}")
        return modules[name]()

    def _setup_radius(self, radius_range):
        """Single radius gene."""
        self.gene_space.append(radius_range)
        self.gene_type.append(float)

    def _setup_obstacle_genes(self, obs_value_range):
        """Obstacle values and signs (no clustering)."""
        for _ in range(self.numObs):
            self.gene_space.append(obs_value_range)
            self.gene_type.append(float)

        for _ in range(self.numObs):
            self.gene_space.append([-1, 1])
            self.gene_type.append(int)

    def _parse_solution(self, solution):
        """Extract radius, obsVal, obsSign from solution vector.

        Gene layout:
          [0]                    = radius
          [1:numObs+1]           = obstacle values
          [numObs+1:2*numObs+1]  = obstacle signs
          [2*numObs+1:]          = matrix params (handled by matrix_module)
        """
        radius = solution[0]
        obsVal = solution[1:self.numObs + 1].astype(float)
        obsSign = solution[self.numObs + 1:2 * self.numObs + 1].astype(float)
        return radius, obsVal, obsSign

    def buildParamsVector(self, radius, obsVal, obsSign):
        """Build params vector using the matrix module's format."""
        # This is called externally — needs solution for matrix genes.
        # Kept for backward compatibility but delegates to matrix_module.
        return self.matrix_module.build_params_vector(
            self._last_solution, radius, obsVal, obsSign
        )

    def fitness_func(self, ga_instance, solution, solution_idx):
        """Fitness function — calls homotopy planner with current solution.

        This is a method (not a closure) so it can be pickled for
        multiprocessing. In parallel mode, each worker process receives
        a pickled copy of this GeneticAlgorithm instance.
        """
        radius, obsVal, obsSign = self._parse_solution(solution)

        a_matrix = self.matrix_module.build_matrix(solution, self.numVar)

        planner = HomotopyPathPlanner(
            radius, self.obstacles, self.start, self.goal,
            obsVal, obsSign, a_matrix, self.arm
        )

        print("Starting homotopy path tracking...")

        path, finalLambda, failed, dis, lambdas = planner.track_path_multi(
            max_steps=self.maxNumRadius
        )

        if not failed:
            self._set_failed(failed)

        fitness = 1 / (dis + 0.000000001)

        return fitness

    def on_generation(self, ga_instance):
        """Callback after each generation — runs in the main process."""
        best_fitness = ga_instance.best_solution(ga_instance.last_generation_fitness)[1]
        gen = ga_instance.generations_completed
        print("Generation = {}".format(gen))
        print("Fitness    = {}".format(best_fitness))
        if not hasattr(self, '_last_fitness'):
            self._last_fitness = 0
        print("Change     = {}".format(best_fitness - self._last_fitness))
        self._last_fitness = best_fitness
        self.fitness_evolution.append(best_fitness)
        if not self._get_failed():
            print("Valid path found — stopping early.")
            return "stop"

    def optimize(self):
        print("start optimizing")
        self._set_failed(True)

        # Use a Manager-backed shared flag for the failed flag when running
        # in process mode, so workers can signal early stopping to the main
        # process. Manager proxies are picklable (unlike raw multiprocessing.Value).
        # The Manager itself is NOT picklable, so we keep it as a local variable
        # and only store the proxy on self (so it gets sent to workers).
        mgr = None
        if (isinstance(self.parallel_processing, (list, tuple))
                and self.parallel_processing[0] == "process"):
            mgr = multiprocessing.Manager()
            self._failed_shared = mgr.Value('b', True)
        else:
            self._failed_shared = None

        num_generations = self.num_generations
        num_parents_mating = self.num_parents_mating
        sol_per_pop = self.sol_per_pop
        num_genes = len(self.gene_space)

        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            fitness_func=self.fitness_func,
            on_generation=self.on_generation,
            gene_space=self.gene_space,
            gene_type=self.gene_type,
            init_range_low=self._init_range_low,
            init_range_high=self._init_range_high,
            parent_selection_type="rank",
            keep_elitism=1,
            crossover_type="sbx",
            mutation_type="random",
            mutation_probability=0.15,
            parallel_processing=self.parallel_processing,
        )

        ga_instance.run()

        # Get the best solution (pass last_generation_fitness to avoid
        # re-evaluating the population, which would re-spawn worker processes
        # after the Manager may have been shut down)
        solution, solution_fitness, solution_idx = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness
        )
        print(f"Best solution: {solution}")
        print(f"Fitness value: {solution_fitness:.6f}")

        self._last_solution = solution

        radius, obsVal, obsSign = self._parse_solution(solution)

        a_matrix = self.matrix_module.build_matrix(solution, self.numVar)

        planner = HomotopyPathPlanner(
            radius, self.obstacles, self.start, self.goal,
            obsVal, obsSign, a_matrix, self.arm
        )

        path, finalLambda, failed, dis, lambdas = planner.track_path_multi(
            max_steps=self.maxNumRadius
        )

        print(f"Path found with {len(path)} steps")
        print(f"Path final positions {path[-1]}")
        print(f"Distance to goal {dis}")

        homotopypParams = self.matrix_module.build_params_vector(
            solution, radius, obsVal, obsSign
        )

        # Clean up the Manager server if we spawned one
        if mgr is not None:
            mgr.shutdown()

        return path, homotopypParams, finalLambda, failed, dis, lambdas, self.fitness_evolution
