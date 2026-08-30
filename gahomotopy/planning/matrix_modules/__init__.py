"""Matrix construction modules for the genetic algorithm.

Each module defines how to:
  - Set up gene space and gene types for pygad (matrix-related genes)
  - Build the dominant matrix from a solution vector
  - Build the homotopy params vector for result serialization

Usage:
    from gahomotopy.planning.matrix_modules import DiagonalDominantMatrix
    matrix_module = DiagonalDominantMatrix()
    matrix_module.setup_gene_space(gene_space, gene_type)
    a_matrix = matrix_module.build_matrix(solution, num_var)
    params = matrix_module.build_params_vector(solution, radius, obsVal, obsSign)
"""

from gahomotopy.planning.matrix_modules.base import MatrixModuleBase
from gahomotopy.planning.matrix_modules.diagonal_dominant import DiagonalDominantMatrix
from gahomotopy.planning.matrix_modules.full_matrix import FullMatrix
from gahomotopy.planning.matrix_modules.per_row import PerRowMatrix