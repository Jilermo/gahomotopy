"""Diagonal dominant matrix module.

Builds an n x n diagonally dominant matrix from two scalar genes:
  - offDiag: off-diagonal weight (alternating signs for spatial mixing)
  - diag: additional diagonal weight

The diagonal value is: (n * |offDiag|) + diag
This guarantees the matrix is invertible (diagonally dominant).

Gene layout in solution vector (at the end):
  solution[-2] = offDiag
  solution[-1] = diag
"""

import numpy as np

from gahomotopy.planning.matrix_modules.base import MatrixModuleBase


class DiagonalDominantMatrix(MatrixModuleBase):
    """Diagonally dominant matrix built from 2 scalar parameters.

    Args:
        off_diagonal: off-diagonal range bound. Gene range becomes [-off_diagonal, off_diagonal].
        diagonal: diagonal range upper bound. Gene range becomes [1, diagonal].
    """

    name = "diagonal_dominant"

    def __init__(self, off_diagonal=50, diagonal=50):
        self.off_diagonal = off_diagonal
        self.diagonal = diagonal

    def setup_gene_space(self, gene_space, gene_type, num_var):
        """Add 2 genes: off-diagonal weight and diagonal weight.

        Args:
            num_var: unused (matrix dimension — this module always uses 2 genes)
        """
        # off-diagonal weight (symmetric range)
        gene_space.append({'low': -self.off_diagonal, 'high': self.off_diagonal})
        gene_type.append(float)

        # diagonal weight (always positive)
        gene_space.append({'low': 1, 'high': self.diagonal})
        gene_type.append(float)

    def build_matrix(self, solution, num_var):
        """Build the dominant matrix from the last 2 genes in the solution.

        Args:
            solution: pygad solution array
            num_var: number of DOF

        Returns:
            numpy array (num_var, num_var)
        """
        off_diag = solution[-2]
        diag = solution[-1]

        a_matrix = np.zeros((num_var, num_var))

        for i in range(num_var):
            for j in range(num_var):
                if i == j:
                    a_matrix[i, j] = (num_var * abs(off_diag)) + diag
                else:
                    a_matrix[i, j] = off_diag if (i + j) % 2 == 0 else -off_diag

        return a_matrix

    def build_params_vector(self, solution, radius, obsVal, obsSign):
        """Build params vector: [radius, offDiag, diag, obsVal..., obsSign...]."""
        off_diag = solution[-2]
        diag = solution[-1]

        numObs = len(obsVal)
        params = np.zeros((numObs * 2) + 3)
        params[0] = radius
        params[1] = off_diag
        params[2] = diag

        for i in range(numObs):
            params[i + 3] = obsVal[i]

        for i in range(numObs):
            params[i + numObs + 3] = obsSign[i]

        return params

    def get_matrix_genes(self, solution):
        """Return the 2 matrix gene values from the solution."""
        return np.array([solution[-2], solution[-1]])