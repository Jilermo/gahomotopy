"""Full matrix module.

Every entry of the n x n matrix is its own GA gene (n*n total).
Off-diagonal genes use a symmetric range [-off_diagonal, off_diagonal].
Diagonal genes use a positive range [1, diagonal].

The matrix is built to be diagonally dominant: each diagonal entry is
the sum of all off-diagonal elements in that row plus the diagonal gene.
This guarantees invertibility.

Gene layout in solution vector (at the end):
  solution[-(n*n) : -(n)]     = off-diagonal genes (n*(n-1) values, row-major)
  solution[-n:]               = diagonal genes (n values, one per row)
"""

import numpy as np

from gahomotopy.planning.matrix_modules.base import MatrixModuleBase


class FullMatrix(MatrixModuleBase):
    """Full n x n diagonally dominant matrix where every entry is a GA gene.

    Args:
        off_diagonal: off-diagonal range bound. Gene range becomes [-off_diagonal, off_diagonal].
        diagonal: diagonal range upper bound. Gene range becomes [1, diagonal].
    """

    name = "full_matrix"

    def __init__(self, off_diagonal=50, diagonal=50):
        self.off_diagonal = off_diagonal
        self.diagonal = diagonal

    def setup_gene_space(self, gene_space, gene_type, num_var):
        """Add n*n genes: n*(n-1) off-diagonal + n diagonal.

        Args:
            gene_space: list of pygad gene space dicts (modified in place)
            gene_type: list of python types (modified in place)
            num_var: number of DOF (matrix dimension)
        """
        n = num_var
        # off-diagonal genes (symmetric range)
        for _ in range(n * (n - 1)):
            gene_space.append({'low': -self.off_diagonal, 'high': self.off_diagonal})
            gene_type.append(float)

        # diagonal genes (always positive)
        for _ in range(n):
            gene_space.append({'low': 1, 'high': self.diagonal})
            gene_type.append(float)

    def build_matrix(self, solution, num_var):
        """Build a diagonally dominant matrix from the last n*n genes.

        Off-diagonal entries are filled row-major from the first n*(n-1) genes.
        Each diagonal entry is the sum of its row's off-diagonal values plus
        the corresponding diagonal gene, ensuring diagonal dominance.

        Args:
            solution: pygad solution array
            num_var: number of DOF (matrix is num_var x num_var)

        Returns:
            numpy array (num_var, num_var)
        """
        n = num_var
        matrix_genes = solution[-(n * n):]
        offdiag_genes = matrix_genes[:n * (n - 1)]
        diag_genes = matrix_genes[n * (n - 1):]

        a_matrix = np.zeros((n, n))

        # Fill off-diagonal entries (row-major, skipping diagonal)
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    a_matrix[i, j] = offdiag_genes[idx]
                    idx += 1

        # Diagonal: sum of off-diagonal elements in the row + diagonal gene
        for i in range(n):
            row_sum = np.sum(np.abs(a_matrix[i, :]))  # sum of off-diagonals (diagonal is still 0)
            a_matrix[i, i] = row_sum + diag_genes[i]
        print(a_matrix)
        return a_matrix

    def build_params_vector(self, solution, radius, obsVal, obsSign):
        """Build params vector: [radius, matrix_genes..., obsVal..., obsSign...]."""
        n = self._num_var
        matrix_genes = solution[-(n * n):]

        numObs = len(obsVal)
        params = np.zeros((n * n) + 1 + (numObs * 2))
        params[0] = radius
        params[1:n * n + 1] = matrix_genes

        for i in range(numObs):
            params[n * n + 1 + i] = obsVal[i]

        for i in range(numObs):
            params[n * n + 1 + numObs + i] = obsSign[i]

        return params

    def get_matrix_genes(self, solution):
        """Return the n*n matrix gene values from the solution."""
        n = self._num_var
        return np.array(solution[-(n * n):])
