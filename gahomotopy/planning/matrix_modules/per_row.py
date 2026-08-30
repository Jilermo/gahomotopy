"""Per-row matrix module.

A middle ground between diagonal_dominant (2 genes) and full_matrix (n*n genes).
Uses 2n genes — one off-diagonal weight and one diagonal weight per row.

The matrix is filled the same way as diagonal_dominant, but each row uses
its own pair of values:
  - Off-diagonal: offDiag_i if (i+j)%2==0 else -offDiag_i
  - Diagonal: (n * |offDiag_i|) + diag_i

This guarantees diagonal dominance per row.

Gene layout in solution vector (at the end, interleaved per row):
  solution[-(2n)]   = offDiag_0
  solution[-(2n)+1] = diag_0
  solution[-(2n)+2] = offDiag_1
  solution[-(2n)+3] = diag_1
  ...
  solution[-2]       = offDiag_{n-1}
  solution[-1]       = diag_{n-1}
"""

import numpy as np

from gahomotopy.planning.matrix_modules.base import MatrixModuleBase


class PerRowMatrix(MatrixModuleBase):
    """Diagonally dominant matrix with 2 genes per row (2n total).

    Args:
        off_diagonal: off-diagonal range bound. Gene range becomes [-off_diagonal, off_diagonal].
        diagonal: diagonal range upper bound. Gene range becomes [1, diagonal].
    """

    name = "per_row"

    def __init__(self, off_diagonal=50, diagonal=50):
        self.off_diagonal = off_diagonal
        self.diagonal = diagonal

    def setup_gene_space(self, gene_space, gene_type, num_var):
        """Add 2*n genes: one off-diagonal and one diagonal per row.

        Args:
            gene_space: list of pygad gene space dicts (modified in place)
            gene_type: list of python types (modified in place)
            num_var: number of DOF (matrix dimension)
        """
        for _ in range(num_var):
            # off-diagonal weight (symmetric range)
            gene_space.append({'low': -self.off_diagonal, 'high': self.off_diagonal})
            gene_type.append(float)
            # diagonal weight (always positive)
            gene_space.append({'low': 1, 'high': self.diagonal})
            gene_type.append(float)

    def build_matrix(self, solution, num_var):
        """Build a diagonally dominant matrix from 2n per-row genes.

        Each row i uses its own offDiag_i and diag_i:
          - Off-diagonal entry [i,j]: offDiag_i if (i+j)%2==0 else -offDiag_i
          - Diagonal entry [i,i]: (n * |offDiag_i|) + diag_i

        Args:
            solution: pygad solution array
            num_var: number of DOF (matrix is num_var x num_var)

        Returns:
            numpy array (num_var, num_var)
        """
        n = num_var
        row_genes = solution[-(2 * n):]
        a_matrix = np.zeros((n, n))

        for i in range(n):
            off_diag = row_genes[2 * i]
            diag = row_genes[2 * i + 1]

            for j in range(n):
                if i == j:
                    a_matrix[i, j] = (n * abs(off_diag)) + diag
                else:
                    a_matrix[i, j] = off_diag if (i + j) % 2 == 0 else -off_diag

        return a_matrix

    def build_params_vector(self, solution, radius, obsVal, obsSign):
        """Build params vector: [radius, row_genes..., obsVal..., obsSign...]."""
        n = self._num_var
        row_genes = solution[-(2 * n):]

        numObs = len(obsVal)
        params = np.zeros((2 * n) + 1 + (numObs * 2))
        params[0] = radius
        params[1:2 * n + 1] = row_genes

        for i in range(numObs):
            params[2 * n + 1 + i] = obsVal[i]

        for i in range(numObs):
            params[2 * n + 1 + numObs + i] = obsSign[i]

        return params

    def get_matrix_genes(self, solution):
        """Return the 2*n matrix gene values from the solution."""
        n = self._num_var
        return np.array(solution[-(2 * n):])