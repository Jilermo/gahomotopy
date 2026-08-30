"""Base interface for matrix construction modules.

Subclasses must implement:
  - setup_gene_space(gene_space, gene_type): add matrix-related genes to pygad
  - build_matrix(solution, num_var): extract genes from solution, return a_matrix
  - build_params_vector(solution, radius, obsVal, obsSign): return params vector
  - get_matrix_genes(solution): return the matrix-related gene values from solution
"""


class MatrixModuleBase:
    """Base class for matrix construction strategies."""

    name = "base"

    def setup_gene_space(self, gene_space, gene_type, num_var):
        """Add matrix-related genes to the gene_space and gene_type lists.

        Args:
            gene_space: list of pygad gene space dicts/values (modified in place)
            gene_type: list of python types (modified in place)
            num_var: number of DOF (matrix dimension)
        """
        raise NotImplementedError

    def build_matrix(self, solution, num_var):
        """Build the dominant matrix from the solution vector.

        Args:
            solution: pygad solution array
            num_var: number of DOF (matrix is num_var x num_var)

        Returns:
            numpy array of shape (num_var, num_var)
        """
        raise NotImplementedError

    def build_params_vector(self, solution, radius, obsVal, obsSign):
        """Build the homotopy params vector for result serialization.

        Args:
            solution: pygad solution array
            radius: the radius value
            obsVal: obstacle values array
            obsSign: obstacle signs array

        Returns:
            numpy array of parameters
        """
        raise NotImplementedError

    def get_matrix_genes(self, solution):
        """Extract just the matrix-related gene values from the solution.

        Args:
            solution: pygad solution array

        Returns:
            numpy array of matrix gene values
        """
        raise NotImplementedError