import numpy as np
import random_eigen as re

def rand_invertible_symmetric_matrix(n: int) -> np.ndarray:
    """
    Generate a random n x n invertible symmetric matrix

    Parameters
    ----------
        n : int 
            The dimension of the matrix
    
    Returns
    -------
        out : np.array
            A random n x n invertible symmetric matrix
    """

    A = re.rand_invertible_matrix(n)
    return A @ A.T

if __name__ == "__main__":
    n = int(input("Enter the size of the square matrix: "))
    A = rand_invertible_symmetric_matrix(n)
    print("Random invertible symmetric matrix: \n" + str(A))

    eigen_values, eigen_vectors = np.linalg.eig(A)

    reconstructed_A = eigen_vectors @ np.diag(eigen_values) @ eigen_vectors.T
    print("Reconstructed matrix: \n" + str(reconstructed_A))

    print("Are generated and reconstructed matrices equal: " + str(np.allclose(A, reconstructed_A)))

