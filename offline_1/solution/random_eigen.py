import numpy as np

INT_MAX = 1e5
INT_MIN = -1e5

def diagonally_dominant_matrix(n: int) -> np.ndarray:
    """
    Generate a random n x n diagonally dominant matrix.

    Parameters
    ----------
    n : int
        The dimension of the matrix.

    Returns
    -------
        out : np.array
            A random n x n diagonally dominant matrix.
    """

    global INT_MAX, INT_MIN
    A = np.random.randint(INT_MIN, INT_MAX, size=(n, n))

    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1

    return A

def rand_invertible_matrix(n: int) -> np.ndarray:
    """
    Generate a random n x n invertible matrix.

    Parameters
    ----------
        n : int 
            The dimension of the matrix.

    Returns
    -------
        out : np.array
            A random n x n invertible matrix.
    """

    return diagonally_dominant_matrix(n)

if __name__ == "__main__":
    n = int(input("Enter the size of the square matrix: "))
    A = rand_invertible_matrix(n)

    print("Random invertible matrix: \n" + str(A))
    eigen_values, eigen_vectors = np.linalg.eig(A)

    reconstructed_A = eigen_vectors @ np.diag(eigen_values) @ np.linalg.inv(eigen_vectors)
    print("Reconstructed matrix: \n" + str(reconstructed_A))

    print("Are generated and reconstructed matrices equal: " + str(np.allclose(A, reconstructed_A)))

    