import numpy as np

INT_MAX = 1e9
INT_MIN = -1e9

def rand_invertible_matrix(n: int) -> np.ndarray:
    """Generate a random n x n invertible matrix."""
    global INT_MAX, INT_MIN
    A = np.random.randint(INT_MIN, INT_MAX, size=(n, n))

    while np.linalg.det(A) == 0:
        A = np.random.rand(n, n)

    return A

if __name__ == "__main__":
    n = int(input("Enter the size of the square matrix: "))
    A = rand_invertible_matrix(n)

    print("Random invertible matrix: \n" + str(A))
    eigen_values, eigen_vectors = np.linalg.eig(A)

    reconstructed_A = eigen_vectors @ np.diag(eigen_values) @ np.linalg.inv(eigen_vectors)
    print("Reconstructed matrix: \n" + str(reconstructed_A))

    print("Are generated and reconstructed matrices equal: " + str(np.allclose(A, reconstructed_A)))

    