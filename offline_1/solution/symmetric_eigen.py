import numpy as np

INT_MIN = -1e9
INT_MAX = 1e9

def rand_invertible_symmetric_matrix(n: int) -> np.ndarray:
    """Generate a random n x n invertible symmetric matrix."""
    global INT_MAX, INT_MIN
    while True:
        A = np.random.randint(INT_MIN // 2, INT_MAX // 2, size=(n, n))
        A = A + A.T

        if np.linalg.det(A) != 0:
            return A

if __name__ == "__main__":
    n = int(input("Enter the size of the square matrix: "))
    A = rand_invertible_symmetric_matrix(n)
    print("Random invertible symmetric matrix: \n" + str(A))

    eigen_values, eigen_vectors = np.linalg.eig(A)

    reconstructed_A = eigen_vectors @ np.diag(eigen_values) @ eigen_vectors.T
    print("Reconstructed matrix: \n" + str(reconstructed_A))

    print("Are generated and reconstructed matrices equal: " + str(np.allclose(A, reconstructed_A)))

