import numpy as np

def rand_invertible_matrix(n):
    """Generate a random n x n invertible matrix."""
    # generate n * n matrix of integers
    A = np.random.randint(0, 10, size=(n, n))
    # count = 1
    while np.linalg.det(A) == 0:
        A = np.random.rand(n, n)
        # count += 1
    
    # print("Number of attempts: ", count)
    return A

for i in range(10):
    A = rand_invertible_matrix(3)
    print(A)
    

    