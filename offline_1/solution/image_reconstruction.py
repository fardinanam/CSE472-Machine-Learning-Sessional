import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

HORIZONTAL_LENGTH = 500

def low_rank_approximation(A: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the k-rank approximation of matrix A

    Parameters
    ----------
        A : np.ndarray
            The matrix to be approximated
        k : int
            The rank of the approximation

    Returns
    -------
        out : np.ndarray
            k-rank approximation of matrix A
    """
    U, S, V = np.linalg.svd(A)
    A_k = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    return A_k

if __name__ == '__main__':
    img = cv.imread('image.jpg')
    aspect_ratio = img.shape[0] / img.shape[1]
    img = cv.resize(img, (HORIZONTAL_LENGTH, int(HORIZONTAL_LENGTH * aspect_ratio)))
    grayscaled_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    n = grayscaled_img.shape[0]
    m = grayscaled_img.shape[1]

    print("grayscaled image shape: " + str(grayscaled_img.shape))
    
    max_rank = min(n, m)

    plt.figure(figsize=(15, 15))

    count = 0
    k_start = int(input(f"Start rank k (1 <= k <={max_rank}): "))
    k_end = int(input(f"End rank k ({k_start + 10} < k <={max_rank}): "))
    k_count = 10

    for k in range(k_start, k_end + 1, (k_end - k_start) // k_count + 1):
        approximated_img = np.array(low_rank_approximation(grayscaled_img, k), dtype=np.uint8)
        count += 1
        plt.subplot(3, 4, count)
        plt.title(f"{k}-rank approximation")
        plt.imshow(approximated_img, cmap="gray")
    
    plt.show()