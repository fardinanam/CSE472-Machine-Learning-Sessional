import cv2 as cv
import numpy as np

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
    img = cv.imread('image.jpeg')
    aspect_ratio = img.shape[0] / img.shape[1]
    img = cv.resize(img, (HORIZONTAL_LENGTH, int(HORIZONTAL_LENGTH * aspect_ratio)))
    grayscaled_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    n = grayscaled_img.shape[0]
    m = grayscaled_img.shape[1]

    print("grayscaled image shape: " + str(grayscaled_img.shape))
    
    max_rank = min(n, m)

    while(True):
        k = int(input(f"Rank of the image k (1 <= k <={max_rank}): "))

        if k <= 0 or k > max_rank:
            print(f"Invalid rank k. Please enter a value between 1 and {max_rank}.")
            continue
        
        approximated_img = np.array(low_rank_approximation(grayscaled_img, k), dtype=np.uint8)
        # plot the resultant k-rank approximation as a grayscale image
        cv.imshow(f"{k}-rank approximation", approximated_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(f"approximations/{k}-rank approximation.jpg", approximated_img)