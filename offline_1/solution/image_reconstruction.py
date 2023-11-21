import cv2 as cv
import numpy as np

HORIZONTAL_LENGTH = 500

def low_rank_approximation(A: np.ndarray, k: int) -> np.ndarray:
    """Return: k-rank approximation of matrix A."""
    U, S, V = np.linalg.svd(A)
    A_k = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    return A_k

if __name__ == '__main__':
    img = cv.imread('image.jpeg')
    aspect_ratio = img.shape[0] / img.shape[1]
    img = cv.resize(img, (HORIZONTAL_LENGTH, int(HORIZONTAL_LENGTH * aspect_ratio)))
    grayscaled_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Original image", grayscaled_img)
    cv.waitKey(0)

    n = grayscaled_img.shape[0]
    m = grayscaled_img.shape[1]

    print("grayscaled image shape: " + str(grayscaled_img.shape))
    
    min_rank = min(n, m)
    for i in range(1, min_rank, min_rank // 10):
        approximated_img = low_rank_approximation(grayscaled_img, i)
        # plot the resultant k-rank approximation as a grayscale image
        cv.imshow(f"{i}-rank approximation", approximated_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(f"approximations/{i}-rank approximation.jpg", approximated_img)


    