import numpy as np

def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.

    Args:
        image: 2D list or numpy array containing 0s and 1s.
        kernel: 2D list or numpy array (structuring element).
        operation: "erode" or "dilate".

    Returns:
        2D list representing the processed image.
    """
    image = np.array(image)
    kernel = np.array(kernel)

    h, w = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # Zero padding
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant")
    output = np.zeros((h, w), dtype=int)

    for i in range(h):
        for j in range(w):
            if operation == "erode":
                output[i, j] = 1
                for ki in range(kh):
                    for kj in range(kw):
                        if kernel[ki, kj] == 1 and padded[i + ki, j + kj] == 0:
                            output[i, j] = 0
                            break
                    if output[i, j] == 0:
                        break

            elif operation == "dilate":
                output[i, j] = 0
                for ki in range(kh):
                    for kj in range(kw):
                        if kernel[ki, kj] == 1 and padded[i + ki, j + kj] == 1:
                            output[i, j] = 1
                            break
                    if output[i, j] == 1:
                        break

            else:
                raise ValueError("Operation must be 'erode' or 'dilate'.")

    return output.tolist()