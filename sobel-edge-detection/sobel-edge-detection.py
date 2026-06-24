def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    rows = len(image)
    cols = len(image[0])

    # Sobel kernels
    Gx_kernel = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

    Gy_kernel = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]

    # Zero padding
    padded = [[0] * (cols + 2) for _ in range(rows + 2)]

    for i in range(rows):
        for j in range(cols):
            padded[i + 1][j + 1] = image[i][j]

    result = [[0.0] * cols for _ in range(rows)]

    # Apply Sobel operator
    for i in range(rows):
        for j in range(cols):
            gx = 0
            gy = 0

            for r in range(3):
                for c in range(3):
                    pixel = padded[i + r][j + c]
                    gx += pixel * Gx_kernel[r][c]
                    gy += pixel * Gy_kernel[r][c]

            result[i][j] = math.sqrt(gx * gx + gy * gy)

    return result