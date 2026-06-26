
def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.

    Args:
        image: 2D list of integers (0-255)

    Returns:
        2D list with histogram equalization applied.
    """
    # Step 1: Build histogram
    histogram = [0] * 256
    for row in image:
        for pixel in row:
            histogram[pixel] += 1

    # Step 2: Compute CDF (Cumulative Distribution Function)
    cdf = [0] * 256
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]

    total_pixels = cdf[-1]

    # Step 3: Find first non-zero CDF value
    cdf_min = 0
    for value in cdf:
        if value > 0:
            cdf_min = value
            break

    # Step 4: Handle case where all pixels are identical
    if total_pixels == cdf_min:
        return [[0 for _ in row] for row in image]

    # Step 5: Create equalized image
    result = []
    for row in image:
        new_row = []
        for pixel in row:
            new_val = round(
                (cdf[pixel] - cdf_min) /
                (total_pixels - cdf_min)
                * 255
            )
            new_row.append(new_val)
        result.append(new_row)

    return result