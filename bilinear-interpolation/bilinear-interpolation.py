def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    # Write code here

    old_h = len(image)
    old_w = len(image[0])

    resized = [[0] * new_w for _ in range(new_h)]

    x_ratio = (old_w - 1) / (new_w - 1) if new_w > 1 else 0
    y_ratio = (old_h - 1) / (new_h - 1) if new_h > 1 else 0

    for i in range(new_h):
        for j in range(new_w):
            # Corresponding position in original image
            x = j * x_ratio
            y = i * y_ratio

            x1 = int(x)
            y1 = int(y)

            x2 = min(x1 + 1, old_w - 1)
            y2 = min(y1 + 1, old_h - 1)

            dx = x - x1
            dy = y - y1

            # Four neighboring pixels
            top_left = image[y1][x1]
            top_right = image[y1][x2]
            bottom_left = image[y2][x1]
            bottom_right = image[y2][x2]

            # Bilinear interpolation
            top = top_left * (1 - dx) + top_right * dx
            bottom = bottom_left * (1 - dx) + bottom_right * dx

            resized[i][j] = top * (1 - dy) + bottom * dy

    return resized
    