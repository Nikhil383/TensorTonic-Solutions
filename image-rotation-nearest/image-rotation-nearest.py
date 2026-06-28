import math

def rotate_image(image, angle_degrees):
    H = len(image)
    W = len(image[0])

    cy = (H - 1) / 2
    cx = (W - 1) / 2

    theta = math.radians(angle_degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    output = [[0] * W for _ in range(H)]

    for i in range(H):
        for j in range(W):

            dy = i - cy
            dx = j - cx

            # Inverse rotation (exactly as given)
            src_y = cy + dy * cos_theta + dx * sin_theta
            src_x = cx - dy * sin_theta + dx * cos_theta

            sy = round(src_y)
            sx = round(src_x)

            if 0 <= sy < H and 0 <= sx < W:
                output[i][j] = image[sy][sx]
            else:
                output[i][j] = 0

    return output