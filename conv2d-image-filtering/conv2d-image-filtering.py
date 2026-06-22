def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    # Add zero padding
    h, w = len(image), len(image[0])
    padded_h = h + 2 * padding
    padded_w = w + 2 * padding

    padded = [[0] * padded_w for _ in range(padded_h)]

    for i in range(h):
        for j in range(w):
            padded[i + padding][j + padding] = image[i][j]

    kh, kw = len(kernel), len(kernel[0])

    # Output dimensions
    out_h = (padded_h - kh) // stride + 1
    out_w = (padded_w - kw) // stride + 1

    output = [[0] * out_w for _ in range(out_h)]

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            total = 0

            for ki in range(kh):
                for kj in range(kw):
                    total += (
                        padded[i * stride + ki][j * stride + kj]
                        * kernel[ki][kj]
                    )

            output[i][j] = total

    return output