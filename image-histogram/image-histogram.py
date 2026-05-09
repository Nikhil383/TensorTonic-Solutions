def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    histogram = [0] * 256

    for row in image:
        for pixel in row:
            histogram[pixel] += 1

    return histogram
    