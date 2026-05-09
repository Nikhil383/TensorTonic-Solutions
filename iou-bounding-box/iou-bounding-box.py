def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    # Intersection coordinates
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    # Intersection width and height
    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)

    # Intersection area
    inter_area = inter_width * inter_height

    # Area of both boxes
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Union area
    union_area = area_a + area_b - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return inter_area / union_area