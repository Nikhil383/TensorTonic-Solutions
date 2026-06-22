def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    # Write code here
    def compute_iou(box1, box2):
        # Intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Intersection area
        inter_width = max(0, x2 - x1)
        inter_height = max(0, y2 - y1)
        intersection = inter_width * inter_height

        # Areas of boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Union area
        union = area1 + area2 - intersection

        return intersection / union if union != 0 else 0

    # Sort indices by score (highest first)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    keep = []

    while indices:
        current = indices.pop(0)
        keep.append(current)

        remaining = []
        for idx in indices:
            if compute_iou(boxes[current], boxes[idx]) < iou_threshold:
                remaining.append(idx)

        indices = remaining

    return keep
    