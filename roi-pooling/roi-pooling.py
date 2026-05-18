import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    # Write code here
    pooled_outputs = []

    for roi in rois:
        x1, y1, x2, y2 = roi

        roi_width = x2 - x1
        roi_height = y2 - y1

        pooled = []

        for i in range(output_size):
            row = []

            for j in range(output_size):

                # Compute bin boundaries
                start_y = y1 + math.floor(i * roi_height / output_size)
                end_y = y1 + math.floor((i + 1) * roi_height / output_size)

                start_x = x1 + math.floor(j * roi_width / output_size)
                end_x = x1 + math.floor((j + 1) * roi_width / output_size)

                # Ensure at least one pixel per bin
                if end_y <= start_y:
                    end_y = start_y + 1

                if end_x <= start_x:
                    end_x = start_x + 1

                # Collect values inside the bin
                values = []

                for y in range(start_y, min(end_y, len(feature_map))):
                    for x in range(start_x, min(end_x, len(feature_map[0]))):
                        values.append(feature_map[y][x])

                # Max pooling
                row.append(max(values))

            pooled.append(row)

        pooled_outputs.append(pooled)

    return pooled_outputs