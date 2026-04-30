import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T=np.array(T)
    points=np.array(points)
    # Handle single point
    single_point = False
    if points.ndim == 1:
        points = points.reshape(1, 3)
        single_point = True

    # Convert to homogeneous coordinates
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply transformation
    transformed_h = (T @ points_h.T).T

    # Convert back to 3D
    transformed = transformed_h[:, :3]

    return transformed[0] if single_point else transformed