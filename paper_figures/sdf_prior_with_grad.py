import numpy as np


def extrapolate_2d(sdf0: np.ndarray, grad: np.ndarray, points0: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Extrapolate SDF values to new points using the gradient at a known point.

    Args:
        sdf0: (N, ) array of the known SDF values at the reference points.
        grad: (N, 2) array of the gradients (normal vectors) at the reference points.
        points0: (N, 2) array of the coordinates of the reference points.
        points: (M, 2) array of the coordinates of the points to extrapolate SDF values to.

    """
    grad = grad / np.linalg.norm(grad, axis=1, keepdims=True)  # Normalize the gradients
    # Project points onto the plane defined by sdf0 and grad
    diff = points.reshape(-1, 1, 2) - points0.reshape(1, -1, 2)  # (M, N, 2)
    return np.einsum("mni,ni->mn", diff, grad) + sdf0.reshape(1, -1)  # (M, N)


def bilinear_interpolate(q: np.ndarray, vertex_values: np.ndarray) -> np.ndarray:
    """
    Perform bilinear interpolation.

    Args:
        q: (N, 2) array of the coordinates of the points to interpolate.
        vertex_values: (N, 4) array of the vertex values for the bilinear interpolation.
    """
    qx = q[:, 0]
    qy = q[:, 1]
    qx_bar = 1 - qx
    qy_bar = 1 - qy

    weights = np.stack(
        [  # (N, 4)
            qx_bar * qy_bar,  # (0, 0)
            qx * qy_bar,  # (1, 0)
            qx_bar * qy,  # (0, 1)
            qx * qy,  # (1, 1)
        ],
        axis=-1,
    )
    return (weights * vertex_values).sum(axis=1)  # (N, )


def sdf_prior_without_grad_2d(sdf0: np.ndarray, points0: np.ndarray, points: np.ndarray) -> np.ndarray:
    x_min = points0[0, 0]
    x_max = points0[1, 0]
    y_min = points0[0, 1]
    y_max = points0[2, 1]
    # Normalize points to [0, 1] range for bilinear interpolation
    q = (points - np.array([x_min, y_min])) / np.array([x_max - x_min, y_max - y_min])
    q = np.clip(q, 0, 1)  # (M, 2)
    return bilinear_interpolate(q, sdf0)


def sdf_prior_with_grad_2d(sdf0: np.ndarray, grad: np.ndarray, points0: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute the SDF prior with gradient information for 2D points.

    Args:
        sdf0: (4, ) array of the known SDF values at the reference points.
        grad: (4, 2) array of the gradients (normal vectors) at the reference points.
        points0: (4, 2) array of the coordinates of the reference points: (xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)
        points: (M, 2) array of the coordinates of the points to extrapolate SDF values to.

    """
    # Extrapolate SDF values to new points using the gradient at a known point
    extrapolated_sdf = extrapolate_2d(sdf0, grad, points0, points)  # (M, 4)
    x_min = points0[0, 0]
    x_max = points0[1, 0]
    y_min = points0[0, 1]
    y_max = points0[2, 1]
    # Normalize points to [0, 1] range for bilinear interpolation
    q = (points - np.array([x_min, y_min])) / np.array([x_max - x_min, y_max - y_min])
    q = np.clip(q, 0, 1)  # (M, 2)
    sdf_prior = bilinear_interpolate(q, extrapolated_sdf)
    return sdf_prior


def trilinear_interpolate(q: np.ndarray, vertex_values: np.ndarray) -> np.ndarray:
    """
    Perform trilinear interpolation.

    Args:
        q: (N, 3) array of the coordinates of the points to interpolate.
        vertex_values: (N, 8) array of the vertex values for the trilinear interpolation.
    """
    qx = q[:, 0]
    qy = q[:, 1]
    qz = q[:, 2]
    qx_bar = 1 - qx
    qy_bar = 1 - qy
    qz_bar = 1 - qz

    weights = np.hstack(
        [  # (N, 8)
            qx_bar * qy_bar * qz_bar,
            qx * qy_bar * qz_bar,
            qx_bar * qy * qz_bar,
            qx * qy * qz_bar,
            qx_bar * qy_bar * qz,
            qx * qy_bar * qz,
            qx_bar * qy * qz,
            qx * qy * qz,
        ]
    )
    return (weights * vertex_values).sum(axis=1)  # (N, )
