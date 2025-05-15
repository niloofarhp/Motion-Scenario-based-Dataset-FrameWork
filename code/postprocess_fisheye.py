import numpy as np

def apply_fisheye_to_flow(flow: np.ndarray, k: float, image_shape: tuple) -> np.ndarray:
    """
    Applies a fisheye (radial distortion) effect to an optical flow field.

    Args:
        flow: Optical flow field as an (H, W, 2) numpy array.
        k: Distortion coefficient; positive values simulate barrel distortion.
           (You can use the same value as FLAGS.fisheye if available.)
        image_shape: The (H, W) dimensions of the image.

    Returns:
        A new optical flow field (H, W, 2) after applying the fisheye distortion.
    """
    H, W = image_shape
    x = np.arange(W)
    y = np.arange(H)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack([xv, yv], axis=-1).astype(np.float32)  # shape (H, W, 2)
    
    center = np.array([W / 2, H / 2], dtype=np.float32)
    
    def distort(points: np.ndarray) -> np.ndarray:
        """
        Apply radial distortion to a set of points.
        
        Args:
            points: A (H, W, 2) array of pixel coordinates.
            
        Returns:
            A (H, W, 2) array with the distorted coordinates.
        """
        delta = points - center  # shift to center
        r2 = np.sum(delta**2, axis=-1, keepdims=True)
        factor = 1 + k * r2
        return center + delta * factor

    # distort the original pixel coordinates.
    distorted_coords = distort(coords)
    moved_coords = distort(coords + flow)
    new_flow = moved_coords - distorted_coords
    
    return new_flow