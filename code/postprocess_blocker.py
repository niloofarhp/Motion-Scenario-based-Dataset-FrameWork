import numpy as np
import imageio
import cv2


def apply_blocker_mask(optical_flow: np.ndarray,
                       blocker_image_path: str = None,
                       threshold: int = 128) -> np.ndarray:
    """
    Masks the optical flow channels using a blocker mask.
    This version loads a custom blocker image if provided and resizes it to match the optical flow.
    
    Arguments:
        optical_flow (np.ndarray): The optical flow with shape (H, W, 2).
        blocker_image_path (str, optional): Path to a custom blocker image.
        threshold (int): Threshold for converting the blocker image to binary.
    
    Returns:
        np.ndarray: The blocked optical flow.
    """
    if blocker_image_path is not None:
        blocker_img = imageio.imread(blocker_image_path)
        if blocker_img.ndim == 3:
            blocker_img = blocker_img[..., 0]
        target_height, target_width, _ = optical_flow.shape
        blocker_img = cv2.resize(blocker_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR) # resize to match
        binary_mask = (blocker_img > threshold).astype(np.float32)
        keep_mask = 1.0 - binary_mask
        
    if optical_flow.ndim == 3 and optical_flow.shape[-1] == 2:
        keep_mask = keep_mask[..., None]
    
    masked_flow = optical_flow * keep_mask
    return masked_flow