import os
import math
import numpy as np

import matplotlib.pyplot as plt

SAVE_DIR = "/home/ethan/Documents/Niloofar/Projects/flowdataset/testOutput"


def save_fig(
        fig_id: plt.figure,
        fig_name: str,
        fig_dir: str = '',
        tight_layout: bool = True
):
    # create the directory if it does not exist
    img_dir = os.path.join(SAVE_DIR, "flow_images", fig_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    path = os.path.join(img_dir, fig_name + ".png")
    print("Saving figure", fig_name)
    if tight_layout:
        fig_id.tight_layout()
    fig_id.savefig(path, format='png', dpi=300)


def get_video_names(metadata: dict):  # -> tuple[str, str] :
    print('-' * 30)
    video_name = metadata['video_name'].numpy().decode('UTF-8')
    print("Video: ", video_name)
    video_type = metadata['video_type'].numpy().decode('UTF-8')
    print("Video type: ", video_type)
    return video_name, video_type


def get_scale_offset(metadata: dict, imgdata: str = 'forward_flow'):  # -> tuple[float, float] :
    i_range = metadata[imgdata + '_range'].numpy()
    print("{0} range {1} to {2}".format(imgdata, i_range[0], i_range[1]))
    i_scale = (i_range[1] - i_range[0]) / 65535.0
    i_offset = i_range[0]
    return i_scale, i_offset


def flow_quiver(flow: np.ndarray,
                x_mesh: np.ndarray = np.empty(1),
                y_mesh: np.ndarray = np.empty(1),
                ax=None,
                n_arrows: int = 32,
                img_size: int = 256):
    if ax is None:
        fig, ax = plt.subplots()
    if (x_mesh.shape[0] < math.floor(img_size / n_arrows)
            or y_mesh.shape[0] < math.floor(img_size / n_arrows)):
        x_mesh, y_mesh = np.meshgrid(
            np.linspace(0, img_size, n_arrows),
            np.linspace(0, img_size, n_arrows)
        )

    # Extract the flow components
    u = flow[0: img_size: img_size // n_arrows, 0: img_size: img_size // n_arrows, 1]
    v = -flow[0: img_size: img_size // n_arrows, 0: img_size: img_size // n_arrows, 0]

    # Calculate the magnitude of the vectors
    magnitude = np.sqrt(u ** 2 + v ** 2)

    # Avoid division by zero by adding a small epsilon where magnitude is zero
    epsilon = 1e-10
    magnitude = np.maximum(magnitude, epsilon)

    # Normalize the vectors
    u_normalized = u / magnitude
    v_normalized = v / magnitude

    # Plot the normalized vectors
    res_quiver = ax.quiver(
        x_mesh, -y_mesh,
        u_normalized,
        v_normalized,
        angles='xy'
    )
    return res_quiver, x_mesh, y_mesh


def get_fg_bg_mask(
        segmentation: np.ndarray,
        mask: np.ndarray
):  # -> tuple[ np.ndarray, np.ndarray ]:
    bg_mask = np.logical_and((segmentation == 0), mask)  # background id is 0
    fg_mask = np.logical_and((segmentation > 0), mask)
    return fg_mask, bg_mask
