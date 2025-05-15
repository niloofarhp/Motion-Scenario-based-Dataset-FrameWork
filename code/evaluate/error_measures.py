import numpy as np
import cv2 as cv

from typing import Tuple
import torch


def calc_epe(
        flow: np.ndarray,
        gt_flow: np.ndarray,
        mask: np.ndarray = np.empty(1),
):  # -> tuple[np.ndarray, float] :
    res = np.sum(np.square(flow - gt_flow), axis=2)
    res = np.sqrt(res)
    # res = torch.sum(torch.from_numpy((flow - gt_flow))**2, dim=0).sqrt()
    # print("res: ", res.shape)
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
    # print("mask: ", mask.shape)
    res = res * mask
    # print(res.shape)
    n_mask = np.sum(mask)
    if n_mask > 0:
        epe = np.sum(res) / n_mask
    else:
        epe = 0.0
    return res, epe


def calc_rel_epe(
        flow: np.ndarray,
        gt_flow: np.ndarray,
        epe: np.ndarray,
        mask: np.ndarray = np.empty(1),
):  # -> tuple[np.ndarray, float] :
    # Calculate magnitude
    flow_mag = np.sum(np.square(flow), axis=2)
    flow_mag = np.sqrt(flow_mag)
    # print("flow_mag: ", flow_mag.shape)
    gt_mag = np.sum(np.square(gt_flow), axis=2)
    gt_mag = np.sqrt(gt_mag)
    # print("gt_mag: ", gt_mag.shape)
    f_norm = gt_mag + flow_mag
    # print("f_norm: ", f_norm.shape)
    res = np.divide(epe, f_norm, out=np.zeros_like(epe),
                    where=~np.isclose(f_norm, np.zeros_like(f_norm)))
    # print("res: ", res.shape)
    # print("mask: ", mask.shape)
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
    # print("mask: ", mask.shape)
    res = res * mask
    # print(res.shape)
    n_mask = np.sum(mask)
    if n_mask > 0:
        rel_epe = np.sum(res) / n_mask
    else:
        rel_epe = 0.0
    return res, rel_epe


def abs_jacobian_determinant(
        flow: np.ndarray
):  # -> np.ndarray :
    udx, udy = np.gradient(flow[:, :, 0])
    vdx, vdy = np.gradient(flow[:, :, 1])
    jacob = np.stack([udx, udy, vdx, vdy])
    res = (jacob[0, :, :] * jacob[3, :, :]) - (jacob[1, :, :] * jacob[2, :, :])
    return np.abs(res)


def l2_jacobian(
        flow: np.ndarray
):  # -> np.ndarray :
    udx, udy = np.gradient(flow[:, :, 0])
    vdx, vdy = np.gradient(flow[:, :, 1])
    jacob = np.stack([udx, udy, vdx, vdy])
    res = jacob[0, :, :] ** 2 + jacob[1, :, :] ** 2 + jacob[3, :, :] ** 2 + jacob[2, :, :] ** 2
    return res


def abs_divergence(
        flow: np.ndarray
):  # -> np.ndarray :
    udx = np.gradient(flow[:, :, 0], axis=0)
    vdy = np.gradient(flow[:, :, 1], axis=1)
    res = udx + vdy
    return np.abs(res)


def mag_grad(
        img: np.ndarray,
        threshold: float = 0.1
):  # -> np.ndarray :
    fdx, fdy = np.gradient(img)
    sq_mag = (np.square(fdx) + np.square(fdy))
    return (sq_mag > threshold)


def rel_mag_grad(
        img: np.ndarray,
        threshold: float = 0.1
):  # -> np.ndarray :
    # Calculate gradient magnitude
    fdx, fdy = np.gradient(img)
    g_mag = np.sqrt(np.square(fdx) + np.square(fdy))
    # Normalize by img
    res = np.divide(g_mag, img, out=np.zeros_like(g_mag),
                    where=~np.isclose(img, np.zeros_like(img)))
    return (res > threshold)


def border_distance(
        border_img: np.ndarray,
        max_dist: float = -1.0,
        min_dist: float = -1.0
):  # -> np.ndarray :
    ui_img = np.uint8(border_img)
    dist = cv.distanceTransform(ui_img, cv.DIST_L2, 3)
    if max_dist < 0:
        return dist
    else:
        return np.logical_and(dist < max_dist, dist > min_dist)


def epe_select(
        epe: np.ndarray,
        gt_flow: np.ndarray,
        min_flow: float = -1.0,
        max_flow: float = 1024.0,
        mask: np.ndarray = np.empty(1),
):  # -> tuple[np.ndarray, float] :
    gt_mag = np.sum(np.square(gt_flow), axis=2)
    gt_mag = np.sqrt(gt_mag)
    if min_flow < 0.0:
        f_mask = gt_mag < max_flow
    else:
        f_mask = np.logical_and(gt_mag > min_flow, gt_mag < max_flow)
    if mask.ndim > 1:
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
        f_mask = f_mask * mask
    res = epe * f_mask
    n_mask = np.sum(f_mask)
    if n_mask > 0:
        epe = np.sum(res) / n_mask
    else:
        epe = 0.0
    return res, epe


def select_epe_region(
        epe: np.ndarray,
        mask: np.ndarray = np.empty(1),
):  # -> tuple[np.ndarray, float] :
    error_region = epe * mask
    n_mask = np.sum(mask)
    if n_mask > 0:
        epe_region = np.sum(error_region) / n_mask
    else:
        epe_region = 0.0
    return error_region, epe_region


def occlusion_mask(forward_flow, backward_flow, mask, scale=0.01, bias=0.5):
    h, w, _ = forward_flow.shape
    xy_location = np.float32(np.transpose(np.mgrid[:h, :w], (1, 2, 0)))
    xy_location = xy_location + forward_flow.astype(np.float32)
    backward_warped = cv.remap(backward_flow,
                               xy_location[:, :, 1], xy_location[:, :, 0],
                               interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    backward_warped = mask * backward_warped
    flow_diff = forward_flow - backward_warped
    # 2 * average the magnitude of the flow vectors per pixel
    mag = np.sum(np.square(forward_flow), 2) + \
          np.sum(np.square(backward_warped), 2)
    threshold = scale * mag + bias
    # Relative threshold of the magnitude of the difference per pixel
    res = np.sum(np.square(flow_diff), 2) > threshold
    return res, flow_diff
