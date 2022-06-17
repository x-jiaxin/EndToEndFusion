import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def inv_R_t(R, t):
    inv_R = R.permute(0, 2, 1).contiguous()
    return inv_R, t


def summary_metrics(r_mse, t_mse, ):
    """
    www

    Args:
        r_mse:
        t_mse:

    Returns:

    """
    r_mse = np.concatenate(r_mse, axis=0)
    t_mse = np.concatenate(t_mse, axis=0)

    r_mse, t_mse, = np.sqrt(np.mean(r_mse)), np.sqrt(np.mean(t_mse))
    return r_mse, t_mse


def anisotropic_R_error(r1, r2, seq='xyz', degrees=True):
    """
    Calculate mse, mae euler agnle error.

    Args:
        r1: shape=(B, 3, 3), pred
        r2: shape=(B, 3, 3), gt
        seq:
        degrees:
    Returns:
    """
    if isinstance(r1, torch.Tensor):
        r1 = r1.cpu().detach().numpy()
    if isinstance(r2, torch.Tensor):
        r2 = r2.cpu().detach().numpy()
    assert r1.shape == r2.shape
    eulers1, eulers2 = [], []
    for i in range(r1.shape[0]):
        euler1 = Rotation.from_matrix(r1[i]).as_euler(seq=seq, degrees=degrees)
        euler2 = Rotation.from_matrix(r2[i]).as_euler(seq=seq, degrees=degrees)
        eulers1.append(euler1)
        eulers2.append(euler2)
    eulers1 = np.stack(eulers1, axis=0)
    eulers2 = np.stack(eulers2, axis=0)
    r_mse = np.mean((eulers1 - eulers2) ** 2, axis=-1)
    return r_mse


def anisotropic_t_error(t1, t2):
    """
    calculate translation mse and mae error.
    :param t1: shape=(B, 3)
    :param t2: shape=(B, 3)
    :return:
    """
    if isinstance(t1, torch.Tensor):
        t1 = t1.cpu().detach().numpy()
    if isinstance(t2, torch.Tensor):
        t2 = t2.cpu().detach().numpy()
    assert t1.shape == t2.shape
    t_mse = np.mean((t1 - t2) ** 2, axis=1)
    return t_mse


def compute_metrics(R, t, gtR, gtt):
    inv_R, inv_t = inv_R_t(gtR, gtt)
    cur_r_mse = anisotropic_R_error(R, inv_R)
    cur_t_mse = anisotropic_t_error(t, inv_t)
    return cur_r_mse, cur_t_mse
