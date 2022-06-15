from numba import vectorize, jit, cuda
import numpy as np
import time
import torch


# @jit(nopython=True)
def get_inv_sigma_cuda(Rxtemp, Rytemp, dirtemp):
    sigma_x = Rxtemp / 2
    sigma_y = Rytemp / 2
    sigma = np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])
    rotation = np.array([[np.cos(dirtemp), np.sin(dirtemp)],
                         [-np.sin(dirtemp), np.cos(dirtemp)]])
    sigma = np.dot(np.dot(rotation, sigma), rotation.T)
    return np.linalg.inv(sigma)


# @jit(nopython=True)   # Uncomment this line if numba-available
def get_coord_numba(img_size, stp, btm_l, btm_r, cx_l, cx_r, cy_l, cy_r, top_l, top_r, inv_RRot, fx_l, fx_r, fy_l, fy_r, Tx_r, Ty_r, Tz_r, Xw_l, Yw_l, Zw_l, Xw_r, Yw_r, Zw_r, Zc_l, Zc_r):
    img_size_x = int(img_size[0])
    img_size_y = int(img_size[1])
    stp_x = int(stp[0])
    stp_y = int(stp[1])
    for i in range(img_size_x):
        for j in range(img_size_y):
            i = float(i + stp_x)
            j = float(j + stp_y)
            bottom_r = np.dot(btm_r, np.array([[j + 1.0 - cx_r], [i + 1.0 - cy_r], [1.0]], dtype="float32"))
            bottom_l = np.dot(btm_l, np.array([[j + 1.0 - cx_l], [i + 1.0 - cy_l], [1.0]], dtype="float32"))
            Zcr = top_r / bottom_r[0][0]
            Zcl = top_l / bottom_l[0][0]
            vec_r = np.dot(inv_RRot, np.array(
                [Zcr * (j + 1.0 - cx_r) / fx_r - Tx_r, Zcr * (i + 1.0 - cy_r) / fy_r - Ty_r, Zcr - Tz_r], dtype="float32"))
            vec_l = np.array([Zcl * (j + 1.0 - cx_l) / fx_l, Zcl * (i + 1.0 - cy_l) / fy_l, Zcl], dtype="float32")
            # Zcl, Zcr, vec_l, vec_r = get_coord_numba(btm_l, btm_r, top_l, top_r, )
            i = int(i - stp_x)
            j = int(j - stp_y)
            Zc_l[i][j] = Zcl
            Zc_r[i][j] = Zcr
            Xw_r[i][j] = vec_r[0]
            Yw_r[i][j] = vec_r[1]
            Zw_r[i][j] = vec_r[2]
            Xw_l[i][j] = vec_l[0]
            Yw_l[i][j] = vec_l[1]
            Zw_l[i][j] = vec_l[2]
    return Xw_l, Yw_l, Zw_l, Xw_r, Yw_r, Zw_r, Zc_l, Zc_r


# @jit(nopython=True)
def get_A_numba(img_size, stp, iRotR, cx_l, cx_r, cy_l, cy_r, fx_l, fx_r, fy_l, fy_r, AR, AL):
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            idx = i * img_size[1] + j
            i = float(i + stp[0])
            j = float(j + stp[1])
            AR[:, idx] = np.dot(iRotR, np.array([[(j+1.0-cx_r)/fx_r], [(i+1.0-cy_r)/fy_r], [1.0]], dtype="float32"))[:, 0]
            AL[:, idx] = np.array([(j+1-cx_l)/fx_l, (i+1.0-cy_l)/fy_l, 1.0], dtype="float32")
            i = int(i - stp[0])
            j = int(j - stp[1])
    return AL, AR


# @jit(nopython=True)
def get_disp_numba(img_size, stp, iRotR, Zcl, Zcr, cx_l, cx_r, cy_l, cy_r, fx_l, fx_r, fy_l, fy_r, Tx_r, Ty_r, Tz_r, Xw_l, Yw_l, Zw_l, Xw_r, Yw_r, Zw_r):
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            m = float(i+stp[0])
            n = float(j+stp[1])
            vec_r = np.dot(iRotR, np.array(
                [Zcr[i, j] * (n + 1.0 - cx_r) / fx_r - Tx_r, Zcr[i, j] * (m + 1.0 - cy_r) / fy_r - Ty_r, Zcr[i, j] - Tz_r], dtype="float32"))
            vec_l = np.array([Zcl[i, j] * (n + 1.0 - cx_l) / fx_l, Zcl[i, j] * (m + 1.0 - cy_l) / fy_l, Zcl[i, j]], dtype="float32")
            # i = int(i)
            # j = int(j)
            Xw_r[i][j] = vec_r[0]
            Yw_r[i][j] = vec_r[1]
            Zw_r[i][j] = vec_r[2]
            Xw_l[i][j] = vec_l[0]
            Yw_l[i][j] = vec_l[1]
            Zw_l[i][j] = vec_l[2]

    return Xw_l, Yw_l, Zw_l, Xw_r, Yw_r, Zw_r


# @jit(nopython=True)
def Fx_z(para_1, para_2, para_3, para_4, para_5, Xs, Ys, Fxk):
    return para_1 * np.sin(para_2 * Xs + para_3) * np.sin(para_4 * Ys + para_5) + Fxk


# Torch 加速
def Fx_z(para_1, para_2, para_3, para_4, para_5, Xs, Ys, Fxk):
    Xs = torch.from_numpy(Xs).cuda()
    Ys = torch.from_numpy(Ys).cuda()
    Fxk = torch.from_numpy(Fxk).cuda()
    result = para_1 * torch.sin(para_2 * Xs + para_3) * torch.sin(para_4 * Ys + para_5) + Fxk
    return result.cpu().numpy()


def get_radius_cuda(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    return (x**2 + y**2).cpu().numpy()


def get_tem_img_cuda(temp_img, x, y, invsigma):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    temp_img = torch.from_numpy(temp_img).cuda()
    resu = temp_img + torch.exp(- ((x * invsigma[0, 0] + y * invsigma[1, 0]) * x + (x * invsigma[0, 1] + y * invsigma[1, 1]) * y) / 2.0)
    return resu.cpu().numpy()


def generate_iter_image_nonlocal(temp_img, length_x, length_y, step, x_minmax, y_minmax, box_size, speckle_map, Rx_map, Ry_map, direction_map, cutsize_x, cutsize_y, tem_coord_x, tem_coord_y, color="black"):

    if color == "black":
        temp_img = torch.from_numpy(temp_img.astype("float32")).cuda()
    else:
        temp_img = torch.from_numpy(temp_img.astype("float32")+1.0).cuda()

    tem_coord_x = torch.from_numpy(tem_coord_x.astype("float32")).cuda()
    tem_coord_y = torch.from_numpy(tem_coord_y.astype("float32")).cuda()
    for i in range(0, length_x, step):
        for j in range(0, length_y, step):
            if i < x_minmax[0] - box_size//2 - 2 * step or i > x_minmax[1] + box_size//2 + 2 * step \
                    or j < y_minmax[0] - box_size//2 - 2 * step or j > y_minmax[1] + box_size//2 + 2 * step:
                continue

            if speckle_map[i][j]:
                Rxtemp = Rx_map[i][j]
                Rytemp = Ry_map[i][j]
                dirtemp = direction_map[i][j]
                sigma_x = Rxtemp / 2
                sigma_y = Rytemp / 2
                sigma = np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])
                rotation = np.array([[np.cos(dirtemp), np.sin(dirtemp)],
                                     [-np.sin(dirtemp), np.cos(dirtemp)]])
                sigma = np.dot(np.dot(rotation, sigma), rotation.T)
                invsigma = np.linalg.inv(sigma)
                asumed_loc = (int(i-x_minmax[0]), int(j-y_minmax[0]))
                # asumed_loc = (i % cutsize_x, j % cutsize_y)
                # add_i = -box_size
                # add_j = -box_size
                box_sx = (asumed_loc[0] - box_size) * (asumed_loc[0] - box_size >= 0)
                box_ex = (asumed_loc[0] + box_size - cutsize_x + 1) * (asumed_loc[0] + box_size <= (cutsize_x -1)) + cutsize_x -1
                box_sy = (asumed_loc[1] - box_size) * (asumed_loc[1] - box_size >= 0)
                box_ey = (asumed_loc[1] + box_size - cutsize_y + 1) * (asumed_loc[1] + box_size <= (cutsize_y -1)) + cutsize_y -1

                x = tem_coord_x[box_sx:box_ex, box_sy:box_ey] - i
                y = tem_coord_y[box_sx:box_ex, box_sy:box_ey] - j

                if color == "black":
                    temp_img[box_sx:box_ex, box_sy:box_ey] += torch.exp(- ((x * invsigma[0, 0] + y * invsigma[1, 0]) * x + (x * invsigma[0, 1] + y * invsigma[1, 1]) * y) / 2)
                else:
                    temp_img[box_sx:box_ex, box_sy:box_ey] -= torch.exp(- ((x * invsigma[0, 0] + y * invsigma[1, 0]) * x + (x * invsigma[0, 1] + y * invsigma[1, 1]) * y) / 2)

    return temp_img.cpu().numpy()
