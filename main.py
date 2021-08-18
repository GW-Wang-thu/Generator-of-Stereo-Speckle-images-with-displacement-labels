import numpy as np
import json
from utils.utils import generate_seedmap, array2img, calculatedisp, img_flip, calculatedisp_ws
from utils.dispsolver import disp_solver_zc, disp_solver_xys
import cv2
import matplotlib.pyplot as plt
import os


def cal_plane_equ(params):
    # Solve proper Zc
    Zc = - params["RTX"] / params["RRot"][0][2]
    Os = np.array([[0], [0], [Zc]])
    # Normal direction
    RRot = params["RRot"]
    RT = [[-params["RTX"]], [-params["RTY"]], [-params["RTZ"]]]
    Or_w = np.dot(np.linalg.inv(np.array(RRot)), np.array(RT))
    n_r = (Or_w - Os) / np.linalg.norm((Or_w - np.array([[0], [0], [Zc]])))
    print(n_r)
    z_s = (np.array([[0], [0], [-1]]) + n_r) / np.linalg.norm(np.array([[0], [0], [-1]]) + n_r)
    print(z_s)
    # Solve beta_sw
    y_s = np.array([[1], [0], [-z_s[0][0] / z_s[2][0]]]) / np.linalg.norm(np.array([[1], [0], [-z_s[0][0] / z_s[2][0]]]))
    x_s = np.array([[y_s[2][0] * z_s[1][0] / (z_s[2][0]-y_s[2][0]*z_s[0][0])], [1], [-z_s[1][0] / (z_s[2][0]-y_s[2][0]*z_s[0][0])]]) / np.linalg.norm(np.array([[y_s[2][0] * z_s[1][0] / (z_s[2][0]-y_s[2][0]*z_s[0][0])], [1], [-z_s[1][0] / (z_s[2][0]-y_s[2][0]*z_s[0][0])]]))
    beta_sw = np.array([[x_s[0][0], x_s[1][0], x_s[2][0]], [y_s[0][0], y_s[1][0], y_s[2][0]], [z_s[0][0], z_s[1][0], z_s[2][0]]])
    # Solve Tsw
    Tsw = np.array([[-Zc * beta_sw[0][2]], [-Zc * beta_sw[1][2]], [-Zc * beta_sw[2][2]]])
    A = - z_s[0][0] / z_s[2][0]
    B = - z_s[1][0] / z_s[2][0]
    C = Zc
    return (A, B, C), beta_sw, Tsw


def cal_plane_world_ordinate(plane_param, imaging_params):
    img_size = imaging_params["RResolution"]
    A = plane_param[0]
    B = plane_param[1]
    C = plane_param[2]
    RRot = imaging_params["RRot"]
    cx_r = imaging_params["RCX"]
    fx_r = imaging_params["RFX"]
    Tx_r = imaging_params["RTX"]
    cy_r = imaging_params["RCY"]
    fy_r = imaging_params["RFY"]
    Ty_r = imaging_params["RTY"]
    Tz_r = imaging_params["RTZ"]
    cx_l = imaging_params["LCX"]
    cy_l = imaging_params["LCY"]
    fx_l = imaging_params["LFX"]
    fy_l = imaging_params["LFY"]
    inv_RRot = np.linalg.inv(np.array(RRot))
    Xw_r = np.zeros(shape=img_size)
    Yw_r = np.zeros(shape=img_size)
    Zw_r = np.zeros(shape=img_size)
    top_r = C + Tx_r * (inv_RRot[2][0] - A * inv_RRot[0][0] - B * inv_RRot[1][0]) \
            + Ty_r * (inv_RRot[2][1] - A * inv_RRot[0][1] - B * inv_RRot[1][1]) \
            + Tz_r * (inv_RRot[2][2] - A * inv_RRot[0][2] - B * inv_RRot[1][2])
    top_l = C
    btm_r = np.array([[(inv_RRot[2][0] - A * inv_RRot[0][0] - B * inv_RRot[1][0])/fx_r, (inv_RRot[2][1] - A * inv_RRot[0][1] - B * inv_RRot[1][1])/fy_r, inv_RRot[2][2] - A * inv_RRot[0][2] - B * inv_RRot[1][2]]])
    btm_l = np.array([[-A/fx_l, -B/fy_l, 1]])

    Xw_l = np.zeros(shape=img_size)
    Yw_l = np.zeros(shape=img_size)
    Zw_l = np.zeros(shape=img_size)

    Zc_l = np.zeros(shape=img_size)
    Zc_r = np.zeros(shape=img_size)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            bottom_r = np.dot(btm_r, np.array([[j+1-cx_r], [i+1-cy_r], [1]]))
            bottom_l = np.dot(btm_l, np.array([[j+1-cx_l], [i+1-cy_l], [1]]))
            Zcr = top_r / bottom_r[0][0]
            Zcl = top_l / bottom_l[0][0]
            vec_r = np.dot(inv_RRot, np.array([Zcr*(j+1-cx_r)/fx_r - Tx_r, Zcr*(i+1-cy_r)/fy_r - Ty_r, Zcr - Tz_r]))
            vec_l = np.array([Zcl*(j+1-cx_l)/fx_l, Zcl*(i+1-cy_l)/fy_l, Zcl])
            Zc_l[i][j] = Zcl
            Zc_r[i][j] = Zcr
            Xw_r[i][j] = vec_r[0]
            Yw_r[i][j] = vec_r[1]
            Zw_r[i][j] = vec_r[2]
            Xw_l[i][j] = vec_l[0]
            Yw_l[i][j] = vec_l[1]
            Zw_l[i][j] = vec_l[2]
    return (Xw_l, Yw_l, Zw_l), (Xw_r, Yw_r, Zw_r), (Zc_l, Zc_r)


def transform_w2s(beta_sw, world_coordinate, Tsw=[[0.0], [0.0], [0.0]]):
    '''Transportation'''
    Xw = world_coordinate[0]
    Yw = world_coordinate[1]
    Zw = world_coordinate[2]
    Xs = beta_sw[0][0] * Xw + beta_sw[0][1] * Yw + beta_sw[0][2] * Zw + Tsw[0][0]
    Ys = beta_sw[1][0] * Xw + beta_sw[1][1] * Yw + beta_sw[1][2] * Zw + Tsw[1][0]
    Zs = beta_sw[2][0] * Xw + beta_sw[2][1] * Yw + beta_sw[2][2] * Zw + Tsw[2][0]
    return (Xs, Ys, Zs)


def generate_displacement(imaging_params, beta_sw, Tsw, Zc, disp_info):
    img_size = imaging_params["RResolution"]
    cx_r = imaging_params["RCX"]
    fx_r = imaging_params["RFX"]
    Tx_r = imaging_params["RTX"]
    cy_r = imaging_params["RCY"]
    fy_r = imaging_params["RFY"]
    Ty_r = imaging_params["RTY"]
    Tz_r = imaging_params["RTZ"]
    cx_l = imaging_params["LCX"]
    cy_l = imaging_params["LCY"]
    fx_l = imaging_params["LFX"]
    fy_l = imaging_params["LFY"]
    AL = np.zeros(shape=(3, img_size[0] * img_size[1]))
    AR = np.zeros_like(AL)
    RRot = params["RRot"]
    iRotR = np.linalg.inv(np.array(RRot))
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            idx = i * img_size[1] + j
            AR[:, idx] = np.dot(iRotR, np.array([[(j+1-cx_r)/fx_r], [(i+1-cy_r)/fy_r], [1]]))[:, 0]
            AL[:, idx] = np.array([(j+1-cx_l)/fx_l, (i+1-cy_l)/fy_l, 1])
    bR = - np.dot(iRotR, np.array([[Tx_r], [Ty_r], [Tz_r]]))
    bL = np.zeros(shape=(3, 1))

    L_disp_solver_zc = disp_solver_zc(type_amp=disp_info[1], beta_sw=beta_sw, Tsw=Tsw, A=AL, b=bL)    #这里只用到z方向的位移
    R_disp_solver_zc = disp_solver_zc(type_amp=disp_info[1], beta_sw=beta_sw, Tsw=Tsw, A=AR, b=bR)

    # initial vector
    Zcl0 = Zc[0].ravel()
    Zcr0 = Zc[1].ravel()

    # Solve Zcl, Zcr based on NR iteration
    Zcr = R_disp_solver_zc.NewtonIteration(xk=Zcr0, max_iter=1000, tolerance=1e-8)
    Zcr.resize(img_size)

    Zcl = L_disp_solver_zc.NewtonIteration(xk=Zcl0, max_iter=1000, tolerance=1e-8)
    Zcl.resize(img_size)

    # Zc -> (Xw,Yw,Zw)
    Xw_l = np.zeros(shape=img_size)
    Yw_l = np.zeros(shape=img_size)
    Zw_l = np.zeros(shape=img_size)
    Xw_r = np.zeros(shape=img_size)
    Yw_r = np.zeros(shape=img_size)
    Zw_r = np.zeros(shape=img_size)

    for i in range(img_size[0]):
        for j in range(img_size[1]):
            vec_r = np.dot(iRotR, np.array([Zcr[i, j] * (j + 1 - cx_r) / fx_r - Tx_r, Zcr[i, j] * (i + 1 - cy_r) / fy_r - Ty_r, Zcr[i, j] - Tz_r]))
            vec_l = np.array([Zcl[i, j] * (j + 1 - cx_l) / fx_l, Zcl[i, j] * (i + 1 - cy_l) / fy_l, Zcl[i, j]])
            Xw_r[i][j] = vec_r[0]
            Yw_r[i][j] = vec_r[1]
            Zw_r[i][j] = vec_r[2]
            Xw_l[i][j] = vec_l[0]
            Yw_l[i][j] = vec_l[1]
            Zw_l[i][j] = vec_l[2]
    # Ow -> Os
    Ls_d = transform_w2s(beta_sw, (Xw_l, Yw_l, Zw_l), Tsw=Tsw)
    Rs_d = transform_w2s(beta_sw, (Xw_r, Yw_r, Zw_r), Tsw=Tsw)

    # Disparity after deformation
    R_Def_U, R_Def_V = projection(params["RRot"], params["RTX"], params["RTY"], params["RTZ"], params["RFX"],
                                  params["RFY"], params["RCX"], params["RCY"], (Xw_l, Yw_l, Zw_l))
    L_Def_U = np.expand_dims(np.arange(1, R_Ref_U.shape[1] + 1, 1), 0).repeat(R_Ref_U.shape[0], axis=0)
    L_Def_V = np.expand_dims(np.arange(1, R_Ref_U.shape[0] + 1, 1), 1).repeat(R_Ref_U.shape[1], axis=1)
    D_Disparity_X = R_Def_U - L_Def_U
    D_Disparity_Y = R_Def_V - L_Def_V

    disp_solver_xsys = disp_solver_xys(type_amp=disp_info[0])
    L_initxs = Ls_d[0].ravel()
    L_initys = Ls_d[1].ravel()
    L_Xs_0, L_Ys_0 = disp_solver_xsys.NewtonIteration(xs0=L_initxs, ys0=L_initys, xs=L_initxs, ys=L_initys, tolerance=1e-8, max_iter=1000)
    L_Xs_0.resize(img_size)
    L_Ys_0.resize(img_size)

    R_initxs = Rs_d[0].ravel()
    R_initys = Rs_d[1].ravel()
    R_Xs_0, R_Ys_0 = disp_solver_xsys.NewtonIteration(R_initxs, R_initys, R_initxs, R_initys, tolerance=1e-8, max_iter=1000)
    R_Xs_0.resize(img_size)
    R_Ys_0.resize(img_size)
    #
    # UV_l = calculatedisp(L_Xs_0, L_Ys_0, disp_info[0])
    # UV_r = calculatedisp(R_Xs_0, R_Ys_0, disp_info[0])
    # W_l = Ls_d[2] - 0
    # W_r = Rs_d[2] - 0
    # L_Zs_2 = calculatedisp_ws(Ls_d[0], Ls_d[1], dispinfo=disp_info[1])

    # plt.subplot(3, 2, 1)
    # plt.imshow(W_l)
    # plt.colorbar()
    # plt.subplot(3, 2, 2)
    # plt.imshow(L_Zs_2)
    # plt.colorbar()
    # plt.subplot(3, 2, 3)
    # plt.imshow(Ls_d[0] - L_Xs_0)
    # plt.colorbar()
    # plt.subplot(3, 2, 4)
    # plt.imshow(UV_l[0])
    # plt.colorbar()
    # plt.subplot(3, 2, 5)
    # plt.imshow(Rs_d[1] - R_Ys_0)
    # plt.colorbar()
    # plt.subplot(3, 2, 6)
    # plt.imshow(UV_r[1])
    # plt.colorbar()
    # plt.show()

    return (Ls_d[0] - L_Xs_0, Ls_d[1] - L_Ys_0, Ls_d[2]), (Rs_d[0] - R_Xs_0, Rs_d[1] - R_Ys_0, Rs_d[2]), (L_Xs_0, L_Ys_0), (R_Xs_0, R_Ys_0), (D_Disparity_X, D_Disparity_Y), (Xw_l, Yw_l, Zw_l), (Xw_r, Yw_r, Zw_r)


def generate_imgs(LXs_ref, LYs_ref, RXs_ref, RYs_ref, LXs_def, LYs_def, RXs_def, RYs_def, speckle_density, speckle_size, randomseeds, zf=0):
    x_min = min([np.min(LXs_ref), np.min(LXs_def), np.min(RXs_ref), np.min(RXs_def)])
    x_max = max([np.max(LXs_ref), np.max(LXs_def), np.max(RXs_ref), np.max(RXs_def)])
    y_min = min([np.min(LYs_ref), np.min(LYs_def), np.min(RYs_ref), np.min(RYs_def)])
    y_max = max([np.max(LYs_ref), np.max(LYs_def), np.max(RYs_ref), np.max(RYs_def)])
    if not zf:
        zf = max([(x_max - x_min)/LXs_def.shape[0], (y_max - y_min)/LXs_def.shape[1]])
    LRX_loc = (LXs_ref - x_min) / zf
    LDX_loc = (LXs_def - x_min) / zf
    RRX_loc = (RXs_ref - x_min) / zf
    RDX_loc = (RXs_def - x_min) / zf
    LRY_loc = (LYs_ref - y_min) / zf
    LDY_loc = (LYs_def - y_min) / zf
    RRY_loc = (RYs_ref - y_min) / zf
    RDY_loc = (RYs_def - y_min) / zf

    length_x = LXs_ref.shape[0]
    length_y = LXs_ref.shape[1]

    speckle_map, direction_map, Rx_map, Ry_map = generate_seedmap((length_x, length_y), speckle_density, speckle_size, randomseeds[:-1])

    LR_img = np.zeros_like(LXs_ref)
    LD_img = np.zeros_like(LXs_ref)
    RR_img = np.zeros_like(LXs_ref)
    RD_img = np.zeros_like(LXs_ref)

    step = int(speckle_size * 0.5 + 1)
    box_size = 80

    for i in range(0, length_x, step):
        print('\r', '%s finished ' % (str(np.ceil(i * 100 / length_x)) + '%'), end='\b')
        for j in range(0, length_y, step):
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

                asumed_loc = (i, j)
                add_i = -box_size
                add_j = -box_size
                box_sx = (asumed_loc[0] - box_size) * (asumed_loc[0] - box_size >= 0)
                box_ex = (asumed_loc[0] + box_size - length_x) * (asumed_loc[0] + box_size <= length_x) + length_x
                box_sy = (asumed_loc[1] - box_size) * (asumed_loc[1] - box_size >= 0)
                box_ey = (asumed_loc[1] + box_size - length_y) * (asumed_loc[1] + box_size <= length_y) + length_y
                LRX_minus = LRX_loc - i
                LRY_minus = LRY_loc - j
                radius = LRX_minus[box_sx: box_ex, box_sy: box_ey] ** 2 + LRY_minus[box_sx: box_ex, box_sy: box_ey] ** 2
                if i < box_size:
                    add_i = -(box_size - (2 * box_size-radius.shape[0]))
                if j < box_size:
                    add_j = -(box_size - (2 * box_size-radius.shape[1]))

                P_LR = (np.argmin(np.min(radius, axis=1)) + add_i + i, np.argmin(np.min(radius, axis=0)) + add_j + j)
                # Left Reference
                # LRX_minus = LRX_loc - i
                # LRY_minus = LRY_loc - j
                # P_LR = (np.argmin(np.min(np.abs(LRX_minus), axis=1)), np.argmin(np.min(np.abs(LRY_minus), axis=0)))
                # radius = LRX_minus ** 2 + LRY_minus ** 2
                # P_LR = (np.argmin(np.min(radius, axis=1)), np.argmin(np.min(radius, axis=0)))
                startx = int(np.rint((P_LR[0] - 4 * Rxtemp) * (P_LR[0] - 4 * Rxtemp >= 0)))
                starty = int(np.rint((P_LR[1] - 4 * Rytemp) * (P_LR[1] - 4 * Rytemp >= 0)))
                stopx = int(np.rint((P_LR[0] + 4 * Rxtemp - length_x) * (P_LR[0] + 4 * Rxtemp < length_x)) + length_x)
                stopy = int(np.rint((P_LR[1] + 4 * Rytemp - length_y) * (P_LR[1] + 4 * Rytemp < length_y)) + length_y)    # 计算子块的始末坐标
                x = LRX_minus[startx:stopx, starty:stopy]
                y = LRY_minus[startx:stopx, starty:stopy]
                LR_img[startx:stopx, starty:stopy] += np.exp(- ((x * invsigma[0, 0] + y * invsigma[1, 0]) * x + (x * invsigma[0, 1] + y * invsigma[1, 1]) * y) / 2)


                LDX_minus = LDX_loc - i
                LDY_minus = LDY_loc - j
                radius = LDX_minus[box_sx: box_ex, box_sy: box_ey] ** 2 + LDY_minus[box_sx: box_ex, box_sy: box_ey] ** 2
                P_LD = (np.argmin(np.min(radius, axis=1)) + add_i + i, np.argmin(np.min(radius, axis=0)) + add_j + j)
                startx = int(np.rint((P_LD[0] - 4 * Rxtemp) * (P_LD[0] - 4 * Rxtemp >= 0)))
                starty = int(np.rint((P_LD[1] - 4 * Rytemp) * (P_LD[1] - 4 * Rytemp >= 0)))
                stopx = int(np.rint((P_LD[0] + 4 * Rxtemp - length_x) * (P_LD[0] + 4 * Rxtemp < length_x)) + length_x)
                stopy = int(np.rint((P_LD[1] + 4 * Rytemp - length_y) * (P_LD[1] + 4 * Rytemp < length_y)) + length_y)  # 计算子块的始末坐标
                x = LDX_minus[startx:stopx, starty:stopy]
                y = LDY_minus[startx:stopx, starty:stopy]
                LD_img[startx:stopx, starty:stopy] += np.exp(- ((x * invsigma[0, 0] + y * invsigma[1, 0]) * x + (x * invsigma[0, 1] + y * invsigma[1, 1]) * y) / 2)

                RRX_minus = RRX_loc - i
                RRY_minus = RRY_loc - j
                radius = RRX_minus[box_sx: box_ex, box_sy: box_ey] ** 2 + RRY_minus[box_sx: box_ex, box_sy: box_ey] ** 2
                P_RR = (np.argmin(np.min(radius, axis=1)) + add_i + i, np.argmin(np.min(radius, axis=0)) + add_j + j)
                startx = int(np.rint((P_RR[0] - 4 * Rxtemp) * (P_RR[0] - 4 * Rxtemp >= 0)))
                starty = int(np.rint((P_RR[1] - 4 * Rytemp) * (P_RR[1] - 4 * Rytemp >= 0)))
                stopx = int(np.rint((P_RR[0] + 4 * Rxtemp - length_x) * (P_RR[0] + 4 * Rxtemp < length_x)) + length_x)
                stopy = int(np.rint((P_RR[1] + 4 * Rytemp - length_y) * (P_RR[1] + 4 * Rytemp < length_y)) + length_y)    # 计算子块的始末坐标
                x = RRX_minus[startx:stopx, starty:stopy]
                y = RRY_minus[startx:stopx, starty:stopy]
                RR_img[startx:stopx, starty:stopy] += np.exp(- ((x * invsigma[0, 0] + y * invsigma[1, 0]) * x + (x * invsigma[0, 1] + y * invsigma[1, 1]) * y) / 2)

                RDX_minus = RDX_loc - i
                RDY_minus = RDY_loc - j
                radius = RDX_minus[box_sx: box_ex, box_sy: box_ey] ** 2 + RDY_minus[box_sx: box_ex, box_sy: box_ey] ** 2
                P_RD = (np.argmin(np.min(radius, axis=1)) + add_i + i, np.argmin(np.min(radius, axis=0)) + add_j + j)
                startx = int(np.rint((P_RD[0] - 4 * Rxtemp) * (P_RD[0] - 4 * Rxtemp >= 0)))
                starty = int(np.rint((P_RD[1] - 4 * Rytemp) * (P_RD[1] - 4 * Rytemp >= 0)))
                stopx = int(np.rint((P_RD[0] + 4 * Rxtemp - length_x) * (P_RD[0] + 4 * Rxtemp < length_x)) + length_x)
                stopy = int(np.rint((P_RD[1] + 4 * Rytemp - length_y) * (P_RD[1] + 4 * Rytemp < length_y)) + length_y)    # 计算子块的始末坐标
                x = RDX_minus[startx:stopx, starty:stopy]
                y = RDY_minus[startx:stopx, starty:stopy]
                RD_img[startx:stopx, starty:stopy] += np.exp(- ((x * invsigma[0, 0] + y * invsigma[1, 0]) * x + (x * invsigma[0, 1] + y * invsigma[1, 1]) * y) / 2)

    np.random.seed(randomseeds[-1])
    bk = np.random.randint(30, 80)
    LR_img = array2img(LR_img, background=bk, noise=np.random.randint(20, 30))
    LD_img = array2img(LD_img, background=bk, noise=np.random.randint(20, 30))
    RR_img = array2img(RR_img, background=bk, noise=np.random.randint(20, 30))
    RD_img = array2img(RD_img, background=bk, noise=np.random.randint(20, 30))

    return LR_img, LD_img, RR_img, RD_img


def projection(rot_array, tx, ty, tz, fx, fy, cx, cy, word_coord):
    Xw = word_coord[0]
    Yw = word_coord[1]
    Zw = word_coord[2]

    Xc = rot_array[0][0] * Xw + rot_array[0][1] * Yw + rot_array[0][2] * Zw + tx
    Yc = rot_array[1][0] * Xw + rot_array[1][1] * Yw + rot_array[1][2] * Zw + ty
    Zc = rot_array[2][0] * Xw + rot_array[2][1] * Yw + rot_array[2][2] * Zw + tz

    u = fx * Xc / Zc + cx
    v = fy * Yc / Zc + cy

    return u, v


def UVW2FLow():
    with open("./Seeds/States.json", 'r') as f:
        params = json.load(f)
    stid = 650

    coord_l_X = np.loadtxt(params["dataset_savepath"] + "/coordinates/X.csv", delimiter=",")
    coord_l_Y = np.loadtxt(params["dataset_savepath"] + "/coordinates/Y.csv", delimiter=",")
    coord_l_Z = np.loadtxt(params["dataset_savepath"] + "/coordinates/Z.csv", delimiter=",")

    for i in range(1000):
        print('\r', '%d of %d finished ' % (i, 1000), end='\b')
        if os.path.exists(params["dataset_savepath"] + str(i) + "L_flow.png"):
            continue
        if i < stid:
            continue

        coord_l_x = np.loadtxt(params["dataset_savepath"]+str(i)+"_LWU.csv") + coord_l_X
        coord_l_y = np.loadtxt(params["dataset_savepath"]+str(i)+"_LWV.csv") + coord_l_Y
        coord_l_z = np.loadtxt(params["dataset_savepath"]+str(i)+"_LWW.csv") + coord_l_Z

        u, v = projection(params["LRot"], params["LTX"], params["LTY"], params["LTZ"], params["LFX"], params["LFY"], params["LCX"], params["LCY"], (coord_l_x, coord_l_y, coord_l_z))

        u0 = np.expand_dims(np.arange(1, u.shape[1] + 1, 1), 0).repeat(u.shape[0], axis=0)
        v0 = np.expand_dims(np.arange(1, u.shape[0] + 1, 1), 1).repeat(u.shape[1], axis=1)

        disp_x = u - u0
        disp_y = v - v0

        np.savetxt(params["dataset_savepath"] + str(i) + "LFU.csv", disp_x)
        np.savetxt(params["dataset_savepath"] + str(i) + "LFV.csv", disp_y)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(disp_x)
        plt.title("LFX")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(disp_y)
        plt.title("LFY")
        plt.colorbar()
        plt.savefig(params["dataset_savepath"] + str(i) + "L_flow.png")

        # # plt.show()
        plt.close()

if __name__ == '__main__':
    with open("./Seeds/States.json", 'r') as f:
        params = json.load(f)
    plane_equ, beta_sw, Tsw = cal_plane_equ(params)
    print(plane_equ)

    start_idx = 0
    seed_array = np.loadtxt("./Seeds/Seeds_3.csv", delimiter=",")

    L_W_plane, R_W_plane, Zc = cal_plane_world_ordinate(plane_equ, params)

    # Cal ref disparity
    R_Ref_U, R_Ref_V = projection(params["RRot"], params["RTX"], params["RTY"], params["RTZ"], params["RFX"], params["RFY"], params["RCX"], params["RCY"], L_W_plane)
    L_Ref_U = np.expand_dims(np.arange(1, R_Ref_U.shape[1]+1, 1), 0).repeat(R_Ref_U.shape[0], axis=0)
    L_Ref_V = np.expand_dims(np.arange(1, R_Ref_U.shape[0] + 1, 1), 1).repeat(R_Ref_U.shape[1], axis=1)
    R_Disparity_X = R_Ref_U - L_Ref_U
    R_Disparity_Y = R_Ref_V - L_Ref_V

    Xs_l = transform_w2s(beta_sw, L_W_plane, Tsw=Tsw)
    Xs_r = transform_w2s(beta_sw, R_W_plane, Tsw=Tsw)

    # save coordinates
    np.savetxt(params["dataset_savepath"] + "/coordinates/X.csv", L_W_plane[0], delimiter=",")
    np.savetxt(params["dataset_savepath"] + "/coordinates/Y.csv", L_W_plane[1], delimiter=",")
    np.savetxt(params["dataset_savepath"] + "/coordinates/Z.csv", L_W_plane[2], delimiter=",")

    np.savetxt(params["dataset_savepath"] + "0_Disparity_RX.csv", -np.flip(np.flip(R_Disparity_X, axis=0), axis=1))
    np.savetxt(params["dataset_savepath"] + "0_Disparity_RY.csv", -np.flip(np.flip(R_Disparity_Y, axis=0), axis=1))

    disp_lib = ["planer", "sin"]

    for k in range(seed_array.shape[0] - start_idx):
        i = k + start_idx
        if os.path.exists(params["dataset_savepath"] + str(i) + "_LR.tif"):
            continue
        print("%d of %i"%(i, seed_array.shape[0]))
        disp_info = [[], []]
        tem_line = seed_array[i]
        for j in range(5):
            tem_disp_xy = []
            type_xy = disp_lib[int(tem_line[j * 11])]
            tem_disp_xy.append(type_xy)
            tem_disp_xy += list(tem_line[j * 11 + 1: j * 11 + 11][:])
            disp_info[0].append(tem_disp_xy)

            tem_disp_z = []
            type_z = disp_lib[int(tem_line[55 + j * 6])]
            tem_disp_z.append(type_z)
            tem_disp_z += list(tem_line[55 + j * 6 + 1: 55 + j * 6 + 6][:])
            disp_info[1].append(tem_disp_z)

        tem_speckle_size = tem_line[55 + 30 + 1]
        tem_speckle_density = tem_line[55 + 30 + 2]

        randomseed = tem_line[55 + 30 + 3:].astype("int32")

        (U_l, V_l, W_l), (U_r, V_r, W_r), (L_Xs_0, L_Ys_0), (R_Xs_0, R_Ys_0), (D_Disparity_X, D_Disparity_Y), L_W_Def, R_W_def= generate_displacement(params, beta_sw, Tsw, Zc, disp_info=disp_info)    #[(XsYs), (Zs)]

        # Os -> Ow
        (U_w_l, V_w_l, W_w_l) = transform_w2s(np.linalg.inv(beta_sw), (U_l, V_l, W_l))

        # UVW in Ow
        np.savetxt(params["dataset_savepath"] + str(i) + "_LWU.csv", -np.flip(np.flip(U_w_l, axis=0), axis=1))
        np.savetxt(params["dataset_savepath"] + str(i) + "_LWV.csv", -np.flip(np.flip(V_w_l, axis=0), axis=1))
        np.savetxt(params["dataset_savepath"] + str(i) + "_LWW.csv", -np.flip(np.flip(W_w_l, axis=0), axis=1))

        # save disparity
        np.savetxt(params["dataset_savepath"] + str(i) + "_Disparity_DX.csv", -np.flip(np.flip(D_Disparity_X, axis=0), axis=1))
        np.savetxt(params["dataset_savepath"] + str(i) + "_Disparity_DY.csv", -np.flip(np.flip(D_Disparity_Y, axis=0), axis=1))

        LR_img, LD_img, RR_img, RD_img = generate_imgs(Xs_l[0], Xs_l[1], Xs_r[0], Xs_r[1], L_Xs_0,  L_Ys_0, R_Xs_0, R_Ys_0, speckle_density=tem_speckle_density, speckle_size=tem_speckle_size, randomseeds=randomseed)

        # Inverse Image and Save File
        [LR_img, LD_img, RR_img, RD_img] = img_flip([LR_img, LD_img, RR_img, RD_img])
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_LR.tif", LR_img)
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_LD.tif", LD_img)
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_RR.tif", RR_img)
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_RD.tif", RD_img)

        (Uw_l, Vw_l ,Ww_l) = transform_w2s(np.linalg.inv(beta_sw), (U_l, V_l, W_l))

        plt.figure()
        plt.subplot(3, 2, 1)
        plt.imshow(U_l)
        plt.title("U")
        plt.colorbar()
        plt.subplot(3, 2, 2)
        plt.imshow(V_l)
        plt.title("V")
        plt.colorbar()
        plt.subplot(3, 2, 3)
        plt.imshow(W_l)
        plt.title("W")
        plt.colorbar()
        plt.subplot(3, 2, 4)
        plt.imshow(D_Disparity_X)
        plt.title("DX")
        plt.colorbar()
        plt.subplot(3, 2, 5)
        plt.imshow(D_Disparity_Y)
        plt.title("DY")
        plt.colorbar()
        plt.subplot(3, 2, 6)
        plt.imshow(W_l)
        plt.title("W")
        plt.colorbar()
        plt.savefig(params["dataset_savepath"] + str(i) + "disp.png")
        plt.close()

    # Calculate Flow Label of Left camera if necessary. The labels are not exactly accurate.
    UVW2FLow()