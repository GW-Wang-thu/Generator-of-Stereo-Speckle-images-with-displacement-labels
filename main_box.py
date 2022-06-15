import numpy as np
import json
from utils.utils import generate_seedmap, array2img, cal_successive_disp, disp_minus, cut_blocks_coordinates, recover_cuts, add_range_bk
from utils.dispsolver_cuda import disp_solver_zc, disp_solver_xys
from utils.cuda_funcs import get_coord_numba, get_disp_numba, get_A_numba, generate_iter_image_nonlocal
import cv2
import matplotlib.pyplot as plt
import os
import time

import warnings
warnings.filterwarnings("ignore")


def cal_plane_equ(params):
    print("*********Calculate Coordinates********")
    Zc = params["Zcl"]
    Os = np.array([[0], [0], [Zc]])
    # Angular bisector of the two cameras is the direction of Zs
    RRot = params["RRot"]
    RT = [[-params["RTX"]], [-params["RTY"]], [-params["RTZ"]]]
    Or_w = np.dot(np.linalg.inv(np.array(RRot)), np.array(RT))
    Pr001 = np.dot(np.linalg.inv(np.array(RRot)), np.array([[0], [0], [1000000000000000]]) - np.array(RT))
    n_r = (Or_w - Pr001) / np.linalg.norm((Or_w - Pr001))
    print("**Normal Direction of Right Camera:\n", n_r)
    z_s = (np.array([[0], [0], [-1]]) + n_r) / np.linalg.norm(np.array([[0], [0], [-1]]) + n_r) # direction of Zs
    print("**Normal Direction of Surface Plane:\n", z_s)
    # affine transpose matrix between Os and Ow
    y_s = np.array([[1], [0], [-z_s[0][0] / z_s[2][0]]]) / np.linalg.norm(np.array([[1], [0], [-z_s[0][0] / z_s[2][0]]]))
    x_s = np.array([[y_s[2][0] * z_s[1][0] / (z_s[2][0]-y_s[2][0]*z_s[0][0])], [1], [-z_s[1][0] / (z_s[2][0]-y_s[2][0]*z_s[0][0])]]) / np.linalg.norm(np.array([[y_s[2][0] * z_s[1][0] / (z_s[2][0]-y_s[2][0]*z_s[0][0])], [1], [-z_s[1][0] / (z_s[2][0]-y_s[2][0]*z_s[0][0])]]))
    beta_sw = np.array([[x_s[0][0], x_s[1][0], x_s[2][0]], [y_s[0][0], y_s[1][0], y_s[2][0]], [z_s[0][0], z_s[1][0], z_s[2][0]]])
    Tsw = np.array([[-Zc * beta_sw[0][2]], [-Zc * beta_sw[1][2]], [-Zc * beta_sw[2][2]]])
    # Os surface equation
    A = - z_s[0][0] / z_s[2][0]
    B = - z_s[1][0] / z_s[2][0]
    C = Zc
    return (A, B, C), beta_sw, Tsw


def cal_plane_world_ordinate(plane_param, imaging_params):
    img_size = imaging_params["RResolution"]
    box_size = imaging_params["box_size"]
    stp = imaging_params["start_pos"]
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
    inv_RRot = np.linalg.inv(np.array(RRot)).astype("float32")

    top_r = C + Tx_r * (inv_RRot[2][0] - A * inv_RRot[0][0] - B * inv_RRot[1][0]) \
            + Ty_r * (inv_RRot[2][1] - A * inv_RRot[0][1] - B * inv_RRot[1][1]) \
            + Tz_r * (inv_RRot[2][2] - A * inv_RRot[0][2] - B * inv_RRot[1][2])
    top_l = C
    btm_r = np.array([[(inv_RRot[2][0] - A * inv_RRot[0][0] - B * inv_RRot[1][0])/fx_r, (inv_RRot[2][1] - A * inv_RRot[0][1] - B * inv_RRot[1][1])/fy_r, inv_RRot[2][2] - A * inv_RRot[0][2] - B * inv_RRot[1][2]]], dtype="float32")
    btm_l = np.array([[-A/fx_l, -B/fy_l, 1]], dtype="float32")

    Xw_r = np.zeros(shape=box_size, dtype="float32")
    Yw_r = np.zeros(shape=box_size, dtype="float32")
    Zw_r = np.zeros(shape=box_size, dtype="float32")
    Xw_l = np.zeros(shape=box_size, dtype="float32")
    Yw_l = np.zeros(shape=box_size, dtype="float32")
    Zw_l = np.zeros(shape=box_size, dtype="float32")
    Zc_l = np.zeros(shape=box_size, dtype="float32")
    Zc_r = np.zeros(shape=box_size, dtype="float32")
    # Solve initial plane projected on the 2 cameras' pixel coordinates. Accelerate with Numba if equipped.
    Xw_l, Yw_l, Zw_l, Xw_r, Yw_r, Zw_r, Zc_l, Zc_r = get_coord_numba(box_size, stp,
                                                       btm_l, btm_r,
                                                       cx_l, cx_r,
                                                       cy_l, cy_r,
                                                       top_l, top_r,
                                                       inv_RRot,
                                                       fx_l, fx_r,
                                                       fy_l, fy_r,
                                                       Tx_r, Ty_r, Tz_r,
                                                       Xw_l, Yw_l, Zw_l, Xw_r, Yw_r, Zw_r, Zc_l, Zc_r)

    return (Xw_l, Yw_l, Zw_l), (Xw_r, Yw_r, Zw_r), (Zc_l, Zc_r)


def transform_w2s(beta_sw, world_coordinate, Tsw=[[0.0], [0.0], [0.0]]):
    '''从世界坐标系到物面坐标系转换'''
    Xw = world_coordinate[0]
    Yw = world_coordinate[1]
    Zw = world_coordinate[2]
    Xs = beta_sw[0][0] * Xw + beta_sw[0][1] * Yw + beta_sw[0][2] * Zw + Tsw[0][0]
    Ys = beta_sw[1][0] * Xw + beta_sw[1][1] * Yw + beta_sw[1][2] * Zw + Tsw[1][0]
    Zs = beta_sw[2][0] * Xw + beta_sw[2][1] * Yw + beta_sw[2][2] * Zw + Tsw[2][0]
    return (Xs, Ys, Zs)


def generate_displacement(imaging_params, beta_sw, Tsw, Zc, disp_info):
    box_size = imaging_params["box_size"]
    stp = imaging_params["start_pos"]
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
    AL = np.zeros(shape=(3, box_size[0] * box_size[1]), dtype="float32")
    AR = np.zeros_like(AL).astype("float32")
    RRot = params["RRot"]
    iRotR = np.linalg.inv(np.array(RRot)).astype("float32")

    AL, AR = get_A_numba(box_size, stp, iRotR, cx_l, cx_r, cy_l, cy_r, fx_l, fx_r, fy_l, fy_r, AR, AL)

    bR = - np.dot(iRotR, np.array([[Tx_r], [Ty_r], [Tz_r]]))
    bL = np.zeros(shape=(3, 1))

    L_disp_solver_zc = disp_solver_zc(type_amp=disp_info[1], beta_sw=beta_sw, Tsw=Tsw, A=AL, b=bL)    #这里只用到z方向的位移
    R_disp_solver_zc = disp_solver_zc(type_amp=disp_info[1], beta_sw=beta_sw, Tsw=Tsw, A=AR, b=bR)

    # Initial vector Zcl0, Zcr0 is selected as the planer results.
    Zcl0 = Zc[0].ravel()
    Zcr0 = Zc[1].ravel()

    # Solve Zc iteratively, off-plane displacement is given first.
    Zcr = R_disp_solver_zc.NewtonIteration(xk=Zcr0, max_iter=1000, tolerance=1e-9)
    Zcr.resize(box_size)

    Zcl = L_disp_solver_zc.NewtonIteration(xk=Zcl0, max_iter=1000, tolerance=1e-9)
    Zcl.resize(box_size)

    # Calculate world coordinate using Zc
    Xw_l = np.zeros(shape=box_size, dtype="float32")
    Yw_l = np.zeros(shape=box_size, dtype="float32")
    Zw_l = np.zeros(shape=box_size, dtype="float32")
    Xw_r = np.zeros(shape=box_size, dtype="float32")
    Yw_r = np.zeros(shape=box_size, dtype="float32")
    Zw_r = np.zeros(shape=box_size, dtype="float32")
    Xw_l, Yw_l, Zw_l, Xw_r, Yw_r, Zw_r = get_disp_numba(box_size, stp, iRotR, Zcl, Zcr, cx_l, cx_r, cy_l, cy_r, fx_l, fx_r, fy_l, fy_r, Tx_r, Ty_r, Tz_r, Xw_l, Yw_l, Zw_l, Xw_r, Yw_r, Zw_r)

    # Transpose world coordinates to surface
    Ls_d = transform_w2s(beta_sw, (Xw_l, Yw_l, Zw_l), Tsw=Tsw)
    Rs_d = transform_w2s(beta_sw, (Xw_r, Yw_r, Zw_r), Tsw=Tsw)

    # Calculate disparity using the disparity using the calculated coordinates
    R_Def_U, R_Def_V = projection(params["RRot"], params["RTX"], params["RTY"], params["RTZ"], params["RFX"],
                                  params["RFY"], params["RCX"], params["RCY"], (Xw_l, Yw_l, Zw_l))
    L_Def_U = np.expand_dims(np.arange(params["start_pos"][1], R_Ref_U.shape[1] + params["start_pos"][1], 1), 0).repeat(R_Ref_U.shape[0], axis=0)
    L_Def_V = np.expand_dims(np.arange(params["start_pos"][0], R_Ref_U.shape[0] + params["start_pos"][0], 1), 1).repeat(R_Ref_U.shape[1], axis=1)
    # equal to the following code
    # L_Def_Uc, L_Def_Vc = projection(params["LRot"], params["LTX"], params["LTY"], params["LTZ"], params["LFX"],
    #                               params["LFY"], params["LCX"], params["LCY"], (Xw_l, Yw_l, Zw_l))
    D_Disparity_X = R_Def_U - L_Def_U
    D_Disparity_Y = R_Def_V - L_Def_V

    # solve in-plane displacement using newton iteration, based on the coordinates solved above.
    disp_solver_xsys = disp_solver_xys(type_amp=disp_info[0])
    L_initxs = Ls_d[0].ravel()
    L_initys = Ls_d[1].ravel()
    L_Xs_0, L_Ys_0 = disp_solver_xsys.NewtonIteration(xs0=L_initxs, ys0=L_initys, xs=L_initxs, ys=L_initys, tolerance=1e-9, max_iter=1000)
    L_Xs_0.resize(box_size)
    L_Ys_0.resize(box_size)

    R_initxs = Rs_d[0].ravel()
    R_initys = Rs_d[1].ravel()
    R_Xs_0, R_Ys_0 = disp_solver_xsys.NewtonIteration(R_initxs, R_initys, R_initxs, R_initys, tolerance=1e-9, max_iter=1000)
    R_Xs_0.resize(box_size)
    R_Ys_0.resize(box_size)

    return (Ls_d[0] - L_Xs_0, Ls_d[1] - L_Ys_0, Ls_d[2]), (Rs_d[0] - R_Xs_0, Rs_d[1] - R_Ys_0, Rs_d[2]), (L_Xs_0, L_Ys_0), (R_Xs_0, R_Ys_0), (D_Disparity_X, D_Disparity_Y), (Xw_l, Yw_l, Zw_l), (Xw_r, Yw_r, Zw_r)


def generate_imgs(coordinate_list, speckle_density, speckle_size, randomseeds, num_cut, cache_dir, bkcolor="black"):
    x_min, y_min, x_max, y_max = np.min(coordinate_list[0][0]), np.min(coordinate_list[0][1]), np.max(coordinate_list[0][0]), np.max(coordinate_list[0][1])
    x_minmax_list = []
    y_minmax_list = []
    for i in range(len(coordinate_list)):
        tem_xmin = np.min(coordinate_list[i][0])
        tem_xmax = np.max(coordinate_list[i][0])
        tem_ymin = np.min(coordinate_list[i][1])
        tem_ymax = np.max(coordinate_list[i][1])
        x_min = min(x_min, tem_xmin)
        y_min = min(y_min, tem_ymin)
        x_max = max(x_max, tem_xmax)
        y_max = max(y_max, tem_ymax)
        x_minmax_list.append((tem_xmin, tem_xmax))
        y_minmax_list.append((tem_ymin, tem_ymax))

    zf = max([(x_max - x_min)/(coordinate_list[0][0].shape[0]*num_cut), (y_max - y_min)/(coordinate_list[0][0].shape[1]*num_cut)])

    coord_list = []
    for i in range(len(coordinate_list)):
        coord_list.append([(coordinate_list[i][0] - x_min) / zf, (coordinate_list[i][1] - y_min) / zf])
        x_minmax_list[i] = ((x_minmax_list[i][0] - x_min) / zf, (x_minmax_list[i][1] - x_min) / zf)
        y_minmax_list[i] = ((y_minmax_list[i][0] - y_min) / zf, (y_minmax_list[i][1] - y_min) / zf)

    length_x = coordinate_list[0][0].shape[0] * num_cut
    length_y = coordinate_list[0][0].shape[1] * num_cut
    cutsize_x = coordinate_list[0][0].shape[0]
    cutsize_y = coordinate_list[0][0].shape[1]

    speckle_map, direction_map, Rx_map, Ry_map = generate_seedmap((length_x, length_y), speckle_density, speckle_size, randomseeds[:-1])
    imgs = []

    step = int(speckle_size * 0.5 + 1)
    box_size = 40

    for m in range(len(coord_list)):
        tem_coord_x = coord_list[m][0]
        tem_coord_y = coord_list[m][1]
        temp_img = np.zeros_like(coordinate_list[m][0])
        print('\r', 'Processing block %d of %d... ' % (m + 1, len(coordinate_list)), end="\b")
        temp_img = generate_iter_image_nonlocal(temp_img=temp_img.copy(),
                                                length_x=length_x,
                                                length_y=length_y,
                                                step=step,
                                                x_minmax=x_minmax_list[m],
                                                y_minmax=y_minmax_list[m],
                                                box_size=box_size,
                                                speckle_map=speckle_map.copy(),
                                                Rx_map=Rx_map.copy(),
                                                Ry_map=Ry_map.copy(),
                                                direction_map=direction_map.copy(),
                                                cutsize_x=cutsize_x,
                                                cutsize_y=cutsize_y,
                                                tem_coord_x=tem_coord_x.copy(),
                                                tem_coord_y=tem_coord_y.copy(),
                                                color=bkcolor)
        imgs.append(temp_img)
    np.random.seed(randomseeds[-1])
    if bkcolor == "black":
        bk = np.random.randint(30, 80)
    else:
        bk = np.random.randint(140, 190)

    for i in range(len(imgs)):
        imgs[i] = array2img(imgs[i], background=bk, noise=np.random.randint(20, 30), color=bkcolor)

    return imgs


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


if __name__ == '__main__':
    '''Load seed files'''
    with open("Seeds/States_box.json", 'r') as f:
        params = json.load(f)
    start_idx = 0      # To continue last generation from the given idx
    seed_array = np.loadtxt("./Seeds/Seeds_exp_box.csv", delimiter=",")
    sttime = time.perf_counter()

    '''Calculate basic coordinates and disparity of the planer surface'''
    plane_equ, beta_sw, Tsw = cal_plane_equ(params)
    L_W_plane, R_W_plane, Zc = cal_plane_world_ordinate(plane_equ, params)
    R_Ref_U, R_Ref_V = projection(params["RRot"], params["RTX"], params["RTY"], params["RTZ"], params["RFX"], params["RFY"], params["RCX"], params["RCY"], L_W_plane)
    L_Ref_U = np.expand_dims(np.arange(params["start_pos"][1], R_Ref_U.shape[1] + params["start_pos"][1], 1), 0).repeat(R_Ref_U.shape[0], axis=0)
    L_Ref_V = np.expand_dims(np.arange(params["start_pos"][0], R_Ref_U.shape[0] + params["start_pos"][0], 1), 1).repeat(R_Ref_U.shape[1], axis=1)
    R_Disparity_X = R_Ref_U - L_Ref_U
    R_Disparity_Y = R_Ref_V - L_Ref_V
    Xs_l = transform_w2s(beta_sw, L_W_plane, Tsw=Tsw)
    Xs_r = transform_w2s(beta_sw, R_W_plane, Tsw=Tsw)
    disp_lib = ["planer", "sin"]
    print("Time Consumption on Calculate Plane Coordinates: %d s" % (int(time.perf_counter() - sttime)))

    '''Save coordinates of the initial plane and disparity if needed'''
    save_init_plane_coord_and_disp = True
    if save_init_plane_coord_and_disp:
        np.save(params["dataset_savepath"] + "X.npy", L_W_plane[0])
        np.save(params["dataset_savepath"] + "Y.npy", L_W_plane[1])
        np.save(params["dataset_savepath"] + "Z.npy", L_W_plane[2])

        np.save(params["dataset_savepath"] + "Plane_Disparity_RX.npy", R_Disparity_X)
        np.save(params["dataset_savepath"] + "Plane_Disparity_RY.npy", R_Disparity_Y)

    '''Generate dataset for each case'''
    print("**********Generate Data sets*********")
    for k in range(seed_array.shape[0] - start_idx):
        sttime = time.perf_counter()
        i = k + start_idx
        if os.path.exists(params["dataset_savepath"] + str(i) + "_L0.tif"):
            continue
        print("******Generating %d of %d Data set*****"%(i, seed_array.shape[0]))
        disp_info = [[], []]
        disp_info_2 = [[], []]
        disp_info_3 = [[], []]
        tem_line = seed_array[i]
        '''Load Parameters from seed array'''
        # curved surface 1 of 3
        for j in range(5):  # 5 displacement components for a summation
            tem_disp_xy = []
            type_xy = disp_lib[int(tem_line[j * 11])]
            tem_disp_xy.append(type_xy)
            tem_disp_xy += list(tem_line[j * 11 + 1: j * 11 + 11][:])
            disp_info[0].append(tem_disp_xy)

            tem_disp_z = []
            type_z = disp_lib[int(tem_line[15*11 + j * 6])]
            tem_disp_z.append(type_z)
            tem_disp_z += list(tem_line[15*11 + j * 6 + 1: 15*11 + j * 6 + 6][:])
            disp_info[1].append(tem_disp_z)

        # curved surface 2 of 3
        for m in range(5):
            j = m + 5
            tem_disp_xy = []
            type_xy = disp_lib[int(tem_line[j * 11])]
            tem_disp_xy.append(type_xy)
            tem_disp_xy += list(tem_line[j * 11 + 1: j * 11 + 11][:])
            disp_info_2[0].append(tem_disp_xy)

            tem_disp_z = []
            type_z = disp_lib[int(tem_line[15*11 + j * 6])]
            tem_disp_z.append(type_z)
            tem_disp_z += list(tem_line[15*11 + j * 6 + 1: 15*11 + j * 6 + 6][:])
            disp_info_2[1].append(tem_disp_z)

        # curved surface 3 of 3
        for m in range(5):
            j = m + 10
            tem_disp_xy = []
            type_xy = disp_lib[int(tem_line[j * 11])]
            tem_disp_xy.append(type_xy)
            tem_disp_xy += list(tem_line[j * 11 + 1: j * 11 + 11][:])
            disp_info_3[0].append(tem_disp_xy)

            tem_disp_z = []
            type_z = disp_lib[int(tem_line[15*11 + j * 6])]
            tem_disp_z.append(type_z)
            tem_disp_z += list(tem_line[15*11 + j * 6 + 1: 15*11 + j * 6 + 6][:])
            disp_info_3[1].append(tem_disp_z)

        # shared speckle pattern
        tem_speckle_size = tem_line[17 * 15 + 1]
        tem_speckle_density = tem_line[17 * 15 + 2]
        randomseed = tem_line[17 * 15 + 3:].astype("int32")

        '''Calculate Displacements'''
        # solve displacement for 3 curved surface
        (U_l_1, V_l_1, W_l_1), (U_r_1, V_r_1, W_r_1), (L_Xs_0_1, L_Ys_0_1), (R_Xs_0_1, R_Ys_0_1), (D_Disparity_X_1, D_Disparity_Y_1), L_W_Def_1, R_W_def_1 = generate_displacement(params, beta_sw, Tsw, Zc, disp_info=disp_info)    #[(XsYs), (Zs)]
        (U_l_2, V_l_2, W_l_2), (U_r_2, V_r_2, W_r_2), (L_Xs_0_2, L_Ys_0_2), (R_Xs_0_2, R_Ys_0_2), (D_Disparity_X_2, D_Disparity_Y_2), L_W_Def_2, R_W_def_2 = generate_displacement(params, beta_sw, Tsw, Zc, disp_info=disp_info_2)  # [(XsYs), (Zs)]
        (U_l_3, V_l_3, W_l_3), (U_r_3, V_r_3, W_r_3), (L_Xs_0_3, L_Ys_0_3), (R_Xs_0_3, R_Ys_0_3), (D_Disparity_X_3, D_Disparity_Y_3), L_W_Def_3, R_W_def_3 = generate_displacement(params, beta_sw, Tsw, Zc, disp_info=disp_info_3)  # [(XsYs), (Zs)]
        print("Time Consumption on Solve Displacement iteratively: %d s" % (int(time.perf_counter() - sttime)))
        sttime = time.perf_counter()

        # calculate multi-scene displacement labels,
        U_l_12, V_l_12, W_l_12 = cal_successive_disp(previous_l_posi=(L_Xs_0_1, L_Ys_0_1), dispinfo=disp_info_2)
        U_l_13, V_l_13, W_l_13 = cal_successive_disp(previous_l_posi=(L_Xs_0_1, L_Ys_0_1), dispinfo=disp_info_3)
        U_l_23, V_l_23, W_l_23 = cal_successive_disp(previous_l_posi=(L_Xs_0_2, L_Ys_0_2), dispinfo=disp_info_3)

        (Us_l_01, Vs_l_01, Ws_l_01) = (U_l_1, V_l_1, W_l_1)
        (Us_l_12, Vs_l_12, Ws_l_12) = disp_minus((U_l_12, V_l_12, W_l_12), (U_l_1, V_l_1, W_l_1))
        (Us_l_13, Vs_l_13, Ws_l_13) = disp_minus((U_l_13, V_l_13, W_l_13), (U_l_1, V_l_1, W_l_1))
        (Us_l_23, Vs_l_23, Ws_l_23) = disp_minus((U_l_23, V_l_23, W_l_23), (U_l_2, V_l_2, W_l_2))

        (Uw_l_01, Vw_l_01, Ww_l_01) = transform_w2s(np.linalg.inv(beta_sw), (Us_l_01, Vs_l_01, Ws_l_01))
        (Uw_l_12, Vw_l_12, Ww_l_12) = transform_w2s(np.linalg.inv(beta_sw), (Us_l_12, Vs_l_12, Ws_l_12))
        (Uw_l_13, Vw_l_13, Ww_l_13) = transform_w2s(np.linalg.inv(beta_sw), (Us_l_13, Vs_l_13, Ws_l_13))
        (Uw_l_23, Vw_l_23, Ww_l_23) = transform_w2s(np.linalg.inv(beta_sw), (Us_l_23, Vs_l_23, Ws_l_23))

        '''Calculate Flow of left camera'''
        u_1, v_1 = projection(params["LRot"], params["LTX"], params["LTY"], params["LTZ"], params["LFX"], params["LFY"], params["LCX"], params["LCY"], L_W_Def_1)
        u_0, v_0 = projection(params["LRot"], params["LTX"], params["LTY"], params["LTZ"], params["LFX"], params["LFY"], params["LCX"], params["LCY"], L_W_plane)
        u_2, v_2 = projection(params["LRot"], params["LTX"], params["LTY"], params["LTZ"], params["LFX"], params["LFY"], params["LCX"], params["LCY"], L_W_Def_2)

        Flow_To_01 = (L_W_plane[0] + Uw_l_01, L_W_plane[1] + Vw_l_01, L_W_plane[2] + Ww_l_01)
        u_01, v_01 = projection(params["LRot"], params["LTX"], params["LTY"], params["LTZ"], params["LFX"], params["LFY"], params["LCX"], params["LCY"], Flow_To_01)
        Flow_01_x, Flow_01_y = u_01 - u_0, v_01 - v_0

        Flow_To_12 = (L_W_Def_1[0] + Uw_l_12, L_W_Def_1[1] + Vw_l_12, L_W_Def_1[2] + Ww_l_12)
        u_12, v_12 = projection(params["LRot"], params["LTX"], params["LTY"], params["LTZ"], params["LFX"], params["LFY"], params["LCX"], params["LCY"], Flow_To_12)
        Flow_12_x, Flow_12_y = u_12 - u_1, v_12 - v_1

        Flow_To_13 = (L_W_Def_1[0] + Uw_l_13, L_W_Def_1[1] + Vw_l_13, L_W_Def_1[2] + Ww_l_13)
        u_13, v_13 = projection(params["LRot"], params["LTX"], params["LTY"], params["LTZ"], params["LFX"], params["LFY"], params["LCX"], params["LCY"], Flow_To_13)
        Flow_13_x, Flow_13_y = u_13 - u_1, v_13 - v_1

        Flow_To_23 = (L_W_Def_2[0] + Uw_l_23, L_W_Def_2[1] + Vw_l_23, L_W_Def_2[2] + Ww_l_23)
        u_23, v_23 = projection(params["LRot"], params["LTX"], params["LTY"], params["LTZ"], params["LFX"], params["LFY"], params["LCX"], params["LCY"], Flow_To_23)
        Flow_23_x, Flow_23_y = u_23 - u_2, v_23 - v_2

        print("Time Consumption on Calculate labels: %d s" % (int(time.perf_counter() - sttime)))
        sttime = time.perf_counter()

        '''Generate Images'''
        # Cut into blocks and then stack.
        num_cut = 4
        blocks = []
        blocks += cut_blocks_coordinates(tup=(Xs_l[0], Xs_l[1]), num_cut=num_cut)
        blocks += cut_blocks_coordinates(tup=(Xs_r[0], Xs_r[1]), num_cut=num_cut)
        blocks += cut_blocks_coordinates(tup=(L_Xs_0_1, L_Ys_0_1), num_cut=num_cut)
        blocks += cut_blocks_coordinates(tup=(R_Xs_0_1, R_Ys_0_1), num_cut=num_cut)
        blocks += cut_blocks_coordinates(tup=(L_Xs_0_2, L_Ys_0_2), num_cut=num_cut)
        blocks += cut_blocks_coordinates(tup=(R_Xs_0_2, R_Ys_0_2), num_cut=num_cut)
        blocks += cut_blocks_coordinates(tup=(L_Xs_0_3, L_Ys_0_3), num_cut=num_cut)
        blocks += cut_blocks_coordinates(tup=(R_Xs_0_3, R_Ys_0_3), num_cut=num_cut)

        cuted_imgs = generate_imgs(coordinate_list=blocks,
                                   speckle_density=tem_speckle_density,
                                   speckle_size=tem_speckle_size,
                                   randomseeds=randomseed,
                                   num_cut=num_cut,
                                   cache_dir=params["dataset_savepath"],
                                   bkcolor=params["bkcolor"])
        imgs = []
        for n in range(8):
            temp_img = recover_cuts(blocks=cuted_imgs[n*num_cut*num_cut:(n+1)*num_cut*num_cut], num_cut=num_cut, imsize=params["box_size"])
            temp_img = add_range_bk(temp_img.astype("int32"), num_x=np.random.randint(2, 6), num_y=np.random.randint(2, 6), range=40)
            imgs.append(temp_img)

        print("\nTime Consumption on Generate Images: %d s" % (int(time.perf_counter() - sttime)))

        '''Save Displacement Fields'''
        np.save(params["dataset_savepath"] + str(i) + "_LWU_01.npy", Uw_l_01)
        np.save(params["dataset_savepath"] + str(i) + "_LWV_01.npy", Vw_l_01)
        np.save(params["dataset_savepath"] + str(i) + "_LWW_01.npy", Ww_l_01)

        np.save(params["dataset_savepath"] + str(i) + "_LWU_12.npy", Uw_l_12)
        np.save(params["dataset_savepath"] + str(i) + "_LWV_12.npy", Vw_l_12)
        np.save(params["dataset_savepath"] + str(i) + "_LWW_12.npy", Ww_l_12)

        np.save(params["dataset_savepath"] + str(i) + "_LWU_13.npy", Uw_l_13)
        np.save(params["dataset_savepath"] + str(i) + "_LWV_13.npy", Vw_l_13)
        np.save(params["dataset_savepath"] + str(i) + "_LWW_13.npy", Ww_l_13)

        np.save(params["dataset_savepath"] + str(i) + "_LWU_23.npy", Uw_l_23)
        np.save(params["dataset_savepath"] + str(i) + "_LWV_23.npy", Vw_l_23)
        np.save(params["dataset_savepath"] + str(i) + "_LWW_23.npy", Ww_l_23)

        '''save disparity'''
        np.save(params["dataset_savepath"] + str(i) + "_Disparity_1.npy", D_Disparity_X_1)
        np.save(params["dataset_savepath"] + str(i) + "_Disparity_2.npy", D_Disparity_X_2)
        np.save(params["dataset_savepath"] + str(i) + "_Disparity_3.npy", D_Disparity_X_3)

        '''save flow'''
        np.save(params["dataset_savepath"] + str(i) + "_FlowX_01.npy", Flow_01_x)
        np.save(params["dataset_savepath"] + str(i) + "_FlowX_12.npy", Flow_12_x)
        np.save(params["dataset_savepath"] + str(i) + "_FlowX_13.npy", Flow_13_x)
        np.save(params["dataset_savepath"] + str(i) + "_FlowX_23.npy", Flow_23_x)
        np.save(params["dataset_savepath"] + str(i) + "_FlowY_01.npy", Flow_01_y)
        np.save(params["dataset_savepath"] + str(i) + "_FlowY_12.npy", Flow_12_y)
        np.save(params["dataset_savepath"] + str(i) + "_FlowY_13.npy", Flow_13_y)
        np.save(params["dataset_savepath"] + str(i) + "_FlowY_23.npy", Flow_23_y)

        '''Save Images'''
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_L0.tif", imgs[0])
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_R0.tif", imgs[1])
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_L1.tif", imgs[2])
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_R1.tif", imgs[3])
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_L2.tif", imgs[4])
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_R2.tif", imgs[5])
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_L3.tif", imgs[6])
        cv2.imwrite(params["dataset_savepath"] + str(i) + "_R3.tif", imgs[7])

        '''Generate dispplots'''
        plt.figure(figsize=(11, 9))
        plt.subplot(2, 3, 1)
        plt.imshow(Uw_l_12)
        plt.title("Uw_l_12")
        plt.colorbar()
        plt.subplot(2, 3, 2)
        plt.imshow(Vw_l_12)
        plt.title("Vw_l_12")
        plt.colorbar()
        plt.subplot(2, 3, 3)
        plt.imshow(Ww_l_12)
        plt.title("Ww_l_12")
        plt.colorbar()
        plt.subplot(2, 3, 4)
        plt.imshow(D_Disparity_X_1)
        plt.title("DX_1")
        plt.colorbar()
        plt.subplot(2, 3, 5)
        plt.imshow(D_Disparity_X_2)
        plt.title("DX_2")
        plt.colorbar()
        plt.subplot(2, 3, 6)
        plt.imshow(D_Disparity_X_3)
        plt.title("DX_3")
        plt.colorbar()
        plt.draw()
        plt.savefig(params["dataset_savepath"] + str(i) + "disp.png")
        plt.close()
