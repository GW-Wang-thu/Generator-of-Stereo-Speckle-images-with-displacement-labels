import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_seedmap(shape, speckle_density, speckle_size, randomseeds):
    np.random.seed(randomseeds[0])
    SpeckleSeedMap = np.random.rand(shape[0], shape[1]) < speckle_density
    np.random.seed(randomseeds[1])
    SpeckleDirectionMap = np.random.rand(shape[0], shape[1]) * 2 * np.pi
    np.random.seed(randomseeds[2])
    radius = np.abs(np.random.normal(speckle_size, speckle_size / 10, size=shape)) + 0.1
    if speckle_size < 4:
        radius += (np.random.rand(shape[0], shape[1]) < 0.01) * np.random.normal(speckle_size + 1, 0.5)
    else:
        radius -= (np.random.rand(shape[0], shape[1]) < 0.02) * np.random.normal(speckle_size + 1.5, 0.5)
    np.random.seed(randomseeds[3])
    Rx = radius * np.random.normal(1, 0.08, size=shape)  # ratio of the long axis over short axis:1/11, factor 1.2/60 to nearlize the areas.
    Ry = radius ** 2 / Rx

    return SpeckleSeedMap, SpeckleDirectionMap, Rx, Ry


def calculatedisp(Xs, Ys, dispinfo):
    us = np.zeros_like(Xs)
    vs = np.zeros_like(Xs)

    for i in range(len(dispinfo)):
        type = dispinfo[i][0]
        info = dispinfo[i][1:]
        if type == "planer":  # Us = AXs0 + BYs0 + C; Vs = DXs0 + EYs0 + F
            us += info[0] * Xs + info[1] * Ys + info[2]
            vs += info[3] * Xs + info[4] * Ys + info[5]

        elif type == "sin":  # Zs = Asin(BXs + C) * sin(DYs + E); Zcr = (Asin(BXs + C) * sin(DYs + E) + minus) / btm
            us += info[0] * np.sin(info[1] * Xs + info[2]) * np.sin(info[3] * Ys + info[4])
            vs += info[5] * np.sin(info[6] * Xs + info[7]) * np.sin(info[8] * Ys + info[9])
    return us, vs


def calculatedisp_ws(Xs, Ys, dispinfo):
    ws = np.zeros_like(Xs)

    for i in range(len(dispinfo)):
        type = dispinfo[i][0]
        info = dispinfo[i][1:]
        if type == "planer":  # Us = AXs0 + BYs0 + C; Vs = DXs0 + EYs0 + F
            ws += info[0] * Xs + info[1] * Ys + info[2]

        elif type == "sin":  # Zs = Asin(BXs + C) * sin(DYs + E); Zcr = (Asin(BXs + C) * sin(DYs + E) + minus) / btm
            ws += info[0] * np.sin(info[1] * Xs + info[2]) * np.sin(info[3] * Ys + info[4])

    return ws


def cal_successive_disp(previous_l_posi, dispinfo):
    Xs = previous_l_posi[0]
    Ys = previous_l_posi[1]

    us = np.zeros_like(Xs)
    vs = np.zeros_like(Xs)
    ws = np.zeros_like(Xs)

    plane_disp = dispinfo[0]
    offplane_disp = dispinfo[1]

    for i in range(len(plane_disp)):
        type = plane_disp[i][0]
        info = plane_disp[i][1:]
        if type == "planer":  # Us = AXs0 + BYs0 + C; Vs = DXs0 + EYs0 + F
            us += info[0] * Xs + info[1] * Ys + info[2]
            vs += info[3] * Xs + info[4] * Ys + info[5]

        elif type == "sin":  # Zs = Asin(BXs + C) * sin(DYs + E); Zcr = (Asin(BXs + C) * sin(DYs + E) + minus) / btm
            us += info[0] * np.sin(info[1] * Xs + info[2]) * np.sin(info[3] * Ys + info[4])
            vs += info[5] * np.sin(info[6] * Xs + info[7]) * np.sin(info[8] * Ys + info[9])

    for i in range(len(offplane_disp)):
        type = offplane_disp[i][0]
        info = offplane_disp[i][1:]
        if type == "planer":  # Us = AXs0 + BYs0 + C; Vs = DXs0 + EYs0 + F
            ws += info[0] * Xs + info[1] * Ys + info[2]

        elif type == "sin":  # Zs = Asin(BXs + C) * sin(DYs + E); Zcr = (Asin(BXs + C) * sin(DYs + E) + minus) / btm
            ws += info[0] * np.sin(info[1] * Xs + info[2]) * np.sin(info[3] * Ys + info[4])

    return us, vs, ws


def disp_minus(list1, list2):

    minus_list = []
    for i in range(len(list1)):
        minus_list.append(list1[i] - list2[i])
    return minus_list


def array2img(array, background, noise, color='black'):

    Gaussian_map = np.random.normal(0.0, 1, size=array.shape) * noise / 256
    if color == "black":
        img = array * 0.6 * (array > 0)
    else:
        img = (array-1) * 0.5 * (array-1 < 0) + 1.0

    if color == "black":
        img = (img < background / 255) * background / 255 + (img >= background / 255) * img
        img += Gaussian_map * 0.2
        img = (((img - 1) * (img < 1) + 1) * 255).astype("uint8")
    else:
        img = (img > background / 255) * background / 255 + (img <= background / 255) * img
        img += Gaussian_map * 0.2
        img = ((((img-0.2) * (img > 0.2)) + 0.2) * 255).astype("uint8")
    return img


def cut_blocks(img, num_cut):
    size = img.shape
    step_dim0 = size[0] // num_cut
    step_dim1 = size[1] // num_cut
    img_blocks = []
    for i in range(num_cut):
        for j in range(num_cut):
            # temp_block = img[i*step_dim0:(i+1)*step_dim0, j*step_dim1:(j+1)*step_dim1]
            if i != num_cut-1 and j != num_cut-1:
                temp_block = img[i * step_dim0:(i + 1) * step_dim0+1, j * step_dim1:(j + 1) * step_dim1+1]
            elif i != num_cut-1 and j == num_cut-1:
                temp_block = img[i * step_dim0:(i + 1) * step_dim0+1, j * step_dim1:(j + 1) * step_dim1]
            elif i == num_cut-1 and j != num_cut-1:
                temp_block = img[i * step_dim0:(i + 1) * step_dim0, j * step_dim1:(j + 1) * step_dim1+1]
            elif i == num_cut-1 and j == num_cut-1:
                temp_block = img[i * step_dim0:(i + 1) * step_dim0, j * step_dim1:(j + 1) * step_dim1]
            img_blocks.append(temp_block)
    return img_blocks


def cut_blocks_coordinates(tup, num_cut, padding=3):
    blocks = []
    X_blocks = cut_blocks(img=tup[0], num_cut=num_cut)
    Y_blocks = cut_blocks(img=tup[1], num_cut=num_cut)
    for m in range(len(X_blocks)):
        blocks.append((X_blocks[m], Y_blocks[m]))
    return blocks


def recover_cuts(blocks, num_cut, imsize):
    blocksize = (imsize[0]//num_cut, imsize[1]//num_cut)
    img_size = (blocksize[0] * num_cut, blocksize[1] * num_cut)
    img = np.zeros(img_size, dtype=blocks[0].dtype)
    for i in range(num_cut):
        for j in range(num_cut):
            # img[i * blocksize[0]:(i + 1) * blocksize[0], j * blocksize[1]:(j + 1) * blocksize[1]] = blocks[i * num_cut + j]
            if i != num_cut-1 and j != num_cut-1:
                img[i * blocksize[0]:(i + 1) * blocksize[0], j * blocksize[1]:(j + 1) * blocksize[1]] = blocks[i * num_cut + j][:-1, :-1]
            elif i != num_cut-1 and j == num_cut-1:
                img[i * blocksize[0]:(i + 1) * blocksize[0], j * blocksize[1]:(j + 1) * blocksize[1]] = blocks[i * num_cut + j][:-1, :]
            elif i == num_cut-1 and j != num_cut-1:
                img[i * blocksize[0]:(i + 1) * blocksize[0], j * blocksize[1]:(j + 1) * blocksize[1]] = blocks[i * num_cut + j][:, :-1]
            elif i == num_cut-1 and j == num_cut-1:
                img[i * blocksize[0]:(i + 1) * blocksize[0], j * blocksize[1]:(j + 1) * blocksize[1]] = blocks[i * num_cut + j]
    return img


def add_range_bk(img, num_x, num_y, range=20):
    bk = cv2.resize(np.random.randint(0, range, (num_x, num_y), dtype="uint8"), dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    img = img + bk.astype("int32") - range // 2
    img = ((img - 255) < 0) * img + 255
    img = img * (img > 0)
    return img.astype("uint8")


if __name__ == '__main__':
    pass