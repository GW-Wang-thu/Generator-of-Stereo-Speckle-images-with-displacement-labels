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


def array2img(array, background, noise):
    Gaussian_map = np.random.normal(0.0, 1, size=array.shape) * noise / 256
    array += Gaussian_map

    img = array * 0.6 * (array > 0)
    img = (img < background / 255) * background / 255 + (img >= background / 255) * img
    img = (((img - 1) * (img < 1) + 1) * 255).astype("uint8")

    return img

def img_flip(imgs):
    flipd_img = []
    for i in range(len(imgs)):
        temp_img = cv2.flip(imgs[i], 0)
        flipd_img.append(cv2.flip(temp_img, 1))
    return flipd_img


if __name__ == '__main__':
    # line = [3, 2, [(["planer", 0.01, 0.02, 0.1, 0.01, 0.02, 0.05], ["planer", -0.03, 0.002, 0.1, 0.01, 0.002, 0.05]), (["planer", 0.01, 0.02, 0.05], ["sin", 0.1, 3, 0.1, 3, 0.1])], 100]
    # f = open("./test.txt", "w")
    # for i in range(3):
    #     f.write(str(line)+"\n")
    # f.close()
    #
    # f1 = open("./test.txt", "r")
    # for line in f1:
    #     print(line)
    #     line = line.split()
    U = np.loadtxt("..\data/0_LWU.csv")
    V = np.loadtxt("..\data/0_LWV.csv")
    W = np.loadtxt("..\data/0_LWW.csv")
    DX = np.loadtxt("..\data/0_Disparity_DX.csv")
    DY = np.loadtxt("..\data/0_Disparity_DY.csv")

    beta_sw = np.linalg.inv(np.array([[0, 1, 0], [0.8660254, 0, 0.5], [0.5, 0, -0.8660254]]))

    beta_lr = np.array([[0.5, 0, 0.8660254], [0, 1, 0], [-0.8660254, 0, 0.5]])

    Uw = beta_sw[0][0] * U + beta_sw[0][1] * V + beta_sw[0][2] * W
    Vw = beta_sw[1][0] * U + beta_sw[1][1] * V + beta_sw[1][2] * W
    Ww = beta_sw[2][0] * U + beta_sw[2][1] * V + beta_sw[2][2] * W

    Ur = beta_lr[0][0] * Uw + beta_lr[0][1] * Vw + beta_lr[0][2] * Ww
    Vr = beta_lr[1][0] * Uw + beta_lr[1][1] * Vw + beta_lr[1][2] * Ww
    Wr = beta_lr[2][0] * Uw + beta_lr[2][1] * Vw + beta_lr[2][2] * Ww

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.imshow(U)
    plt.title("U")
    plt.colorbar()
    plt.subplot(3, 2, 2)
    plt.imshow(V)
    plt.title("V")
    plt.colorbar()
    plt.subplot(3, 2, 3)
    plt.imshow(W)
    plt.title("W")
    plt.colorbar()
    plt.subplot(3, 2, 5)
    plt.imshow(DX)
    plt.title("DX")
    plt.colorbar()
    plt.subplot(3, 2, 6)
    plt.imshow(DY)
    plt.title("DY")
    plt.colorbar()


    plt.show()
    plt.savefig("disp.png")
    plt.close()



