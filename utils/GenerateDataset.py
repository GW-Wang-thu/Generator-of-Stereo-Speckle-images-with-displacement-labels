import numpy as np
import cv2
import os

'''Generate dataset to train StrainNet-3D'''
def dataset_generation(dir, num_samples, traindir, testdir, H, coord_l, coord_r, train_percent=0.85, stid=0):

    trainid_list = []
    testid_list = []

    zero_disp = coord_r - coord_l
    for i in range(num_samples):
        k = i + stid
        if os.path.exists(traindir + str(k) + "_uvws.npy"):
            trainid_list.append(k)
            continue

        if os.path.exists(testdir + str(k) + "_uvws.npy"):
            testid_list.append(k)
            continue

        if not os.path.exists(dir + str(i) + "_L0.tif"):
            continue

        print("Converting %d ..."%(i))
        if np.random.rand() < train_percent:
            temp_imgs_dir = traindir + str(k) + "_imgs.npy"
            temp_flow_dir = traindir + str(k) + "_flow.npy"
            temp_disp_dir = traindir + str(k) + "_disp.npy"
            temp__uvw_dir = traindir + str(k) + "_uvws.npy"
            trainid_list.append(k)
        else:
            temp_imgs_dir = testdir + str(k) + "_imgs.npy"
            temp_flow_dir = testdir + str(k) + "_flow.npy"
            temp_disp_dir = testdir + str(k) + "_disp.npy"
            temp__uvw_dir = testdir + str(k) + "_uvws.npy"
            testid_list.append(k)

        '''Load Images'''
        img_L0 = cv2.imread(dir + str(i) + "_L0.tif", cv2.IMREAD_GRAYSCALE)
        img_L1 = cv2.imread(dir + str(i) + "_L1.tif", cv2.IMREAD_GRAYSCALE)
        img_L2 = cv2.imread(dir + str(i) + "_L2.tif", cv2.IMREAD_GRAYSCALE)
        img_L3 = cv2.imread(dir + str(i) + "_L3.tif", cv2.IMREAD_GRAYSCALE)

        '''Calculate Right Transposed image(right to left)'''
        img_R0 = pad2black(cv2.imread(dir + str(i) + "_R0.tif", cv2.IMREAD_GRAYSCALE))
        img_R1 = pad2black(cv2.imread(dir + str(i) + "_R1.tif", cv2.IMREAD_GRAYSCALE))
        img_R2 = pad2black(cv2.imread(dir + str(i) + "_R2.tif", cv2.IMREAD_GRAYSCALE))
        img_R3 = pad2black(cv2.imread(dir + str(i) + "_R3.tif", cv2.IMREAD_GRAYSCALE))

        img_R0T = cv2.warpPerspective(img_R0, H, (img_R0.shape[1], img_R0.shape[0]), flags=cv2.INTER_CUBIC)[500:500+2048, 1000:1000+2048]
        img_R1T = cv2.warpPerspective(img_R1, H, (img_R1.shape[1], img_R1.shape[0]), flags=cv2.INTER_CUBIC)[500:500+2048, 1000:1000+2048]
        img_R2T = cv2.warpPerspective(img_R2, H, (img_R2.shape[1], img_R2.shape[0]), flags=cv2.INTER_CUBIC)[500:500+2048, 1000:1000+2048]
        img_R3T = cv2.warpPerspective(img_R3, H, (img_R3.shape[1], img_R3.shape[0]), flags=cv2.INTER_CUBIC)[500:500+2048, 1000:1000+2048]

        imgs = np.array([img_L0, img_L1, img_L2, img_L3,
                         img_R0T, img_R1T, img_R2T, img_R3T], dtype="uint8")
        np.save(temp_imgs_dir, imgs)


        if os.path.exists(dir + str(i) + "_FlowX_01.csv"):
            # For old version saved in csv format
            # flow_01_X = np.loadtxt(dir + str(i) + "_FlowX_01.csv")
            # flow_12_X = np.loadtxt(dir + str(i) + "_FlowX_12.csv")
            # flow_13_X = np.loadtxt(dir + str(i) + "_FlowX_13.csv")
            # flow_23_X = np.loadtxt(dir + str(i) + "_FlowX_23.csv")
            # flow_01_Y = np.loadtxt(dir + str(i) + "_FlowY_01.csv")
            # flow_12_Y = np.loadtxt(dir + str(i) + "_FlowY_12.csv")
            # flow_13_Y = np.loadtxt(dir + str(i) + "_FlowY_13.csv")
            # flow_23_Y = np.loadtxt(dir + str(i) + "_FlowY_23.csv")
            # # Disparity
            # disp_1 = np.loadtxt(dir + str(i) + "_Disparity_1.csv")
            # disp_2 = np.loadtxt(dir + str(i) + "_Disparity_2.csv")
            # disp_3 = np.loadtxt(dir + str(i) + "_Disparity_3.csv")
            pass

        else:
            # Optical Flow
            flow_01_X = np.load(dir + str(i) + "_FlowX_01.npy")
            flow_12_X = np.load(dir + str(i) + "_FlowX_12.npy")
            flow_13_X = np.load(dir + str(i) + "_FlowX_13.npy")
            flow_23_X = np.load(dir + str(i) + "_FlowX_23.npy")
            flow_01_Y = np.load(dir + str(i) + "_FlowY_01.npy")
            flow_12_Y = np.load(dir + str(i) + "_FlowY_12.npy")
            flow_13_Y = np.load(dir + str(i) + "_FlowY_13.npy")
            flow_23_Y = np.load(dir + str(i) + "_FlowY_23.npy")
            # Disparity
            disp_1 = np.load(dir + str(i) + "_Disparity_1.npy")
            disp_2 = np.load(dir + str(i) + "_Disparity_2.npy")
            disp_3 = np.load(dir + str(i) + "_Disparity_3.npy")

        flows = np.array([flow_01_X, flow_01_Y, flow_12_X, flow_12_Y,
                          flow_13_X, flow_13_Y, flow_23_X, flow_23_Y], dtype="float32")
        np.save(temp_flow_dir, flows)

        disps = np.array([disp_1-zero_disp-1000.0, disp_2-zero_disp-1000.0, disp_3-zero_disp-1000.0], dtype="float32")
        np.save(temp_disp_dir, disps)

        if os.path.exists(dir + str(i) + "_FlowX_01.csv"):
            # UVW
            uvw_01_U = np.loadtxt(dir + str(i) + "_LWU_01.csv")
            uvw_12_U = np.loadtxt(dir + str(i) + "_LWU_12.csv")
            uvw_13_U = np.loadtxt(dir + str(i) + "_LWU_13.csv")
            uvw_23_U = np.loadtxt(dir + str(i) + "_LWU_23.csv")

            uvw_01_V = np.loadtxt(dir + str(i) + "_LWV_01.csv")
            uvw_12_V = np.loadtxt(dir + str(i) + "_LWV_12.csv")
            uvw_13_V = np.loadtxt(dir + str(i) + "_LWV_13.csv")
            uvw_23_V = np.loadtxt(dir + str(i) + "_LWV_23.csv")

            uvw_01_W = np.loadtxt(dir + str(i) + "_LWW_01.csv")
            uvw_12_W = np.loadtxt(dir + str(i) + "_LWW_12.csv")
            uvw_13_W = np.loadtxt(dir + str(i) + "_LWW_13.csv")
            uvw_23_W = np.loadtxt(dir + str(i) + "_LWW_23.csv")
        else:
            # UVW
            uvw_01_U = np.load(dir + str(i) + "_LWU_01.npy")
            uvw_12_U = np.load(dir + str(i) + "_LWU_12.npy")
            uvw_13_U = np.load(dir + str(i) + "_LWU_13.npy")
            uvw_23_U = np.load(dir + str(i) + "_LWU_23.npy")

            uvw_01_V = np.load(dir + str(i) + "_LWV_01.npy")
            uvw_12_V = np.load(dir + str(i) + "_LWV_12.npy")
            uvw_13_V = np.load(dir + str(i) + "_LWV_13.npy")
            uvw_23_V = np.load(dir + str(i) + "_LWV_23.npy")

            uvw_01_W = np.load(dir + str(i) + "_LWW_01.npy")
            uvw_12_W = np.load(dir + str(i) + "_LWW_12.npy")
            uvw_13_W = np.load(dir + str(i) + "_LWW_13.npy")
            uvw_23_W = np.load(dir + str(i) + "_LWW_23.npy")

        uvws = np.array([ uvw_01_U, uvw_01_V, uvw_01_W,
                          uvw_12_U, uvw_12_V, uvw_12_W,
                          uvw_13_U, uvw_13_V, uvw_13_W,
                          uvw_23_U, uvw_23_V, uvw_23_W], dtype="float32")
        np.save(temp__uvw_dir, uvws)

    # important: idlist contains the dataset idx info
    np.savetxt(traindir + "idlist.csv", np.array(trainid_list), delimiter=",")
    np.savetxt(testdir + "idlist.csv", np.array(testid_list), delimiter=",")


'''Padding generated image block to its initial position in a void image'''
def pad2black(img, blackshape=[3000, 4096], stp=[1000, 1500]):
    bk = np.zeros(shape=blackshape, dtype="uint8")
    bk[stp[0]:stp[0]+img.shape[0], stp[1]:stp[1]+img.shape[1]] = img
    return bk


def proj_get_src(dst_point, H):
    x = dst_point[0]
    y = dst_point[1]
    x_src = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / (H[2, 0] * x + H[2, 1] * y + H[2, 2])
    y_src = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / (H[2, 0] * x + H[2, 1] * y + H[2, 2])
    return (x_src, y_src)


'''Calculate Transpose matrix from right to left using padded images through openCV'''
def calculate_Transpose_matrix(lpath, rpath, stp=[500, 1000]):
    limg = cv2.imread(lpath, cv2.IMREAD_GRAYSCALE)
    rimg = cv2.imread(rpath, cv2.IMREAD_GRAYSCALE)
    lbk = pad2black(limg)
    rbk = pad2black(rimg)
    cv2.imwrite(lpath[:-4]+"_pad.bmp", lbk)
    cv2.imwrite(rpath[:-4]+"_pad.bmp", rbk)
    akaze = cv2.AKAZE_create()
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = akaze.detectAndCompute(rbk, None)
    kp2, des2 = akaze.detectAndCompute(lbk, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good_matches.append([m])

    src_automatic_points = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    den_automatic_points = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # 调用库函数计算特征矩阵
    H, status = cv2.findHomography(src_automatic_points, den_automatic_points, cv2.RANSAC, 5.0)
    np.savetxt(lpath[:-5]+"Transpose_matrix.txt", H, delimiter=',')

    warped_automatic_image = cv2.warpPerspective(rbk, H, (lbk.shape[1], lbk.shape[0]))[500:500+2048, 1000:1000+2048]
    cv2.imwrite(rpath[:-4]+"_T.bmp", warped_automatic_image)

    invH = np.linalg.inv(H)
    shape_x = limg.shape[0]
    shape_y = limg.shape[1]
    array_left_x = np.zeros_like(limg, dtype='float32')
    array_left_y = np.zeros_like(limg, dtype='float32')
    array_right_x = np.zeros_like(limg, dtype='float32')
    array_right_y = np.zeros_like(limg, dtype='float32')
    for i in range(shape_y):
        for j in range(shape_x):
            array_left_x[i, j] = stp[1] + j
            array_left_y[i, j] = stp[0] + i
            temp_right_point = (proj_get_src((stp[1] + j, stp[0] + i), invH))
            array_right_x[i, j] = temp_right_point[0]
            array_right_y[i, j] = temp_right_point[1]
    # np.savetxt(lpath[:-5] + "PixelCoord_L_X.csv", array_left_x)
    # np.savetxt(lpath[:-5] + "PixelCoord_L_Y.csv", array_left_y)
    # np.savetxt(lpath[:-5] + "PixelCoord_R_X.csv", array_right_x)
    # np.savetxt(lpath[:-5] + "PixelCoord_R_Y.csv", array_right_y)   #矩阵意义上的位置

    return H, array_left_x, array_right_x


'''Load npy to bmp image'''
def process_img(npy, outdir):
    '''Get imgs from *Imgs.npy'''
    all_imgs = np.load(npy)
    img_l1 = all_imgs[2, :, :]
    cv2.imwrite(outdir+"2_L1T.bmp", img_l1)
    img_l2 = all_imgs[3, :, :]
    cv2.imwrite(outdir+"2_L2T.bmp", img_l2)
    img_r1 = all_imgs[6, :, :]
    cv2.imwrite(outdir+"2_R1T.bmp", img_r1)
    img_r2 = all_imgs[7, :, :]
    cv2.imwrite(outdir+"2_R2T.bmp", img_r2)


if __name__ == '__main__':
    Transpose_matrix, lcoord, rcoord = calculate_Transpose_matrix(r'I:\DLDIC_3D_Dataset\Paper_Exp\parameters\0.tif',
                                                  r'I:\DLDIC_3D_Dataset\Paper_Exp\parameters\1.tif')    # Calculate transpose matrix
    #
    dataset_generation(dir="I:\\DLDIC_3D_Dataset\\Paper_Exp\\Dataset/",
                       num_samples=100,
                       traindir="I:\\DLDIC_3D_Dataset\\Paper_Exp\\train/",
                       testdir="I:\\DLDIC_3D_Dataset\\Paper_Exp/valid/",
                       train_percent=0.82,
                       H=Transpose_matrix,
                       coord_l = lcoord,
                       coord_r = rcoord,
                       stid=0)
    # process_img(npy=r'F:\case\case8\imgs\10_imgs.npy', outdir=r'F:\case\case8\imgs\\')

    # img_P = pad2black(cv2.imread(r'F:\case\case8\imgs\10_R2.tif', cv2.IMREAD_GRAYSCALE))
    # cv2.imwrite(r'F:\case\case8\imgs\10_R1_P.bmp', img_P)
    #
    # img_P = pad2black(cv2.imread(r'F:\case\case8\imgs\10_R3.tif', cv2.IMREAD_GRAYSCALE))
    # cv2.imwrite(r'F:\case\case8\imgs\10_R2_P.bmp', img_P)
    #
    # img_P = pad2black(cv2.imread(r'F:\case\case8\imgs\10_L2.tif', cv2.IMREAD_GRAYSCALE))
    # cv2.imwrite(r'F:\case\case8\imgs\10_L1_P.bmp', img_P)
    #
    # img_P = pad2black(cv2.imread(r'F:\case\case8\imgs\10_L3.tif', cv2.IMREAD_GRAYSCALE))
    # cv2.imwrite(r'F:\case\case8\imgs\10_L2_P.bmp', img_P)