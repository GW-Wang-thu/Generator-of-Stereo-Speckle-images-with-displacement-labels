import numpy as np
import utils
import matplotlib.pyplot as plt


class disp_solver_zc:
    def __init__(self, type_amp, beta_sw, Tsw, A, b):
        self.dispinfo = type_amp
        self.betasw = beta_sw
        self.Tsw = Tsw
        self.A = A
        self.b = b

    def NewtonIteration(self, xk, tolerance, max_iter):
        i = 0
        tor = tolerance + 1
        xk = np.array([xk])
        torrec = []
        while((i < max_iter) and (tor > tolerance)):
            Fxk = self.__F_x(xk)
            dFxk = self.__dF_x(xk)
            xk1 = xk - Fxk / dFxk
            tor = np.linalg.norm(xk1 - xk)
            xk = xk1
            i += 1
            torrec.append(tor)

        # plt.plot(torrec)
        # plt.show()
        return xk

    # def __F_x(self, xk):
    #     Fxk = np.zeros_like(xk)
    #     Xs = np.dot(np.array([self.betasw[0, :]]), self.A) * xk + np.dot(np.array([self.betasw[0, :]]), self.b) + self.Tsw[0]
    #     Ys = np.dot(np.array([self.betasw[1, :]]), self.A) * xk + np.dot(np.array([self.betasw[1, :]]), self.b) + self.Tsw[1]
    #     minus = - np.dot(np.array([self.betasw[2, :]]), self.b) - self.Tsw[2]
    #     btm = np.dot(np.array([self.betasw[2, :]]), self.A)
    #     for i in range(len(self.dispinfo)):
    #         disptype = self.dispinfo[i][0]
    #         dispinfo = self.dispinfo[i][1:]
    #         if disptype == "planer":    #Zs = AXs + BYs + C; Zcr = (AXs + BYs + C - beta3ibi - Tsw3) / btm
    #             Fxk += (dispinfo[0] * Xs + dispinfo[1] * Ys + dispinfo[2] + minus) / btm - xk
    #         elif disptype == "sin":     #Zs = Asin(BXs + C) * sin(DYs + E); Zcr = (Asin(BXs + C) * sin(DYs + E) + minus) / btm
    #             Fxk += (dispinfo[0] * np.sin(dispinfo[1] * Xs + dispinfo[2]) * np.sin(dispinfo[3] * Ys + dispinfo[4]) + minus) / btm - xk
    #     return Fxk
    def __F_x(self, xk):
        Fxk = np.zeros_like(xk)
        Xs = np.dot(np.array([self.betasw[0, :]]), self.A) * xk + np.dot(np.array([self.betasw[0, :]]), self.b) + self.Tsw[0]
        Ys = np.dot(np.array([self.betasw[1, :]]), self.A) * xk + np.dot(np.array([self.betasw[1, :]]), self.b) + self.Tsw[1]
        minus = - np.dot(np.array([self.betasw[2, :]]), self.b) - self.Tsw[2]
        btm = np.dot(np.array([self.betasw[2, :]]), self.A)
        for i in range(len(self.dispinfo)):
            disptype = self.dispinfo[i][0]
            dispinfo = self.dispinfo[i][1:]
            if disptype == "planer":    #Zs = AXs + BYs + C; Zcr = (AXs + BYs + C - beta3ibi - Tsw3) / btm
                Fxk += (dispinfo[0] * Xs + dispinfo[1] * Ys + dispinfo[2])
            elif disptype == "sin":     #Zs = Asin(BXs + C) * sin(DYs + E); Zcr = (Asin(BXs + C) * sin(DYs + E) + minus) / btm
                Fxk += (dispinfo[0] * np.sin(dispinfo[1] * Xs + dispinfo[2]) * np.sin(dispinfo[3] * Ys + dispinfo[4]))
            # Other displacement can be added
        Fxk += minus
        Fxk = Fxk / btm - xk

        return Fxk

    def __dF_x(self, xk):
        dFxk = np.zeros_like(xk)
        Xs = np.dot(np.array([self.betasw[0, :]]), self.A) * xk + np.dot(np.array([self.betasw[0, :]]), self.b) + self.Tsw[0]
        Ys = np.dot(np.array([self.betasw[1, :]]), self.A) * xk + np.dot(np.array([self.betasw[1, :]]), self.b) + self.Tsw[1]
        btm = np.dot(np.array([self.betasw[2, :]]), self.A)
        for i in range(len(self.dispinfo)):
            disptype = self.dispinfo[i][0]
            dispinfo = self.dispinfo[i][1:]
            if disptype == "planer":
                dFxk += (dispinfo[0] * np.dot(np.array([self.betasw[0, :]]), self.A) + dispinfo[1] * np.dot(np.array([self.betasw[1, :]]), self.A))
            elif disptype == "sin":
                dFxk += (dispinfo[0] * dispinfo[3] * (np.dot(np.array([self.betasw[1, :]]), self.A)) * np.sin(dispinfo[1] * Xs + dispinfo[2]) * np.cos(dispinfo[3] * Ys + dispinfo[4]) + \
                        dispinfo[0] * dispinfo[0] * (np.dot(np.array([self.betasw[0, :]]), self.A)) * np.cos(dispinfo[1] * Xs + dispinfo[2]) * np.sin(dispinfo[3] * Ys + dispinfo[4]))
            # Other displacement can be added.
        dFxk = dFxk/btm - np.ones_like(xk)
        return dFxk


class disp_solver_xys:
    def __init__(self, type_amp):
        self.dispinfo = type_amp

    def NewtonIteration(self, xs0, ys0, xs, ys, tolerance, max_iter):
        i = 0
        tor = tolerance + 1
        init_vect_x = np.array([xs0])
        init_vect_y = np.array([ys0])
        xs = np.array([xs])
        ys = np.array([ys])
        torrec = []

        # 迭代求解
        vect_k_x = init_vect_x
        vect_k_y = init_vect_y
        while((i < max_iter) and (tor > tolerance)):
            Fxk, Fyk = self.__F_x(vect_k_x, vect_k_y, xs, ys)
            idFxk, idFyk = self.__idF_x(vect_k_x, vect_k_y, xs, ys)
            vect_k1_x = vect_k_x - np.array([idFxk[0] * Fxk[0] + idFxk[1] * Fyk[0]])
            vect_k1_y = vect_k_y - np.array([idFyk[0] * Fyk[0] + idFyk[1] * Fyk[0]])
            tor = np.linalg.norm(vect_k1_x - vect_k_x) + np.linalg.norm(vect_k1_y - vect_k_y)
            vect_k_x = vect_k1_x
            vect_k_y = vect_k1_y
            i += 1
            torrec.append(tor)

        # plt.plot(torrec)
        # plt.show()
        return vect_k_x, vect_k_y

    def __F_x(self, vect_k_x, vect_k_y, xs, ys):

        Fxk = np.zeros_like(vect_k_x) + xs
        Fyk = np.zeros_like(vect_k_y) + ys

        for i in range(len(self.dispinfo)):
            disptype = self.dispinfo[i][0]
            dispinfo = self.dispinfo[i][1:]
            if disptype == "planer":    #Us = AXs0 + BYs0 + C; Vs = DXs0 + EYs0 + F
                Fxk += - dispinfo[0] * vect_k_x - dispinfo[1] * vect_k_y - dispinfo[2]
                Fyk += - dispinfo[3] * vect_k_x - dispinfo[4] * vect_k_y - dispinfo[5]

            elif disptype == "sin":     #Zs = Asin(BXs + C) * sin(DYs + E); Zcr = (Asin(BXs + C) * sin(DYs + E) + minus) / btm
                Fxk += - dispinfo[0] * np.sin(dispinfo[1]*vect_k_x + dispinfo[2]) * np.sin(dispinfo[3]*vect_k_y + dispinfo[4])
                Fyk += - dispinfo[5] * np.sin(dispinfo[6]*vect_k_x + dispinfo[7]) * np.sin(dispinfo[8]*vect_k_y + dispinfo[9])

        Fxk += - vect_k_x
        Fyk += - vect_k_y

        return Fxk, Fyk

    def __idF_x(self, vect_k_x, vect_k_y, xs, ys):

        dFxk = np.zeros(shape=(2, vect_k_x.shape[1]))
        dFyk = np.zeros(shape=(2, vect_k_x.shape[1]))

        for i in range(len(self.dispinfo)):
            disptype = self.dispinfo[i][0]
            dispinfo = self.dispinfo[i][1:]
            if disptype == "planer":
                dFxk[0, :] += - (dispinfo[0]+1)
                dFxk[1, :] += - dispinfo[1]
                dFyk[0, :] += - dispinfo[3]
                dFyk[1, :] += - (dispinfo[4]+1)
            elif disptype == "sin":
                dFxk[0, :] -= dispinfo[0] * dispinfo[1] * np.cos(dispinfo[1]*vect_k_x[0] + dispinfo[2]) * np.sin(dispinfo[3]*vect_k_y[0] + dispinfo[4]) + 1
                dFxk[1, :] -= dispinfo[0] * dispinfo[3] * np.sin(dispinfo[1]*vect_k_x[0] + dispinfo[2]) * np.cos(dispinfo[3]*vect_k_y[0] + dispinfo[4])
                dFyk[0, :] -= dispinfo[5] * dispinfo[6] * np.cos(dispinfo[6]*vect_k_x[0] + dispinfo[7]) * np.sin(dispinfo[8]*vect_k_y[0] + dispinfo[9])
                dFyk[1, :] -= dispinfo[5] * dispinfo[8] * np.sin(dispinfo[6]*vect_k_x[0] + dispinfo[7]) * np.cos(dispinfo[8]*vect_k_y[0] + dispinfo[9]) + 1
        # solve inverse array
        idFxk = np.zeros_like(dFxk)
        idFyk = np.zeros_like(dFyk)
        btm = dFxk[0, :] * dFyk[1, :] - dFxk[1, :] * dFyk[0, :]
        idFxk[0, :] = dFyk[1, :]
        idFxk[1, :] = - dFxk[1, :]
        idFyk[0, :] = - dFyk[0, :]
        idFyk[1, :] = dFxk[0, :]
        idFxk /= btm
        idFyk /= btm

        return idFxk, idFyk
