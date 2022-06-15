import numpy as np
import matplotlib.pyplot as plt
# from numba_cuda_funcs.cuda_funcs import Fx_z
import torch


class disp_solver_zc:

    def __init__(self, type_amp, beta_sw, Tsw, A, b):
        self.dispinfo = type_amp
        self.betasw = torch.from_numpy(beta_sw.astype("float64")).cuda()
        self.Tsw = torch.from_numpy(Tsw.astype("float64")).cuda()
        self.A = torch.from_numpy(A.astype("float64")).cuda()
        self.b = torch.from_numpy(b.astype("float64")).cuda()

    def NewtonIteration(self, xk, tolerance, max_iter):
        i = 0
        tor = tolerance + 1
        xk = torch.from_numpy(np.array([xk], dtype="float64")).cuda()
        torrec = []
        while((i < max_iter) and (tor > tolerance)):
            Fxk = self.__F_x(xk)
            dFxk = self.__dF_x(xk)
            xk1 = xk - torch.div(Fxk, dFxk)
            tor = np.linalg.norm(xk1.cpu().numpy() - xk.cpu().numpy())
            xk = xk1
            i += 1
            torrec.append(tor)
        return xk.cpu().numpy()

    def __F_x(self, xk):
        Fxk = torch.zeros_like(xk)
        Xs = torch.mm(self.betasw[0, :].unsqueeze(0), self.A) * xk + torch.mm(self.betasw[0, :].unsqueeze(0), self.b) + self.Tsw[0]
        Ys = torch.mm(self.betasw[1, :].unsqueeze(0), self.A) * xk + torch.mm(self.betasw[1, :].unsqueeze(0), self.b) + self.Tsw[1]
        minus = - torch.mm(self.betasw[2, :].unsqueeze(0), self.b) - self.Tsw[2]
        btm = torch.mm(self.betasw[2, :].unsqueeze(0), self.A)
        for i in range(len(self.dispinfo)):
            disptype = self.dispinfo[i][0]
            dispinfo = self.dispinfo[i][1:]
            if disptype == "planer":    #Zs = AXs + BYs + C; Zcr = (AXs + BYs + C - beta3ibi - Tsw3) / btm
                Fxk += (dispinfo[0] * Xs + dispinfo[1] * Ys + dispinfo[2])
            elif disptype == "sin":     #Zs = Asin(BXs + C) * sin(DYs + E); Zcr = (Asin(BXs + C) * sin(DYs + E) + minus) / btm
                # Fxk = Fx_z(dispinfo[0], dispinfo[1], dispinfo[2], dispinfo[3], dispinfo[4], Xs, Ys, Fxk)    # 使用GPU加速，等效于下式
                Fxk += (dispinfo[0] * torch.sin(dispinfo[1] * Xs + dispinfo[2]) * torch.sin(dispinfo[3] * Ys + dispinfo[4]))
        Fxk1 = Fxk + minus
        Fxk = torch.div(Fxk1, btm) - xk

        return Fxk

    def __dF_x(self, xk):
        dFxk = torch.zeros_like(xk).cuda()
        Xs = torch.mm(self.betasw[0, :].unsqueeze(0), self.A) * xk + torch.mm(self.betasw[0, :].unsqueeze(0), self.b) + self.Tsw[0]
        Ys = torch.mm(self.betasw[1, :].unsqueeze(0), self.A) * xk + torch.mm(self.betasw[1, :].unsqueeze(0), self.b) + self.Tsw[1]
        btm = torch.mm(self.betasw[2, :].unsqueeze(0), self.A)
        for i in range(len(self.dispinfo)):
            disptype = self.dispinfo[i][0]
            dispinfo = self.dispinfo[i][1:]
            if disptype == "planer":
                dFxk += (dispinfo[0] * torch.mm(self.betasw[0, :].unsqueeze(0), self.A) + dispinfo[1] * torch.mm(self.betasw[1, :].unsqueeze(0), self.A))
            elif disptype == "sin":
                dFxk += (dispinfo[0] * dispinfo[3] * (torch.mm(self.betasw[1, :].unsqueeze(0), self.A)) * torch.sin(dispinfo[1] * Xs + dispinfo[2]) * torch.cos(dispinfo[3] * Ys + dispinfo[4]) + \
                        dispinfo[0] * dispinfo[0] * (torch.mm(self.betasw[0, :].unsqueeze(0), self.A)) * torch.cos(dispinfo[1] * Xs + dispinfo[2]) * torch.sin(dispinfo[3] * Ys + dispinfo[4]))
        dFxk = torch.div(dFxk, btm) - torch.ones_like(xk).cuda()
        return dFxk


class disp_solver_xys:

    def __init__(self, type_amp):
        self.dispinfo = type_amp

    def NewtonIteration(self, xs0, ys0, xs, ys, tolerance, max_iter):
        i = 0
        tor = tolerance + 1
        init_vect_x = torch.from_numpy(np.array([xs0], dtype="float64")).cuda()
        init_vect_y = torch.from_numpy(np.array([ys0], dtype="float64")).cuda()
        xs = torch.from_numpy(np.array([xs], dtype="float64")).cuda()
        ys = torch.from_numpy(np.array([ys], dtype="float64")).cuda()
        torrec = []

        # 迭代求解
        vect_k_x = init_vect_x
        vect_k_y = init_vect_y
        while((i < max_iter) and (tor > tolerance)):
            Fxk, Fyk = self.__F_x(vect_k_x, vect_k_y, xs, ys)
            idFxk, idFyk = self.__idF_x(vect_k_x, vect_k_y, xs, ys)
            vect_k1_x = vect_k_x - (idFxk[0] * Fxk[0] + idFxk[1] * Fyk[0]).unsqueeze(0).cuda()
            vect_k1_y = vect_k_y - (idFyk[0] * Fyk[0] + idFyk[1] * Fyk[0]).unsqueeze(0).cuda()
            tor = np.linalg.norm(vect_k1_x.cpu().numpy() - vect_k_x.cpu().numpy()) + np.linalg.norm(vect_k1_y.cpu().numpy() - vect_k_y.cpu().numpy())
            vect_k_x = vect_k1_x
            vect_k_y = vect_k1_y
            i += 1
            torrec.append(tor)

        # plt.plot(torrec)
        # plt.show()
        return vect_k_x.cpu().numpy(), vect_k_y.cpu().numpy()

    def __F_x(self, vect_k_x, vect_k_y, xs, ys):

        Fxk = torch.zeros_like(vect_k_x).cuda() + xs
        Fyk = torch.zeros_like(vect_k_y).cuda() + ys

        for i in range(len(self.dispinfo)):
            disptype = self.dispinfo[i][0]
            dispinfo = self.dispinfo[i][1:]
            if disptype == "planer":    #Us = AXs0 + BYs0 + C; Vs = DXs0 + EYs0 + F
                Fxk += - dispinfo[0] * vect_k_x - dispinfo[1] * vect_k_y - dispinfo[2]
                Fyk += - dispinfo[3] * vect_k_x - dispinfo[4] * vect_k_y - dispinfo[5]

            elif disptype == "sin":     #Zs = Asin(BXs + C) * sin(DYs + E); Zcr = (Asin(BXs + C) * sin(DYs + E) + minus) / btm
                Fxk += - dispinfo[0] * torch.sin(dispinfo[1]*vect_k_x + dispinfo[2]) * torch.sin(dispinfo[3]*vect_k_y + dispinfo[4])
                Fyk += - dispinfo[5] * torch.sin(dispinfo[6]*vect_k_x + dispinfo[7]) * torch.sin(dispinfo[8]*vect_k_y + dispinfo[9])

        Fxk += - vect_k_x
        Fyk += - vect_k_y

        return Fxk, Fyk

    def __idF_x(self, vect_k_x, vect_k_y, xs, ys):

        dFxk = torch.zeros(size=(2, vect_k_x.shape[1])).cuda()
        dFyk = torch.zeros(size=(2, vect_k_x.shape[1])).cuda()

        for i in range(len(self.dispinfo)):
            disptype = self.dispinfo[i][0]
            dispinfo = self.dispinfo[i][1:]
            if disptype == "planer":
                dFxk[0, :] += - (dispinfo[0]+1)
                dFxk[1, :] += - dispinfo[1]
                dFyk[0, :] += - dispinfo[3]
                dFyk[1, :] += - (dispinfo[4]+1)
            elif disptype == "sin":
                dFxk[0, :] -= dispinfo[0] * dispinfo[1] * torch.cos(dispinfo[1]*vect_k_x[0] + dispinfo[2]) * torch.sin(dispinfo[3]*vect_k_y[0] + dispinfo[4]) + 1
                dFxk[1, :] -= dispinfo[0] * dispinfo[3] * torch.sin(dispinfo[1]*vect_k_x[0] + dispinfo[2]) * torch.cos(dispinfo[3]*vect_k_y[0] + dispinfo[4])
                dFyk[0, :] -= dispinfo[5] * dispinfo[6] * torch.cos(dispinfo[6]*vect_k_x[0] + dispinfo[7]) * torch.sin(dispinfo[8]*vect_k_y[0] + dispinfo[9])
                dFyk[1, :] -= dispinfo[5] * dispinfo[8] * torch.sin(dispinfo[6]*vect_k_x[0] + dispinfo[7]) * torch.cos(dispinfo[8]*vect_k_y[0] + dispinfo[9]) + 1
        # save inverse matrix as vector
        idFxk = torch.zeros_like(dFxk).cuda()
        idFyk = torch.zeros_like(dFyk).cuda()
        btm = dFxk[0, :] * dFyk[1, :] - dFxk[1, :] * dFyk[0, :]
        idFxk[0, :] = dFyk[1, :]
        idFxk[1, :] = - dFxk[1, :]
        idFyk[0, :] = - dFyk[0, :]
        idFyk[1, :] = dFxk[0, :]
        idFxk = torch.div(idFxk, btm)
        idFyk = torch.div(idFyk, btm)

        return idFxk, idFyk