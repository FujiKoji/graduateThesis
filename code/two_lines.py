import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from numba import jit
import time
import japanize_matplotlib
import matplotlib.ticker as ptick
#from mpl_toolkits.mplot3d import Axes3D
import lines_module.py

# 定数の定義
ε = 8.854187817e-12  #真空の誘電率
μ = 1.25663706212e-6  #真空の透磁率
ρ = 1.59e-8


c = np.sqrt(1/(ε*μ))
Nx = 100
Ny = 100
Nz = 100
Nt = 300
Δx = 0.1/Nx
Δy = 0.1/Ny
Δz = 0.1/Nz
end_n=200
end_s=180


# クーラン条件を満たす必要性
Δt = 1/(c*(((1/Δx)**2+(1/Δy)**2+(1/Δz)**2)**0.5))
T = Δt*Nt #計算時間
σ = 2.5e-11 #半値全幅に関係する
b = T/7 #ピーク時の時間

α1 = α2 = 1       
d1 = d2 = 0.005   
bx1 = (α1*c*Δt - Δx)/(α1*c*Δt + Δx)
bx2 = (α2*c*Δt - Δx)/(α2*c*Δt + Δx)
by1 = (α1*c*Δt - Δy)/(α1*c*Δt + Δy)
by2 = (α2*c*Δt - Δy)/(α2*c*Δt + Δy)
bz1 = (α1*c*Δt - Δz)/(α1*c*Δt + Δz)
bz2 = (α2*c*Δt - Δz)/(α2*c*Δt + Δz)

# 変数の定義
Jx_0 = np.zeros([Nx,Ny-1,Nz-1])
Jx_1 = np.zeros([Nx,Ny-1,Nz-1])
Jx_2 = np.zeros([Nx,Ny-1,Nz-1])
Jy_0 = np.zeros([Nx-1,Ny,Nz-1])
Jy_1 = np.zeros([Nx-1,Ny,Nz-1])
Jy_2 = np.zeros([Nx-1,Ny,Nz-1])
Jz_0 = np.zeros([Nx-1,Ny-1,Nz])
Jz_1 = np.zeros([Nx-1,Ny-1,Nz])
Jz_2 = np.zeros([Nx-1,Ny-1,Nz])
Ax_0 = np.zeros([Nx,Ny-1,Nz-1])
Ax_1 = np.zeros([Nx,Ny-1,Nz-1])
Ax_2 = np.zeros([Nx,Ny-1,Nz-1])
Ay_0 = np.zeros([Nx-1,Ny,Nz-1])
Ay_1 = np.zeros([Nx-1,Ny,Nz-1])
Ay_2 = np.zeros([Nx-1,Ny,Nz-1])
Az_0 = np.zeros([Nx-1,Ny-1,Nz])
Az_1 = np.zeros([Nx-1,Ny-1,Nz])
Az_2 = np.zeros([Nx-1,Ny-1,Nz])
U_0 = np.zeros([Nx-1,Ny-1,Nz-1])
U_1 = np.zeros([Nx-1,Ny-1,Nz-1])
U_2 = np.zeros([Nx-1,Ny-1,Nz-1])
Xe = np.zeros([Nx-1,Ny-1,Nz-1])
Jx = np.zeros([Nt,Nx,Ny-1,Nz-1])
U = np.zeros([Nt,Nx-1,Ny-1,Nz-1])
P_n_max_0=np.zeros([Nx-1])
P_s_max_0=np.zeros([Nx-1])

inn =0
out =0

for k in range(Nx-1):
    for l in range(Ny-1):
        for n in range(Nz-1):
            if 20<=k<80 and 49<=l<=51 and 49<=n<=50:
                if k==20 and l==50 and 49<=n<=50:
                    Xe[k,l,n] = out
                else:
                    Xe[k,l,n] = inn
            else:
                Xe[k,l,n] = out

σ = 2.5e-11 #半値全幅に関係する
b = T/7 #ピーク時の時間

start = time.time()
Jx_result,U_result = lines_module.cal(Jx,U,Ax_0,Ax_1,Ax_2,Ay_0,Ay_1,Ay_2,Az_0,Az_1,Az_2,U_0,U_1,U_2,Jx_0,Jx_1,Jx_2,Jy_0,Jy_1,Jy_2,Jz_0,Jz_1,Jz_2,Xe,ρ,Nt,Nx,Ny,Nz,c,Δt,μ,ε,Δx,Δy,Δz,d1,d2,bx1,bx2,by1,by2,bz1,bz2,σ,b)
processtime = time.time()-start
print(processtime)

V_n_0,V_s_0,I_n_0,I_s_0 = lines_module.common(V_n,V_s,I_n,I_s,I_1,I_2,U,Jx,Nt,Nx)
