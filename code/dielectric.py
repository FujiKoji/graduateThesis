import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from numba import jit
import time
import japanize_matplotlib
import matplotlib.ticker as ptick
import dielectric_func

# 定数の定義
ε = 8.854187817e-12  #真空の誘電率
μ = 1.25663706212e-6  #真空の透磁率
ρ = 1.59e-8


c = np.sqrt(1/(ε*μ))
Nx = 300
Ny = 60
Nz = 60
Nt = 500
Δx = 0.3/Nx
Δy = 0.1/Ny
Δz = 0.1/Nz

# クーラン条件を満たす必要性
Δt = 1/(c*(((1/Δx)**2+(1/Δy)**2+(1/Δz)**2)**0.5))
T = Δt*Nt #計算時間

α1 = α2 = 1       
d1 = d2 = 0.005
bx1 = (α1*c*Δt - Δx)/(α1*c*Δt + Δx)
bx2 = (α2*c*Δt - Δx)/(α2*c*Δt + Δx)
by1 = (α1*c*Δt - Δy)/(α1*c*Δt + Δy)
by2 = (α2*c*Δt - Δy)/(α2*c*Δt + Δy)
bz1 = (α1*c*Δt - Δz)/(α1*c*Δt + Δz)
bz2 = (α2*c*Δt - Δz)/(α2*c*Δt + Δz)
σ = 2.5e-11
b = T/7 #ピーク時の時間

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
# Ax = np.zeros([Nt,Nx,Ny-1,Nz-1])
# Ay = np.zeros([Nt,Nx-1,Ny,Nz-1])
# Az = np.zeros([Nt,Nx-1,Ny-1,Nz])
Jx = np.zeros([Nt,Nx,Ny-1,Nz-1])
Jz = np.zeros([Nt,Nx-1,Ny-1,Nz])
U = np.zeros([Nt,Nx-1,Ny-1,Nz-1])
inn =4
out =0
doutai_x = np.zeros([Nx,Ny-1,Nz-1])
doutai_y = np.zeros([Nx-1,Ny,Nz-1])
doutai_z = np.zeros([Nx-1,Ny-1,Nz])
V_n = np.zeros([Nt,Nx-1])
V_s = np.zeros([Nt,Nx-1])
V_s_1 = np.zeros([Nt,Nx-1])
V_c = np.zeros([Nt,Nx-1])
I_1 = np.zeros([Nt,Nx])
I_2 = np.zeros([Nt,Nx])
I_3 = np.zeros([Nt,Nx])
I_2_in = np.zeros([Nt,Nx])
I_2_m = np.zeros([Nt,Nx])
I_2_out = np.zeros([Nt,Nx])
I_n = np.zeros([Nt,Nx])
I_s = np.zeros([Nt,Nx])
I_c = np.zeros([Nt,Nx])

for k in range(Nx-1):
    for l in range(Ny-1):
        for n in range(Nz-1):
            if 50<=k<250 and 29<=l<=31 and 29<=n<=31:
                if 50<=k<250 and l==30 and n==30:
                    Xe[k,l,n] = out
                else:
                    Xe[k,l,n] = inn
            elif 50<=k<250 and l==28 and n==30:
                Xe[k,l,n] = inn
            elif 50<=k<250 and l==32 and n==30:
                Xe[k,l,n] = inn
            elif 50<=k<250 and l==30 and n==28:
                Xe[k,l,n] = inn
            elif 50<=k<250 and l==30 and n==32:
                Xe[k,l,n] = inn
            else:
                Xe[k,l,n] = out

start = time.time()
Jx_result,Jz_result,U_result = dielectric_func.cal(Jx,Jz,U,Ax_0,Ax_1,Ax_2,Ay_0,Ay_1,Ay_2,Az_0,Az_1,Az_2,U_0,U_1,U_2,Jx_0,Jx_1,Jx_2,Jy_0,Jy_1,Jy_2,Jz_0,Jz_1,Jz_2,Xe,ρ,Nt,Nx,Ny,Nz,c,Δt,μ,ε,Δx,Δy,Δz,d1,d2,bx1,bx2,by1,by2,bz1,bz2,σ,b,doutai_x,doutai_y,doutai_z)
processtime = time.time()-start
print(processtime)

I_1_result,I_2_result,I_3_result,I_n_result,I_s_result,I_c_result = dielectric_func.origin(Nt,Nx,Ny,Nz,Jx_result,I_1,I_2,I_3,I_n,I_s,I_c,I_2_in,I_2_out)

fig, ax = plt.subplots(figsize=(12,7.3))
def anime(i):
    plt.clf()
    ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True)) 
    ax.yaxis.offsetText.set_fontsize(80)
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlabel("距離(mm)",fontsize=30)
    plt.ylabel("電流(A)",fontsize=30)
    plt.ylim(-6e-8,6e-8)
    plt.plot(I_2_out[2*i,50:251],color = "orange")
    plt.plot(I_3_result[2*i,50:251],color = "red")
    plt.gca().yaxis.set_tick_params(direction='in')
    plt.gca().xaxis.set_tick_params(direction='in')
    ax.tick_params(labelsize=18)
    plt.rcParams["font.size"] = 25
anim = animation.FuncAnimation(fig,anime,frames=249,interval=100)
anim.save("out.gif",writer="imagemagick")
