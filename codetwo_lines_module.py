import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from numba import jit
import time
import japanize_matplotlib
import matplotlib.ticker as ptick
#from mpl_toolkits.mplot3d import Axes3D

@jit(nopython=True)
def gausian(t,σ,b):
    result = 2*(1/((2*3.14)**0.5)*σ)*math.exp(-((t-b)**2)/(2*σ**2))/1.9950963996680075e-11
    return result

@jit(nopython=True)
def cal(Jx,U,Ax_0,Ax_1,Ax_2,Ay_0,Ay_1,Ay_2,Az_0,Az_1,Az_2,U_0,U_1,U_2,Jx_0,Jx_1,Jx_2,Jy_0,Jy_1,Jy_2,Jz_0,Jz_1,Jz_2,Xe,ρ,Nt,Nx,Ny,Nz,c,Δt,μ,ε,Δx,Δy,Δz,d1,d2,bx1,bx2,by1,by2,bz1,bz2,σ,b):
    for m in range(Nt-1):
        print(m)
        for k in range(Nx):
            for l in range(Ny-1):
                for n in range(Nz-1):
                    if k==0 or k==Nx-1 or l==0 or l==Ny-2 or n==0 or n==Nz-2:
                        Ax_2[0,l,n] = (bx1+bx2)*(Ax_2[1,l,n]-Ax_1[0,l,n]) \
                                        -bx1*bx2*(Ax_2[2,l,n]-2*Ax_1[1,l,n]+Ax_0[0,l,n]) \
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Ax_1[2,l,n]-Ax_0[1,l,n]) \
                                        +((1-d1)+(1-d2))*Ax_1[1,l,n] \
                                        -(1-d1)*(1-d2)*Ax_0[2,l,n]
                        Ax_2[-1,l,n] = (bx1+bx2)*(Ax_2[-2,l,n]-Ax_1[-1,l,n]) \
                                        -bx1*bx2*(Ax_2[-3,l,n]-2*Ax_1[-2,l,n]+Ax_0[-1,l,n]) \
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Ax_1[-3,l,n]-Ax_0[-2,l,n])\
                                        +((1-d1)+(1-d2))*Ax_1[-2,l,n]\
                                        -(1-d1)*(1-d2)*Ax_0[-3,l,n]
                        Ax_2[k,0,n] = (by1+by2)*(Ax_2[k,1,n]-Ax_1[k,0,n]) \
                                        -by1*by2*(Ax_2[k,2,n]-2*Ax_1[k,1,n]+Ax_0[k,0,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Ax_1[k,2,n]-Ax_0[k,1,n])\
                                        +((1-d1)+(1-d2))*Ax_1[k,1,n]\
                                        -(1-d1)*(1-d2)*Ax_0[k,2,n]
                        Ax_2[k,-1,n] = (by1+by2)*(Ax_2[k,-2,n]-Ax_1[k,-1,n])\
                                        -by1*by2*(Ax_2[k,-3,n]-2*Ax_1[k,-2,n]+Ax_0[k,-1,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Ax_1[k,-3,n]-Ax_0[k,-2,n])\
                                        +((1-d1)+(1-d2))*Ax_1[k,-2,n]\
                                        -(1-d1)*(1-d2)*Ax_0[k,-3,n]
                        Ax_2[k,l,0] = (bz1+bz2)*(Ax_2[k,l,1]-Ax_1[k,l,0])\
                                        -bz1*bz2*(Ax_2[k,l,2]-2*Ax_1[k,l,1]+Ax_0[k,l,0])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Ax_1[k,l,2]-Ax_0[k,l,1])\
                                        +((1-d1)+(1-d2))*Ax_1[k,l,1]\
                                        -(1-d1)*(1-d2)*Ax_0[k,l,2]
                        Ax_2[k,l,-1] = (bz1+bz2)*(Ax_2[k,l,-2]-Ax_1[k,l,-1])\
                                        -bz1*bz2*(Ax_2[k,l,-3]-2*Ax_1[k,l,-2]+Ax_0[k,l,-1])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Ax_1[k,l,-3]-Ax_0[k,l,-2])\
                                        +((1-d1)+(1-d2))*Ax_1[k,l,-2]\
                                        -(1-d1)*(1-d2)*Ax_0[k,l,-3]
                        Jx_2[0,l,n] = (bx1+bx2)*(Jx_2[1,l,n]-Jx_1[0,l,n]) \
                                        -bx1*bx2*(Jx_2[2,l,n]-2*Jx_1[1,l,n]+Jx_0[0,l,n]) \
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Jx_1[2,l,n]-Jx_0[1,l,n]) \
                                        +((1-d1)+(1-d2))*Jx_1[1,l,n] \
                                        -(1-d1)*(1-d2)*Jx_0[2,l,n]
                        Jx_2[-1,l,n] = (bx1+bx2)*(Jx_2[-2,l,n]-Jx_1[-1,l,n]) \
                                        -bx1*bx2*(Jx_2[-3,l,n]-2*Jx_1[-2,l,n]+Jx_0[-1,l,n]) \
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Jx_1[-3,l,n]-Jx_0[-2,l,n])\
                                        +((1-d1)+(1-d2))*Jx_1[-2,l,n]\
                                        -(1-d1)*(1-d2)*Jx_0[-3,l,n]
                        Jx_2[k,0,n] = (by1+by2)*(Jx_2[k,1,n]-Jx_1[k,0,n]) \
                                        -by1*by2*(Jx_2[k,2,n]-2*Jx_1[k,1,n]+Jx_0[k,0,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Jx_1[k,2,n]-Jx_0[k,1,n])\
                                        +((1-d1)+(1-d2))*Jx_1[k,1,n]\
                                        -(1-d1)*(1-d2)*Jx_0[k,2,n]
                        Jx_2[k,-1,n] = (by1+by2)*(Jx_2[k,-2,n]-Jx_1[k,-1,n])\
                                        -by1*by2*(Jx_2[k,-3,n]-2*Jx_1[k,-2,n]+Jx_0[k,-1,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Jx_1[k,-3,n]-Jx_0[k,-2,n])\
                                        +((1-d1)+(1-d2))*Jx_1[k,-2,n]\
                                        -(1-d1)*(1-d2)*Jx_0[k,-3,n]
                        Jx_2[k,l,0] = (bz1+bz2)*(Jx_2[k,l,1]-Jx_1[k,l,0])\
                                        -bz1*bz2*(Jx_2[k,l,2]-2*Jx_1[k,l,1]+Jx_0[k,l,0])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Jx_1[k,l,2]-Jx_0[k,l,1])\
                                        +((1-d1)+(1-d2))*Jx_1[k,l,1]\
                                        -(1-d1)*(1-d2)*Jx_0[k,l,2]
                        Jx_2[k,l,-1] = (bz1+bz2)*(Jx_2[k,l,-2]-Jx_1[k,l,-1])\
                                        -bz1*bz2*(Jx_2[k,l,-3]-2*Jx_1[k,l,-2]+Jx_0[k,l,-1])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Jx_1[k,l,-3]-Jx_0[k,l,-2])\
                                        +((1-d1)+(1-d2))*Jx_1[k,l,-2]\
                                        -(1-d1)*(1-d2)*Jx_0[k,l,-3]
                    elif (20<k<=80 and 49<=l<=51 and n==48) or (20<k<=80 and l==50 and n==51):#導体
                        Xe_ave = (Xe[k,l,n]+Xe[k-1,l,n])/2
                        Cx = (Δt**2)*(
                                        (Ax_1[k+1,l,n]-2*Ax_1[k,l,n]+Ax_1[k-1,l,n])/(Δx**2)
                                        +(Ax_1[k,l+1,n]-2*Ax_1[k,l,n]+Ax_1[k,l-1,n])/(Δy**2)
                                        +(Ax_1[k,l,n+1]-2*Ax_1[k,l,n]+Ax_1[k,l,n-1])/(Δz**2)
                                        -ε*μ*Xe_ave*(U_1[k,l,n]-U_0[k,l,n]-U_1[k-1,l,n]+U_0[k-1,l,n])/(Δx*Δt)
                                        +μ*(Jx_1[k,l,n]+Jx_0[k,l,n])/3
                                        )/(ε*μ*(1+Xe_ave)) \
                                +2*Ax_1[k,l,n]-Ax_0[k,l,n]
                        Dx = -(Δt/Δx)*(U_1[k,l,n]-U_1[k-1,l,n])\
                            +Ax_1[k,l,n]\
                            -ρ*Δt*Jx_1[k,l,n]/2
                        Ax_2[k,l,n] = 1/(ρ*Δt/2+(Δt**2)/(3*ε*(1+Xe_ave)))\
                                        *(ρ*Δt*Cx/2+(Δt**2)*Dx/(3*ε*(1+Xe_ave)))
                        Jx_2[k,l,n] = 1/(ρ*Δt/2+(Δt**2)/(3*ε*(1+Xe_ave)))\
                                        *(-Cx+Dx)
                    else:
                        Xe_ave = (Xe[k,l,n]+Xe[k-1,l,n])/2 

                        Ax_2[k,l,n] = (Δt**2)*(
                                                (Ax_1[k+1,l,n]-2*Ax_1[k,l,n]+Ax_1[k-1,l,n])/(Δx**2)
                                                +(Ax_1[k,l+1,n]-2*Ax_1[k,l,n]+Ax_1[k,l-1,n])/(Δy**2)
                                                +(Ax_1[k,l,n+1]-2*Ax_1[k,l,n]+Ax_1[k,l,n-1])/(Δz**2)
                                                -ε*μ*Xe_ave*(U_1[k,l,n]-U_0[k,l,n]-U_1[k-1,l,n]+U_0[k-1,l,n])/(Δx*Δt)
                                                )/(ε*μ*(1+Xe_ave)) \
                                        +2*Ax_1[k,l,n]-Ax_0[k,l,n]
        for k in range(Nx-1):
            for l in range(Ny):
                for n in range(Nz-1):
                    if k==0 or k==Nx-2 or l==0 or l==Ny-1 or n==0 or n==Nz-2:
                        Ay_2[0,l,n] = (bx1+bx2)*(Ay_2[1,l,n]-Ay_1[0,l,n])\
                                        -bx1*bx2*(Ay_2[2,l,n]-2*Ay_1[1,l,n]+Ay_0[0,l,n])\
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Ay_1[2,l,n]-Ay_0[1,l,n])\
                                        +((1-d1)+(1-d2))*Ay_1[1,l,n]\
                                        -(1-d1)*(1-d2)*Ay_0[2,l,n]
                        Ay_2[-1,l,n] = (bx1+bx2)*(Ay_2[-2,l,n]-Ay_1[-1,l,n])\
                                        -bx1*bx2*(Ay_2[-3,l,n]-2*Ay_1[-2,l,n]+Ay_0[-1,l,n])\
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Ay_1[-3,l,n]-Ay_0[-2,l,n])\
                                        +((1-d1)+(1-d2))*Ay_1[-2,l,n]\
                                        -(1-d1)*(1-d2)*Ay_0[-3,l,n]
                        Ay_2[k,0,n] = (by1+by2)*(Ay_2[k,1,n]-Ay_1[k,0,n])\
                                        -by1*by2*(Ay_2[k,2,n]-2*Ay_1[k,1,n]+Ay_0[k,0,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Ay_1[k,2,n]-Ay_0[k,1,n])\
                                        +((1-d1)+(1-d2))*Ay_1[k,1,n]\
                                        -(1-d1)*(1-d2)*Ay_0[k,2,n]
                        Ay_2[k,-1,n] = (by1+by2)*(Ay_2[k,-2,n]-Ay_1[k,-1,n])\
                                        -by1*by2*(Ay_2[k,-3,n]-2*Ay_1[k,-2,n]+Ay_0[k,-1,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Ay_1[k,-3,n]-Ay_0[k,-2,n])\
                                        +((1-d1)+(1-d2))*Ay_1[k,-2,n]\
                                        -(1-d1)*(1-d2)*Ay_0[k,-3,n]
                        Ay_2[k,l,0] = (bz1+bz2)*(Ay_2[k,l,1]-Ay_1[k,l,0])\
                                        -bz1*bz2*(Ay_2[k,l,2]-2*Ay_1[k,l,1]+Ay_0[k,l,0])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Ay_1[k,l,2]-Ay_0[k,l,1])\
                                        +((1-d1)+(1-d2))*Ay_1[k,l,1]\
                                        -(1-d1)*(1-d2)*Ay_0[k,l,2]
                        Ay_2[k,l,-1] = (bz1+bz2)*(Ay_2[k,l,-2]-Ay_1[k,l,-1])\
                                        -bz1*bz2*(Ay_2[k,l,-3]-2*Ay_1[k,l,-2]+Ay_0[k,l,-1])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Ay_1[k,l,-3]-Ay_0[k,l,-2])\
                                        +((1-d1)+(1-d2))*Ay_1[k,l,-2]\
                                        -(1-d1)*(1-d2)*Ay_0[k,l,-3]
                        Jy_2[0,l,n] = (bx1+bx2)*(Jy_2[1,l,n]-Jy_1[0,l,n])\
                                        -bx1*bx2*(Jy_2[2,l,n]-2*Jy_1[1,l,n]+Jy_0[0,l,n])\
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Jy_1[2,l,n]-Jy_0[1,l,n])\
                                        +((1-d1)+(1-d2))*Jy_1[1,l,n]\
                                        -(1-d1)*(1-d2)*Jy_0[2,l,n]
                        Jy_2[-1,l,n] = (bx1+bx2)*(Jy_2[-2,l,n]-Jy_1[-1,l,n])\
                                        -bx1*bx2*(Jy_2[-3,l,n]-2*Jy_1[-2,l,n]+Jy_0[-1,l,n])\
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Jy_1[-3,l,n]-Jy_0[-2,l,n])\
                                        +((1-d1)+(1-d2))*Jy_1[-2,l,n]\
                                        -(1-d1)*(1-d2)*Jy_0[-3,l,n]
                        Jy_2[k,0,n] = (by1+by2)*(Jy_2[k,1,n]-Jy_1[k,0,n])\
                                        -by1*by2*(Jy_2[k,2,n]-2*Jy_1[k,1,n]+Jy_0[k,0,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Jy_1[k,2,n]-Jy_0[k,1,n])\
                                        +((1-d1)+(1-d2))*Jy_1[k,1,n]\
                                        -(1-d1)*(1-d2)*Jy_0[k,2,n]
                        Jy_2[k,-1,n] = (by1+by2)*(Jy_2[k,-2,n]-Jy_1[k,-1,n])\
                                        -by1*by2*(Jy_2[k,-3,n]-2*Jy_1[k,-2,n]+Jy_0[k,-1,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Jy_1[k,-3,n]-Jy_0[k,-2,n])\
                                        +((1-d1)+(1-d2))*Jy_1[k,-2,n]\
                                        -(1-d1)*(1-d2)*Jy_0[k,-3,n]
                        Jy_2[k,l,0] = (bz1+bz2)*(Jy_2[k,l,1]-Jy_1[k,l,0])\
                                        -bz1*bz2*(Jy_2[k,l,2]-2*Jy_1[k,l,1]+Jy_0[k,l,0])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Jy_1[k,l,2]-Jy_0[k,l,1])\
                                        +((1-d1)+(1-d2))*Jy_1[k,l,1]\
                                        -(1-d1)*(1-d2)*Jy_0[k,l,2]
                        Jy_2[k,l,-1] = (bz1+bz2)*(Jy_2[k,l,-2]-Jy_1[k,l,-1])\
                                        -bz1*bz2*(Jy_2[k,l,-3]-2*Jy_1[k,l,-2]+Jy_0[k,l,-1])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Jy_1[k,l,-3]-Jy_0[k,l,-2])\
                                        +((1-d1)+(1-d2))*Jy_1[k,l,-2]\
                                        -(1-d1)*(1-d2)*Jy_0[k,l,-3]
                    elif k==20 and 50<=l<=51 and n==48:#導体
                        Xe_ave = (Xe[k,l,n]+Xe[k,l-1,n])/2
                        Cy = (Δt**2)*(
                                        (Ay_1[k+1,l,n]-2*Ay_1[k,l,n]+Ay_1[k-1,l,n])/(Δx**2)
                                        +(Ay_1[k,l+1,n]-2*Ay_1[k,l,n]+Ay_1[k,l-1,n])/(Δy**2)
                                        +(Ay_1[k,l,n+1]-2*Ay_1[k,l,n]+Ay_1[k,l,n-1])/(Δz**2)
                                        -ε*μ*Xe_ave*(U_1[k,l,n]-U_0[k,l,n]-U_1[k,l-1,n]+U_0[k,l-1,n])/(Δx*Δt)
                                        +μ*(Jy_1[k,l,n]+Jy_0[k,l,n])/3
                                        )/(ε*μ*(1+Xe_ave)) \
                                +2*Ay_1[k,l,n]-Ay_0[k,l,n]
                        Dy = -(Δt/Δy)*(U_1[k,l,n]-U_1[k,l-1,n])\
                            +Ay_1[k,l,n]\
                            -ρ*Δt*Jy_1[k,l,n]/2
                        Ay_2[k,l,n] = 1/(ρ*Δt/2+(Δt**2)/(3*ε*(1+Xe_ave)))\
                                        *(ρ*Δt*Cy/2+(Δt**2)*Dy/(3*ε*(1+Xe_ave)))
                        Jy_2[k,l,n] = 1/(ρ*Δt/2+(Δt**2)/(3*ε*(1+Xe_ave)))\
                                        *(-Cy+Dy)
                    else:
                        Xe_ave = (Xe[k,l,n]+Xe[k,l-1,n])/2
                        Ay_2[k,l,n] = (Δt**2)*(
                                                (Ay_1[k+1,l,n]-2*Ay_1[k,l,n]+Ay_1[k-1,l,n])/(Δx**2)
                                                +(Ay_1[k,l+1,n]-2*Ay_1[k,l,n]+Ay_1[k,l-1,n])/(Δy**2)
                                                +(Ay_1[k,l,n+1]-2*Ay_1[k,l,n]+Ay_1[k,l,n-1])/(Δz**2)
                                                -ε*μ*Xe_ave*(U_1[k,l,n]-U_0[k,l,n]-U_1[k,l-1,n]+U_0[k,l-1,n])/(Δy*Δt)
                                                )/(ε*μ*(1+Xe_ave)) \
                                        +2*Ay_1[k,l,n]-Ay_0[k,l,n]
        for k in range(Nx-1):
            for l in range(Ny-1):
                for n in range(Nz):
                    if k==0 or k==Nx-2 or l==0 or l==Ny-2 or n==0 or n==Nz-1:
                        Az_2[0,l,n] = (bx1+bx2)*(Az_2[1,l,n]-Az_1[0,l,n])\
                                        -bx1*bx2*(Az_2[2,l,n]-2*Az_1[1,l,n]+Az_0[0,l,n])\
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Az_1[2,l,n]-Az_0[1,l,n])\
                                        +((1-d1)+(1-d2))*Az_1[1,l,n]\
                                        -(1-d1)*(1-d2)*Az_0[2,l,n]
                        Az_2[-1,l,n] = (bx1+bx2)*(Az_2[-2,l,n]-Az_1[-1,l,n])\
                                        -bx1*bx2*(Az_2[-3,l,n]-2*Az_1[-2,l,n]+Az_0[-1,l,n])\
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Az_1[-3,l,n]-Az_0[-2,l,n])\
                                        +((1-d1)+(1-d2))*Az_1[-2,l,n]\
                                        -(1-d1)*(1-d2)*Az_0[-3,l,n]
                        Az_2[k,0,n] = (by1+by2)*(Az_2[k,1,n]-Az_1[k,0,n])\
                                        -by1*by2*(Az_2[k,2,n]-2*Az_1[k,1,n]+Az_0[k,0,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Az_1[k,2,n]-Az_0[k,1,n])\
                                        +((1-d1)+(1-d2))*Az_1[k,1,n]\
                                        -(1-d1)*(1-d2)*Az_0[k,2,n]
                        Az_2[k,-1,n] = (by1+by2)*(Az_2[k,-2,n]-Az_1[k,-1,n])\
                                        -by1*by2*(Az_2[k,-3,n]-2*Az_1[k,-2,n]+Az_0[k,-1,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Az_1[k,-3,n]-Az_0[k,-2,n])\
                                        +((1-d1)+(1-d2))*Az_1[k,-2,n]\
                                        -(1-d1)*(1-d2)*Az_0[k,-3,n]
                        Az_2[k,l,0] = (bz1+bz2)*(Az_2[k,l,1]-Az_1[k,l,0])\
                                        -bz1*bz2*(Az_2[k,l,2]-2*Az_1[k,l,1]+Az_0[k,l,0])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Az_1[k,l,2]-Az_0[k,l,1])\
                                        +((1-d1)+(1-d2))*Az_1[k,l,1]\
                                        -(1-d1)*(1-d2)*Az_0[k,l,2]
                        Az_2[k,l,-1] = (bz1+bz2)*(Az_2[k,l,-2]-Az_1[k,l,-1])\
                                        -bz1*bz2*(Az_2[k,l,-3]-2*Az_1[k,l,-2]+Az_0[k,l,-1])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Az_1[k,l,-3]-Az_0[k,l,-2])\
                                        +((1-d1)+(1-d2))*Az_1[k,l,-2]\
                                        -(1-d1)*(1-d2)*Az_0[k,l,-3]
                        Jz_2[0,l,n] = (bx1+bx2)*(Jz_2[1,l,n]-Jz_1[0,l,n])\
                                        -bx1*bx2*(Jz_2[2,l,n]-2*Jz_1[1,l,n]+Jz_0[0,l,n])\
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Jz_1[2,l,n]-Jz_0[1,l,n])\
                                        +((1-d1)+(1-d2))*Jz_1[1,l,n]\
                                        -(1-d1)*(1-d2)*Jz_0[2,l,n]
                        Jz_2[-1,l,n] = (bx1+bx2)*(Jz_2[-2,l,n]-Jz_1[-1,l,n])\
                                        -bx1*bx2*(Jz_2[-3,l,n]-2*Jz_1[-2,l,n]+Jz_0[-1,l,n])\
                                        -(bx1*(1-d1)+bx2*(1-d2))*(Jz_1[-3,l,n]-Jz_0[-2,l,n])\
                                        +((1-d1)+(1-d2))*Jz_1[-2,l,n]\
                                        -(1-d1)*(1-d2)*Jz_0[-3,l,n]
                        Jz_2[k,0,n] = (by1+by2)*(Jz_2[k,1,n]-Jz_1[k,0,n])\
                                        -by1*by2*(Jz_2[k,2,n]-2*Jz_1[k,1,n]+Jz_0[k,0,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Jz_1[k,2,n]-Jz_0[k,1,n])\
                                        +((1-d1)+(1-d2))*Jz_1[k,1,n]\
                                        -(1-d1)*(1-d2)*Jz_0[k,2,n]
                        Jz_2[k,-1,n] = (by1+by2)*(Jz_2[k,-2,n]-Jz_1[k,-1,n])\
                                        -by1*by2*(Jz_2[k,-3,n]-2*Jz_1[k,-2,n]+Jz_0[k,-1,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(Jz_1[k,-3,n]-Jz_0[k,-2,n])\
                                        +((1-d1)+(1-d2))*Jz_1[k,-2,n]\
                                        -(1-d1)*(1-d2)*Jz_0[k,-3,n]
                        Jz_2[k,l,0] = (bz1+bz2)*(Jz_2[k,l,1]-Jz_1[k,l,0])\
                                        -bz1*bz2*(Jz_2[k,l,2]-2*Jz_1[k,l,1]+Jz_0[k,l,0])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Jz_1[k,l,2]-Jz_0[k,l,1])\
                                        +((1-d1)+(1-d2))*Jz_1[k,l,1]\
                                        -(1-d1)*(1-d2)*Jz_0[k,l,2]
                        Jz_2[k,l,-1] = (bz1+bz2)*(Jz_2[k,l,-2]-Jz_1[k,l,-1])\
                                        -bz1*bz2*(Jz_2[k,l,-3]-2*Jz_1[k,l,-2]+Jz_0[k,l,-1])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(Jz_1[k,l,-3]-Jz_0[k,l,-2])\
                                        +((1-d1)+(1-d2))*Jz_1[k,l,-2]\
                                        -(1-d1)*(1-d2)*Jz_0[k,l,-3]
                    elif k==20 and l==50 and 48<n<=51:#導体
                        if k==20 and l==50 and n==50:
                            Jz_0[20,50,50] = gausian(Δt*(m-1),σ,b)
                            Jz_1[20,50,50] = gausian(Δt*m,σ,b)
                            Jz_2[20,50,50] = gausian(Δt*(m+1),σ,b)                     
                            Az_2[k,l,n] = (Δt**2)*(
                                            (Az_1[k+1,l,n]-2*Az_1[k,l,n]+Az_1[k-1,l,n])/(Δx**2)
                                            +(Az_1[k,l+1,n]-2*Az_1[k,l,n]+Az_1[k,l-1,n])/(Δy**2)
                                            +(Az_1[k,l,n+1]-2*Az_1[k,l,n]+Az_1[k,l,n-1])/(Δz**2)
                                            -ε*μ*Xe_ave*(U_1[k,l,n]-U_0[k,l,n]-U_1[k,l,n-1]+U_0[k,l,n-1])/(Δz*Δt)
                                            +μ*(Jz_2[k,l,n]+Jz_1[k,l,n]+Jz_0[k,l,n])/3
                                            )/(ε*μ*(1+Xe_ave)) \
                                    +2*Az_1[k,l,n]-Az_0[k,l,n]
                        else:
                            Xe_ave = (Xe[k,l,n]+Xe[k,l,n-1])/2
                            Cz = (Δt**2)*(
                                            (Az_1[k+1,l,n]-2*Az_1[k,l,n]+Az_1[k-1,l,n])/(Δx**2)
                                            +(Az_1[k,l+1,n]-2*Az_1[k,l,n]+Az_1[k,l-1,n])/(Δy**2)
                                            +(Az_1[k,l,n+1]-2*Az_1[k,l,n]+Az_1[k,l,n-1])/(Δz**2)
                                            -ε*μ*Xe_ave*(U_1[k,l,n]-U_0[k,l,n]-U_1[k,l,n-1]+U_0[k,l,n-1])/(Δx*Δt)
                                            +μ*(Jz_1[k,l,n]+Jz_0[k,l,n])/3
                                            )/(ε*μ*(1+Xe_ave)) \
                                    +2*Az_1[k,l,n]-Az_0[k,l,n]
                            Dz = -(Δt/Δy)*(U_1[k,l,n]-U_1[k,l,n-1])\
                                +Az_1[k,l,n]\
                                -ρ*Δt*Jz_1[k,l,n]/2
                            Az_2[k,l,n] = 1/(ρ*Δt/2+(Δt**2)/(3*ε*(1+Xe_ave)))\
                                            *(ρ*Δt*Cz/2+(Δt**2)*Dz/(3*ε*(1+Xe_ave)))
                            Jz_2[k,l,n] = 1/(ρ*Δt/2+(Δt**2)/(3*ε*(1+Xe_ave)))\
                                            *(-Cz+Dz)
                    else:
                        Xe_ave = (Xe[k,l,n]+Xe[k,l,n-1])/2
                        Az_2[k,l,n] = (Δt**2)*(
                                                (Az_1[k+1,l,n]-2*Az_1[k,l,n]+Az_1[k-1,l,n])/(Δx**2)
                                                +(Az_1[k,l+1,n]-2*Az_1[k,l,n]+Az_1[k,l-1,n])/(Δy**2)
                                                +(Az_1[k,l,n+1]-2*Az_1[k,l,n]+Az_1[k,l,n-1])/(Δz**2)
                                                -ε*μ*Xe_ave*(U_1[k,l,n]-U_0[k,l,n]-U_1[k,l,n-1]+U_0[k,l,n-1])/(Δz*Δt)
                                                )/(ε*μ*(1+Xe_ave)) \
                                        +2*Az_1[k,l,n]-Az_0[k,l,n]
        for k in range(Nx-1):
            for l in range(Ny-1):
                for n in range(Nz-1):
                    if k==0 or k==Nx-2 or l==0 or l==Ny-2 or n==0 or n==Nz-2:
                        U_2[0,l,n] = (bx1+bx2)*(U_2[1,l,n]-U_1[0,l,n])\
                                        -bx1*bx2*(U_2[2,l,n]-2*U_1[1,l,n]+U_0[0,l,n])\
                                        -(bx1*(1-d1)+bx2*(1-d2))*(U_1[2,l,n]-U_0[1,l,n])\
                                        +((1-d1)+(1-d2))*U_1[1,l,n]\
                                        -(1-d1)*(1-d2)*U_0[2,l,n]
                        U_2[-1,l,n] = (bx1+bx2)*(U_2[-2,l,n]-U_1[-1,l,n])\
                                        -bx1*bx2*(U_2[-3,l,n]-2*U_1[-2,l,n]+U_0[-1,l,n])\
                                        -(bx1*(1-d1)+bx2*(1-d2))*(U_1[-3,l,n]-U_0[-2,l,n])\
                                        +((1-d1)+(1-d2))*U_1[-2,l,n]\
                                        -(1-d1)*(1-d2)*U_0[-3,l,n]
                        U_2[k,0,n] = (by1+by2)*(U_2[k,1,n]-U_1[k,0,n])\
                                        -by1*by2*(U_2[k,2,n]-2*U_1[k,1,n]+U_0[k,0,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(U_1[k,2,n]-U_0[k,1,n])\
                                        +((1-d1)+(1-d2))*U_1[k,1,n]\
                                        -(1-d1)*(1-d2)*U_0[k,2,n]
                        U_2[k,-1,n] = (by1+by2)*(U_2[k,-2,n]-U_1[k,-1,n])\
                                        -by1*by2*(U_2[k,-3,n]-2*U_1[k,-2,n]+U_0[k,-1,n])\
                                        -(by1*(1-d1)+by2*(1-d2))*(U_1[k,-3,n]-U_0[k,-2,n])\
                                        +((1-d1)+(1-d2))*U_1[k,-2,n]\
                                        -(1-d1)*(1-d2)*U_0[k,-3,n]
                        U_2[k,l,0] = (bz1+bz2)*(U_2[k,l,1]-U_1[k,l,0])\
                                        -bz1*bz2*(U_2[k,l,2]-2*U_1[k,l,1]+U_0[k,l,0])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(U_1[k,l,2]-U_0[k,l,1])\
                                        +((1-d1)+(1-d2))*U_1[k,l,1]\
                                        -(1-d1)*(1-d2)*U_0[k,l,2]
                        U_2[k,l,-1] = (bz1+bz2)*(U_2[k,l,-2]-U_1[k,l,-1])\
                                        -bz1*bz2*(U_2[k,l,-3]-2*U_1[k,l,-2]+U_0[k,l,-1])\
                                        -(bz1*(1-d1)+bz2*(1-d2))*(U_1[k,l,-3]-U_0[k,l,-2])\
                                        +((1-d1)+(1-d2))*U_1[k,l,-2]\
                                        -(1-d1)*(1-d2)*U_0[k,l,-3]
                    else:
                        U_2[k,l,n] = -(c**2)*Δt*(
                                                    (Ax_2[k+1,l,n]-Ax_2[k,l,n])/Δx
                                                    +(Ay_2[k,l+1,n]-Ay_2[k,l,n])/Δy
                                                    +(Az_2[k,l,n+1]-Az_2[k,l,n])/Δz)\
                                                +U_1[k,l,n]
        Jx[m+1,:,:,:] = Jx_2
        U[m+1,:,:,:] = U_2
        Ax_0 = np.copy(Ax_1)
        Ax_1 = np.copy(Ax_2)
        Ay_0 = np.copy(Ay_1)
        Ay_1 = np.copy(Ay_2)
        Az_0 = np.copy(Az_1)
        Az_1 = np.copy(Az_2)
        Jx_0 = np.copy(Jx_1)
        Jx_1 = np.copy(Jx_2)
        Jy_0 = np.copy(Jy_1)
        Jy_1 = np.copy(Jy_2)
        Jz_0 = np.copy(Jz_1)
        Jz_1 = np.copy(Jz_2)
        U_0 = np.copy(U_1)
        U_1 = np.copy(U_2)
    return Jx,U

start = time.time()
Jx_result,U_result = cal(Jx,U,Ax_0,Ax_1,Ax_2,Ay_0,Ay_1,Ay_2,Az_0,Az_1,Az_2,U_0,U_1,U_2,Jx_0,Jx_1,Jx_2,Jy_0,Jy_1,Jy_2,Jz_0,Jz_1,Jz_2,Xe,ρ,Nt,Nx,Ny,Nz,c,Δt,μ,ε,Δx,Δy,Δz,d1,d2,bx1,bx2,by1,by2,bz1,bz2,σ,b)
processtime = time.time()-start

print(processtime)

#NMとCMの導出
V_n = np.zeros([Nt,Nx-1])
V_s = np.zeros([Nt,Nx-1])
I_n = np.zeros([Nt,Nx])
I_s = np.zeros([Nt,Nx])
I_1 = np.zeros([Nt,Nx])
I_2 = np.zeros([Nt,Nx])


def common(V_n,V_s,I_n,I_s,I_1,I_2,U,Jx,Nt,Nx):
    for m in range(Nt):
        for k in range(Nx-1):
            V_n[m,k] = U[m,k,50,51]-U[m,k,50,48]
            V_s[m,k] = 0.5*(U[m,k,50,51]+U[m,k,50,48])
        for k in range(Nx):
            I_1[m,k] = (Jx[m,k,50,51])*0.001*0.001
            I_2[m,k] = (Jx[m,k,50,48]+Jx[m,k,49,48]+Jx[m,k,51,48])*0.001*0.001
            I_n[m,k] = 0.5*(I_1[m,k]-I_2[m,k])
            I_s[m,k] = I_1[m,k]+I_2[m,k]
    return V_n,V_s,I_n,I_s

V_n_0,V_s_0,I_n_0,I_s_0 = common(V_n,V_s,I_n,I_s,I_1,I_2,U,Jx,Nt,Nx)

#電力の導出
def power(P_s_0,P_n_0):
  for i in range(Nt):
      for k in range(Nx-1):
          P_n_0[i,k]=V_n_0[i,k]*((I_n_0[i,k]+I_n_0[i,k+1])/2)
          P_s_0[i,k]=V_s_0[i,k]*((I_s_0[i,k]+I_s_0[i,k+1])/2)

#電力の最大値のプロット
end_n=200
end_s=180
P_n_max_0=np.zeros([Nx-1])
P_s_max_0=np.zeros([Nx-1])
for i in range(Nx-1):
    P_n_max_0[i]=P_n_0[0:end_n,i].max()
for i in range(Nx-1):
    P_s_max_0[i]=P_s_0[0:end_s,i].max()
