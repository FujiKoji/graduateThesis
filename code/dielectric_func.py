import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from numba import jit
import time
import japanize_matplotlib
import matplotlib.ticker as ptick

@jit(nopython=True)
def gausian(t,σ,b):
    result = 2*(1/((2*3.14)**0.5)*σ)*math.exp(-((t-b)**2)/(2*σ**2))/1.9950064981822565e-11
    return result

@jit(nopython=True)
def cal(Jx,Jz,U,Ax_0,Ax_1,Ax_2,Ay_0,Ay_1,Ay_2,Az_0,Az_1,Az_2,U_0,U_1,U_2,Jx_0,Jx_1,Jx_2,Jy_0,Jy_1,Jy_2,Jz_0,Jz_1,Jz_2,Xe,ρ,Nt,Nx,Ny,Nz,c,Δt,μ,ε,Δx,Δy,Δz,d1,d2,bx1,bx2,by1,by2,bz1,bz2,σ,b,doutai_x,doutai_y,doutai_z):
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
                                        
                    elif (30<k<280 and 20<=l<=40 and n==24) or (51<k<250 and l==30 and n==30) or (50<k<250 and l==26 and 29<=n<=31) or (50<k<250 and l==27 and 28<=n<=32) or (50<k<250 and l==28 and 27<=n<=29) or (50<k<250 and l==28 and 31<=n<=33) or (50<k<250 and l==29 and 26<=n<=28) or (50<k<250 and l==29 and 32<=n<=34) or (50<k<250 and l==30 and 26<=n<=27) or (50<k<250 and l==30 and 33<=n<=34) or (50<k<250 and l==31 and 26<=n<=28) or (50<k<250 and l==31 and 32<=n<=34) or (50<k<250 and l==32 and 27<=n<=29) or (50<k<250 and l==32 and 31<=n<=33) or (50<k<250 and l==33 and 28<=n<=32) or (50<k<250 and l==34 and 29<=n<=31):#導体
                        doutai_x[k,l,n]=1
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
                    elif k==51 and l==30 and n==30:
                        doutai_x[k,l,n]=10
                        Xe_ave = (Xe[k,l,n]+Xe[k-1,l,n])/2
                        Jx_0[51,30,30] = gausian(Δt*(m-1),σ,b)
                        Jx_1[51,30,30] = gausian(Δt*m,σ,b)
                        Jx_2[51,30,30] = gausian(Δt*(m+1),σ,b)
                        Ax_2[k,l,n] = (Δt**2)*(
                                        (Ax_1[k+1,l,n]-2*Ax_1[k,l,n]+Ax_1[k-1,l,n])/(Δx**2)
                                        +(Ax_1[k,l+1,n]-2*Ax_1[k,l,n]+Ax_1[k,l-1,n])/(Δy**2)
                                        +(Ax_1[k,l,n+1]-2*Ax_1[k,l,n]+Ax_1[k,l,n-1])/(Δz**2)
                                        -ε*μ*Xe_ave*(U_1[k,l,n]-U_0[k,l,n]-U_1[k-1,l,n]+U_0[k-1,l,n])/(Δx*Δt)
                                        +μ*(Jx_2[k,l,n]+Jx_1[k,l,n]+Jx_0[k,l,n])/3
                                        )/(ε*μ*(1+Xe_ave)) \
                                +2*Ax_1[k,l,n]-Ax_0[k,l,n]
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
                                        
                    elif (30<=k<280 and 21<=l<=40 and n==24) or (50<=k<250 and 30<=l<=31 and n==26) or(50<=k<250 and 29<=l<=32 and n==27) or (50<=k<250 and 28<=l<=29 and n==28) or (50<=k<250 and 32<=l<=33 and n==28) or (50<=k<250 and 27<=l<=28 and n==29) or (50<=k<250 and 33<=l<=34 and n==29) or (50<=k<250 and l==27 and n==30) or (50<=k<250 and l==34 and n==30) or (50<=k<250 and 27<=l<=28 and n==31) or (50<=k<250 and 33<=l<=34 and n==31) or (50<=k<250 and 28<=l<=29 and n==32) or (50<=k<250 and 32<=l<=33 and n==32) or (50<=k<250 and 29<=l<=32 and n==33) or (50<=k<250 and 30<=l<=31 and n==34):#導体
                        doutai_y[k,l,n]=1
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
                    
                    elif (50<=k<250 and l==26 and 30<=n<=31) or (50<=k<250 and l==27 and 29<=n<=32) or (50<=k<250 and l==28 and 28<=n<=29) or (50<=k<250 and l==28 and 32<=n<=33) or (50<=k<250 and l==29 and 27<=n<=28) or (50<=k<250 and l==29 and 33<=n<=34) or (50<=k<250 and l==30 and n==27) or (50<=k<250 and l==30 and n==34) or (50<=k<250 and l==31 and 27<=n<=28) or (50<=k<250 and l==31 and 33<=n<=34) or (50<=k<250 and l==32 and 28<=n<=29) or (50<=k<250 and l==32 and 32<=n<=33) or (50<=k<250 and l==33 and 29<=n<=32) or (50<=k<250 and l==34 and 30<=n<=31):
                        doutai_z[k,l,n]=1
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
        Jz[m+1,:,:,:] = Jz_2
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
    return Jx,Jz,U

#NMとCMの導出
@jit(nopython=True)
def origin(Nt,Nx,Ny,Nz,Jx_result,I_1,I_2,I_3,I_n,I_s,I_c,I_2_in,I_2_out):
    for m in range(Nt):
        for k in range(Nx):
            for l in range(Ny-1):
                for n in range(Nz-1):
                    if (50<k<250 and l==26 and 29<=n<=31) or (50<k<250 and l==27 and 28<=n<=32) or (50<k<250 and l==28 and 27<=n<=29) or (50<k<250 and l==28 and 31<=n<=33) or (50<k<250 and l==29 and 26<=n<=28) or (50<k<250 and l==29 and 32<=n<=34) or (50<k<250 and l==30 and 26<=n<=27) or (50<k<250 and l==30 and 33<=n<=34) or (50<k<250 and l==31 and 26<=n<=28) or (50<k<250 and l==31 and 32<=n<=34) or (50<k<250 and l==32 and 27<=n<=29) or (50<k<250 and l==32 and 31<=n<=33) or (50<k<250 and l==33 and 28<=n<=32) or (50<k<250 and l==34 and 29<=n<=31):#導体
                        I_2[m,k] += Jx_result[m,k,l,n]*0.001*0.001
                        if (50<k<250 and l==26 and 29<=n<=31) or (50<k<250 and l==27 and n==28) or (50<k<250 and l==27 and n==32) or (50<k<250 and l==28 and n==26) or (50<k<250 and l==28 and n==33) or (50<k<250 and 29<=l<=31 and n==26) or (50<k<250 and 29<=l<=31 and n==34) or (50<k<250 and l==32 and n==27) or (50<k<250 and l==32 and n==33) or (50<k<250 and l==33 and 28<=n<=32) or (50<k<250 and l==34 and 29<=n<=31):
                            I_2_out[m,k] += Jx_result[m,k,l,n]*0.001*0.001
                        elif (50<k<250 and l==27 and n==30) or (50<k<250 and l==28 and n==29) or (50<k<250 and l==28 and n==31) or (50<k<250 and l==29 and n==28) or (50<k<250 and l==29 and n==32) or (50<k<250 and l==30 and n==27) or (50<k<250 and l==30 and n==33) or (50<k<250 and l==31 and n==28) or (50<k<250 and l==31 and n==32) or (50<k<250 and l==32 and n==29) or (50<k<250 and l==32 and n==31) or (50<k<250 and l==33 and n==30):
                            I_2_in[m,k] += Jx_result[m,k,l,n]*0.001*0.001
                    elif 50<k<250 and l==30 and n==30:
                        I_1[m,k]=Jx_result[m,k,l,n]*0.001*0.001
                    elif (50<k<250 and 20<=l<=40 and n==24):
                        I_3[m,k]+=Jx_result[m,k,l,n]*0.001*0.001
    I_n = (I_1-I_2)*0.5
    I_s = I_1+I_2
    I_c = (I_1+I_2-I_3)*0.5
    return I_1,I_2,I_3,I_n,I_s,I_c
