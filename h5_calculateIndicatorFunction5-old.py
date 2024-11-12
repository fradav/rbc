
## Coded by Francisco ROJAS-PEREZ
## This code calculates the indicator function in 3D (including PBC)
## assuming cartesian mesh and RBC shapes provided from h5py files with the geometry
#%%
import matplotlib.pyplot as plt
import numpy as np
import time

#!/usr/bin/python

import sys
from math import pi
import os

import h5py
import numpy as np
import time

from h5_rbc_treatement import *

#plt.style.use('_mpl-gallery-nogrid')


star=time.time()
print("   Calculation starting...")

# os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# os.environ['NUMBA_PARALLEL_DIAGNOSTICS'] = '4'
#list_dir=os.listdir(sys.argv[1])
list_dir=os.listdir("./")

list_surf_h5=[]
for name in list_dir:
    if (('surface0' in name) & ('.sol.h5' in name)):
        list_surf_h5.append("./"+name)
list_surf_h5=sorted(list_surf_h5)


print('list_surf_h5 =',list_surf_h5)
print()

plotting=1

print("\nGenerating indicator function")
## GENERATING THE DOMAIN AND THE INDICATOR FUNCTION (INITIALLY IN = -1)

####################################################################
########################### PARAMETERS #############################
####################################################################
x0=-50.0E-6
x1= 50.0E-6
y0=-78.0E-6
y1= 78.0E-6
z0=-50.0E-6
z1= 50.0E-6
Nx=200
Ny=312
Nz=200
Nl=4
xIndex=int(Nx*0.50)
yIndex=int(Ny*0.50)
zIndex=int(Nz*0.50)
global dx,dy,dz
dx=(x1-x0)/Nx
dy=(y1-y0)/Ny
dz=(z1-z0)/Nz
dxp=10.0E-6
dyp=10.0E-6
dzp=10.0E-6

print()
print("x0=",x0)
print("x1=",x1)
print("Nx=",Nx)
print("dxp=",dxp)
print("dx=",dx)
print("y0=",y0)
print("y1=",y1)
print("Ny=",Ny)
print("dyp=",dyp)
print("dy=",dy)
print("z0=",z0)
print("z1=",z1)
print("Nz=",Nz)
print("dzp=",dzp)
print("dz=",dz)
print()

#global xmin_ext #,Nx_ext,ymin_ext,ymax_ext,Ny_ext,zmin_ext,zmax_ext,Nz_ext
#global xmax_ext #,Nx_ext,ymin_ext,ymax_ext,Ny_ext,zmin_ext,zmax_ext,Nz_ext
#global xmin_ext,xmax_ext,Nx_ext,ymin_ext,ymax_ext,Ny_ext,zmin_ext,zmax_ext,Nz_ext

xmin_ext=x0-np.ceil(dxp/dx)*dx
xmax_ext=x1+np.ceil(dxp/dx)*dx-dx
Nx_ext=Nx+int(np.ceil(dxp/dx))*2
ymin_ext=y0-np.ceil(dyp/dy)*dy
ymax_ext=y1+np.ceil(dyp/dy)*dy-dy
Ny_ext=Ny+int(np.ceil(dyp/dy))*2
zmin_ext=z0-np.ceil(dzp/dz)*dz
zmax_ext=z1+np.ceil(dzp/dz)*dz-dz
Nz_ext=Nz+int(np.ceil(dzp/dz))*2
print("xmin_ext=",xmin_ext)
print("xmax_ext=",xmax_ext)
print("Nx_ext=",Nx_ext)
print("ymin_ext=",ymin_ext)
print("ymax_ext=",ymax_ext)
print("Ny_ext=",Ny_ext)
print("zmin_ext=",zmin_ext)
print("zmax_ext=",zmax_ext)
print("Nz_ext=",Nz_ext)
print()

#%%
#X, Y = np.meshgrid(np.linspace(x0,x1-dx,Nx), np.linspace(y0,y1-dy,Ny))
#X, Y, Z = np.meshgrid(np.linspace(x0,x1-dx,Nx), np.linspace(y0,y1-dy,Ny), np.linspace(z0,z1-dz,Nz))
#X, Y = np.meshgrid(np.linspace(xmin_ext,xmax_ext,Nx_ext), np.linspace(ymin_ext,ymax_ext,Ny_ext))
X, Y, Z   = np.meshgrid(np.linspace(xmin_ext,xmax_ext,Nx_ext), np.linspace(ymin_ext,ymax_ext,Ny_ext), np.linspace(zmin_ext,zmax_ext,Nz_ext))
#Gn = X-X
#Gn = Gn-1
# Initialization of G at -1
G  = X-X
Gb = (G>0)
G  = G-1

#%%
Id_point=924
dico_center_of_mass={}
cpt=0

# Get the directory of the xmf solutions
#dataDir  = './solution/'
dataDir  = "./"

D_list=[]
DI_list=[]
IEE_a=[]
IEE_b=[]
IEE_c=[] 
RBC_asph=[]
RBC_zAngl=[]
t_list=[]
vol_list=[]
com_list=[]
inertia_matrix={}
l_big=0.0
l_small=0.0
D=0.0


#%%

def apply_read_compute(filename):
    Gbb = np.zeros_like(X,dtype=np.bool_)
    with h5py.File(filename,'r') as f:
        for key in f:
            if (key[0:4]=="SURF"):
                path1=key+'/Coordinates/XYZ'
                path2=key+'/Connectivity/ELEM2NODE'
                coorArra=np.array(f[path1][:])
                connArra=np.array(f[path2][:])
                Gb = CalculateLocalIndicatorFunction(X,Y,Z,coorArra,connArra,dx,dy,dz,xmin_ext,xmax_ext,ymin_ext,ymax_ext,zmin_ext,zmax_ext,Nx_ext,Ny_ext,Nz_ext)
                Gbb = np.logical_xor(Gbb,Gb)
    return Gbb

#%%
if __name__ == '__main__':  # This avoids infinite subprocess creation
    from dask.distributed import Client
    import dask.bag as db
    client = Client(processes=True, threads_per_worker=1, n_workers=8, memory_limit='4GB')
    #%%

    Gb = db.from_sequence(list_surf_h5)\
            .map(apply_read_compute)\
            .fold(np.logical_xor,lambda *args: np.any(args,axis=0))\
            .compute()


    #%%
    # ####################################################################
    # ### MAIN LOOP: THE INDICATOR FUNCTION IS CALCULATED FOR EACH RBC ###
    # ####################################################################
    # for i in range(0,len(list_surf_h5)):
    #     f=h5py.File(list_surf_h5[i],'r')
    #     for key in f:
    #         if (key[0:4]=="SURF"):
    #             print()
    #             print(key)
    #             path1=key+'/Coordinates/XYZ'
    #             path2=key+'/Connectivity/ELEM2NODE'
    #             coorArra=np.array(f[path1][:])
    #             connArra=np.array(f[path2][:])
    #             print(xmin_ext)

    #             Gn=CalculateLocalIndicatorFunction(X,Y,Z,coorArra,connArra,dx,dy,dz,xmin_ext,xmax_ext,ymin_ext,ymax_ext,zmin_ext,zmax_ext,Nx_ext,Ny_ext,Nz_ext)
    #             Gvoxe = (Gn>0)
    #             Gb = Gb | Gvoxe
    #             G=Gn


    #%%
    ####################################################################
    ###### CROPPING THE DOMAIN TO TAKE INTO ACCOUNT PERIODICITIES ######
    ####################################################################

    ## reduction of the extended G function
    XX, YY, ZZ = np.meshgrid(np.linspace(x0,x1-dx,Nx), np.linspace(y0,y1-dy,Ny), np.linspace(z0,z1-dz,Nz))
    GG = XX*0
    GGb = (GG>0)
    Nxp=int(np.ceil(dxp/dx))
    Nyp=int(np.ceil(dyp/dy))
    Nzp=int(np.ceil(dzp/dz))
    #Nxp=0
    #Nyp=0
    #Nzp=0
    ## copy the central zone (unextended function)
    for i in range(int(0),int(Nx)):
        for j in range(int(0),int(Ny)):
            for k in range(int(0),int(Nz)):
                GGb[j,i,k] = Gb[j+Nyp,i+Nxp,k+Nzp]
    ## reflection of extended function on original function on:

    ## 1) right zone reflection (+x)
    for i in range(int(Nx+Nxp),int(Nx_ext)):
        for j in range(int(Nyp),int(Ny+Nyp)):
            for k in range(int(Nzp),int(Nz+Nzp)):
                if (Gb[j,i,k]==True):
                    if (GGb[j-Nyp,i-(Nx+Nxp),k-Nzp]==True):
                        print("Warning, superposition of shapes on reflection (right zone (+x))")
                    GGb[j-Nyp,i-(Nx+Nxp),k-Nzp] = Gb[j,i,k]
    ## 2) left zone reflection (-x)
    for i in range(int(0),int(Nxp)):
        for j in range(int(Nyp),int(Ny+Nyp)):
            for k in range(int(Nzp),int(Nz+Nzp)):
                if (Gb[j,i,k]==True):
                    if (GGb[j-Nyp,Nx-0-(Nxp-i),k-Nzp]==True):
                        print("Warning, superposition of shapes on reflection (left zone (-x))")
                    GGb[j-Nyp,Nx-0-(Nxp-i),k-Nzp] = Gb[j,i,k]
    ## 3) front zone reflection (+z)
    for i in range(int(Nxp),int(Nx+Nxp)):
        for j in range(int(Nyp),int(Ny+Nyp)):
            for k in range(int(Nz+Nzp),int(Nz_ext)):
                if (Gb[j,i,k]==True):
                    if (GGb[j-Nyp,i-Nxp,k-(Nz+Nzp)]==True):
                        print("Warning, superposition of shapes on reflection (front zone (+z))")
                    GGb[j-Nyp,i-Nxp,k-(Nz+Nzp)] = Gb[j,i,k]
    ## 4) back zone reflection (-z)
    for i in range(int(Nxp),int(Nx+Nxp)):
        for j in range(int(Nyp),int(Ny+Nyp)):
            for k in range(int(0),int(Nzp)):
                if (Gb[j,i,k]==True):
                    if (GGb[j-Nyp,i-Nxp,Nz-0-(Nzp-k)]==True):
                        print("Warning, superposition of shapes on reflection (back zone (-z))")
                    GGb[j-Nyp,i-Nxp,Nz-0-(Nzp-k)] = Gb[j,i,k]
    ## 5) right-front zone reflection (+x+z)
    for i in range(int(Nx+Nxp),int(Nx_ext)):
        for j in range(int(Nyp),int(Ny+Nyp)):
            for k in range(int(Nz+Nzp),int(Nz_ext)):
                if (Gb[j,i,k]==True):
                    if (GGb[j-Nyp,i-(Nx+Nxp),k-(Nz+Nzp)]==True):
                        print("Warning, superposition of shapes on reflection (right-front zone (+x+z))")
                    GGb[j-Nyp,i-(Nx+Nxp),k-(Nz+Nzp)] = Gb[j,i,k]
    ## 6) right-back zone reflection (+x-z)
    for i in range(int(Nx+Nxp),int(Nx_ext)):
        for j in range(int(Nyp),int(Ny+Nyp)):
            for k in range(int(0),int(Nzp)):
                if (Gb[j,i,k]==True):
                    if (GGb[j-Nyp,i-(Nx+Nxp),Nz-0-(Nzp-k)]==True):
                        print("Warning, superposition of shapes on reflection (right-back zone (+x-z))")
                    GGb[j-Nyp,i-(Nx+Nxp),Nz-0-(Nzp-k)] = Gb[j,i,k]
    ## 7) left-front zone reflection (-x+z)
    for i in range(int(0),int(Nxp)):
        for j in range(int(Nyp),int(Ny+Nyp)):
            for k in range(int(Nz+Nzp),int(Nz_ext)):
                if (Gb[j,i,k]==True):
                    if (GGb[j-Nyp,Nx-0-(Nxp-i),k-(Nz+Nzp)]==True):
                        print("Warning, superposition of shapes on reflection (left-front zone (-x+z))")
                    GGb[j-Nyp,Nx-0-(Nxp-i),k-(Nz+Nzp)] = Gb[j,i,k]
    ## 8) left-back zone reflection (-x-z)
    for i in range(int(0),int(Nxp)):
        for j in range(int(Nyp),int(Ny+Nyp)):
            for k in range(int(0),int(Nzp)):
                if (Gb[j,i,k]==True):
                    if (GGb[j-Nyp,Nx-0-(Nxp-i),Nz-0-(Nzp-k)]==True):
                        print("Warning, superposition of shapes on reflection (left-back zone (-x-z))")
                    GGb[j-Nyp,Nx-0-(Nxp-i),Nz-0-(Nzp-k)] = Gb[j,i,k]
    ## 9) full-top zone reflection (+y) (in principle forbidden given the slidding wall BC on +y_max, considered here only for warning)
    for i in range(int(0),int(Nx_ext)):
        for j in range(int(Ny+Nyp),int(Ny_ext)):
            for k in range(int(0),int(Nz_ext)):
                if (Gb[j,i,k]==True):
                    print("Warning, some shape exceeds the top zone (+y, where slidding wall BC is assumed)")
    ## 10) full-bottom zone reflection (-y) (in principle forbidden given the slidding wall BC on -y_max, considered here only for warning)
    for i in range(int(0),int(Nx_ext)):
        for j in range(int(0),int(Nyp)):
            for k in range(int(0),int(Nz_ext)):
                if (Gb[j,i,k]==True):
                    print("Warning, some shape exceeds the bottom zone (-y, where slidding wall BC is assumed)")


    np.save('xxx.dat',XX)
    np.save('yyy.dat',YY)
    np.save('zzz.dat',ZZ)
    #np.save('ggg.dat',GG)


    end=time.time()
    elap=end-star
    mess= "   Elapsed time: " + str(elap) + " seconds."
    print(mess)
    print("   Calculation finished!")


    print("plotting 3D voxels representation... ommited!")
    ### PLOT of GG[X,Y,Z] using voxels
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #ax.voxels(GGb, edgecolor='k')
    #ax.set(xticklabels=[],
    #       yticklabels=[],
    #       zticklabels=[])
    #plt.savefig("figureA.eps")
    #plt.show()

    ## conversion of GGb (booleans used for voxels) into GG (0's and 1's for plotting cut planes)
    for i in range(int(0),int(Nx)):
        for j in range(int(0),int(Ny)):
            for k in range(int(0),int(Nz)):
                if(GGb[j,i,k]==True):
                    GG[j,i,k]=1
                else:
                    GG[j,i,k]=0
    np.save('ggg.dat',GG)

    ####################################################################
    ######################## PLOTTING ##################################
    ####################################################################

    if (plotting==1):
        ## PLOT OF GG[Y,X,Z] using a cutting plane
        levels = np.linspace(GG.min(), GG.max(), Nl)
        pbcTestCutPlan = 0

        print()
        print("plotting 2D cut-plane representation (xz)...")
        print()
        ## x-z plane
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pcolormesh(XX[yIndex,:,:],ZZ[yIndex,:,:],GG[yIndex,:,:])
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title("Indicator function G (Nx=60, Ny=60, Nz=60)", fontsize=14)
        ax.set_xlim(x0,x1)
        ax.set_ylim(y0,y1) 
        if (pbcTestCutPlan==0):
            ax.set_xlim(x0,x1)
            ax.set_ylim(z0,z1)
        else:
            ax.set_xlim(xmin_ext,xmax_ext)
            ax.set_ylim(zmin_ext,zmax_ext)
            xp=np.linspace(xmin_ext,xmax_ext)
            yp=xp-xp+z0
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            yp=xp-xp+z1
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            yp=np.linspace(zmin_ext,zmax_ext)
            xp=yp-yp+x0
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            xp=yp-yp+x1
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            ax.text(16e-6,-15e-6,"dxp=8e-6")
            ax.text(15e-6,-22e-6,"dzp=8e-6",rotation=90)
        plt.subplots_adjust(bottom=0.09)
        plt.subplots_adjust(top=0.90)
        plt.subplots_adjust(left=0.11)
        plt.subplots_adjust(right=0.95)
        plt.savefig("figureB.eps")
        plt.show()

        print()
        print("plotting 2D cut-plane representation (xy)...")
        print()
        ## x-y plane
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pcolormesh(XX[:,:,zIndex],YY[:,:,zIndex],GG[:,:,zIndex])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Indicator function G (Nx=60, Ny=60, Nz=60)", fontsize=14)
        if (pbcTestCutPlan==0):
            ax.set_xlim(x0,x1)
            ax.set_ylim(y0,y1) 
        else:
            ax.set_xlim(xmin_ext,xmax_ext)
            ax.set_ylim(ymin_ext,ymax_ext)
            xp=np.linspace(xmin_ext,xmax_ext)
            yp=xp-xp+y0
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            yp=xp-xp+y1
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            yp=np.linspace(ymin_ext,ymax_ext)
            xp=yp-yp+x0
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            xp=yp-yp+x1
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            ax.text(16e-6,-15e-6,"dxp=8e-6")
            ax.text(15e-6,-22e-6,"dyp=8e-6",rotation=90)
        plt.subplots_adjust(bottom=0.09)
        plt.subplots_adjust(top=0.90)
        plt.subplots_adjust(left=0.11)
        plt.subplots_adjust(right=0.95)
        plt.savefig("figureC.eps")
        plt.show()

        print()
        print("plotting 2D cut-plane representation (yz)...")
        print()
        ## y-z plane
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pcolormesh(YY[:,xIndex,:],ZZ[:,xIndex,:],GG[:,xIndex,:])
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_title("Indicator function G (Nx=60, Ny=60, Nz=60)", fontsize=14)
        ax.set_xlim(x0,x1)
        ax.set_ylim(y0,y1) 
        if (pbcTestCutPlan==0):
            ax.set_xlim(y0,y1)
            ax.set_ylim(z0,z1)
        else:
            ax.set_xlim(ymin_ext,ymax_ext)
            ax.set_ylim(zmin_ext,zmax_ext)
            xp=np.linspace(ymin_ext,ymax_ext)
            yp=xp-xp+z0
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            yp=xp-xp+z1
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            yp=np.linspace(zmin_ext,zmax_ext)
            xp=yp-yp+y0
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            xp=yp-yp+y1
            plot1=ax.plot(xp,yp,'k--', linewidth=0.5)
            ax.text(16e-6,-15e-6,"dyp=8e-6")
            ax.text(15e-6,-22e-6,"dzp=8e-6",rotation=90)
        plt.subplots_adjust(bottom=0.09)
        plt.subplots_adjust(top=0.90)
        plt.subplots_adjust(left=0.11)
        plt.subplots_adjust(right=0.95)
        plt.savefig("figureD.eps")
        plt.show()


    end2=time.time()
    elap=end2-star
    mess= "   Elapsed time: " + str(elap) + " seconds."
    print(mess)
    print("   Calculation and plotting finished!")


    print()


    ####################################################################
    ####################################################################
                    


    # %%
