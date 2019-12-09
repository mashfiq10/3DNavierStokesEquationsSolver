#!/usr/bin/python

#################################################
# 3D Navier-Stokes/Euler solver
# Sk. Mashfiqur Rahman
# Oklahoma State University
# CWID: A20102717
#################################################

import numpy as np
import time
import matplotlib.pyplot as plt


def initialize(nx,ny,nz,x,y,z,ma,alpha,denr,ukhi,gamma,ampy):
    if ipr == 1:  #khi
        ur = 1.
        q = np.zeros(shape=(nx+6,ny+6,nz+6,5),dtype='double')
        q[:,:,:,0] = (np.abs(z[:,:,:]) >= 0.25)*1. + (np.abs(z[:,:,:]) < 0.25)*denr
        q[:,:,:,1] = ((np.abs(z[:,:,:]) >= 0.25)*1.+(np.abs(z[:,:,:]) < 0.25)*denr)*((np.abs(z[:,:,:]) >= 0.25)*ukhi +
                     (np.abs(z[:,:,:]) < 0.25)*(-ukhi*ur))
        q[:,:,:,2] = ((np.abs(z[:,:,:]) >= 0.25)*1. + (np.abs(z[:,:,:]) < 0.25)*denr)*\
                     (ampy*np.sin(2.*np.pi*alpha*y[:,:,:]))
        q[:,:,:,3] = ((np.abs(z[:,:,:]) >= 0.25)*1. + (np.abs(z[:,:,:]) < 0.25)*denr)*\
                     (ampy*np.sin(2.*np.pi*alpha*x[:,:,:]))
        q[:,:,:,4] = ((np.abs(z[:,:,:]) >= 0.25)*1. + (np.abs(z[:,:,:]) < 0.25)*denr)*\
                     (2.5/(((np.abs(z[:,:,:]) >= 0.25)*1. + (np.abs(z[:,:,:]) < 0.25)*denr)*(gamma - 1.))
                      + 0.5*(np.square((np.abs(z[:,:,:]) >= 0.25)*ukhi + (np.abs(z[:,:,:]) < 0.25)*(-ukhi*ur)) +
                      np.square(ampy*np.sin(2.*np.pi*alpha*y[:,:,:]))+np.square(ampy*np.sin(2.*np.pi*alpha*x[:,:,:]))))

        ma = ukhi/np.sqrt(gamma*2.5/1.)
        mach = open("a_initial_mach.txt", "w+")
        mach.write('initial mach number (outer region) = ')
        mach.write(str(ma))
        mach.write('\n')
        mach.write('initial mach number (inner region) = ')
        mach.write(str(ukhi/np.sqrt(gamma*2.5/denr)))

    else:  # TGV
        tinit = 1.
        rinit = 1.
        vinit = 1.
        pinit = rinit*tinit/(gamma*ma*ma)
        q = np.zeros(shape=(nx+6,ny+6,nz+6,5),dtype='double')
        q[:,:,:,0] = rinit
        q[:,:,:,1] = rinit * (vinit*np.sin(x[:,:,:])*np.cos(y[:,:,:])*np.cos(z[:,:,:]))
        q[:,:,:,2] = rinit * (-vinit*np.cos(x[:,:,:])*np.sin(y[:,:,:])*np.cos(z[:,:,:]))
        q[:,:,:,3] = rinit * 0.
        q[:,:,:,4] = rinit*((pinit + rinit*(np.square(vinit))/(16.)*(np.cos(2.*x[:,:,:])+
                     np.cos(2.*y[:,:,:]))*(np.cos(2.*z[:,:,:])+2.)) / (rinit*(gamma - 1.)) +
                     0.5*(np.square(vinit*np.sin(x[:,:,:])*np.cos(y[:,:,:])*np.cos(z[:,:,:]))
                     + np.square(-vinit*np.cos(x[:,:,:])*np.sin(y[:,:,:])*np.cos(z[:,:,:])) + np.square(rinit*0.)))

    return q


def rhscs(nx,ny,nz,dx,dy,dz,_q,_s,gamma):
    g = 1./60.
    a = 37.
    b = -8.
    c = 1.
    gm = gamma - 1.

    u = np.zeros(shape=(nx+6,ny+6,nz+6),dtype='double')
    v = np.zeros(shape=(nx+6,ny+6,nz+6),dtype='double')
    w = np.zeros(shape=(nx+6,ny+6,nz+6),dtype='double')
    p = np.zeros(shape=(nx+6,ny+6,nz+6),dtype='double')
    vf = np.zeros(shape=(nx+3,ny+3,nz+3,5),dtype='double')
    vg = np.zeros(shape=(nx+3,ny+3,nz+3,5),dtype='double')
    vh = np.zeros(shape=(nx+3,ny+3,nz+3,5),dtype='double')

    u[:,:,:] = _q[:,:,:,1]/_q[:,:,:,0]
    v[:,:,:] = _q[:,:,:,2]/_q[:,:,:,0]
    w[:,:,:] = _q[:,:,:,3]/_q[:,:,:,0]
    p[:,:,:] = gm*(_q[:,:,:,4]- 0.5*(_q[:,:,:,1]*u[:,:,:] + _q[:,:,:,2]*v[:,:,:] + _q[:,:,:,3]*w[:,:,:]))

    vf[2:nx+3,2:ny+3,2:nz+3,0] = (g*(a*(_q[3:nx+4,2:ny+3,2:nz+3,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                 + b*(_q[4:nx+5,2:ny+3,2:nz+3,0] + _q[1:nx+2,2:ny+3,2:nz+3,0])
                                 + c*(_q[5:nx+6,2:ny+3,2:nz+3,0] + _q[0:nx+1,2:ny+3,2:nz+3,0])))* \
                                 (g*(a*(u[3:nx+4,2:ny+3,2:nz+3] + u[2:nx+3,2:ny+3,2:nz+3])+ b*(u[4:nx+5,2:ny+3,2:nz+3]
                                 + u[1:nx+2,2:ny+3,2:nz+3])+ c*(u[5:nx+6, 2:ny+3, 2:nz+3] + u[0:nx+1, 2:ny+3, 2:nz+3])))
    vf[2:nx+3,2:ny+3,2:nz+3,1] = ((g*(a*(_q[3:nx+4,2:ny+3,2:nz+3,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                 + b*(_q[4:nx+5,2:ny+3,2:nz+3,0] + _q[1:nx+2,2:ny+3,2:nz+3,0])
                                 + c*(_q[5:nx+6,2:ny+3,2:nz+3,0] + _q[0:nx+1,2:ny+3,2:nz+3,0])))* \
                                 np.square(g*(a*(u[3:nx+4,2:ny+3,2:nz+3] + u[2:nx+3,2:ny+3,2:nz+3])+
                                 b*(u[4:nx+5,2:ny+3,2:nz+3] + u[1:nx+2,2:ny+3,2:nz+3])+ c*(u[5:nx+6, 2:ny+3, 2:nz+3] +
                                 u[0:nx+1, 2:ny+3, 2:nz+3]))))+(g*(a*(p[3:nx+4,2:ny+3,2:nz+3]+p[2:nx+3,2:ny+3,2:nz+3])
                                 + b*(p[4:nx+5,2:ny+3,2:nz+3] + p[1:nx+2,2:ny+3,2:nz+3])+
                                 c*(p[5:nx+6,2:ny+3,2:nz+3] + p[0:nx+1,2:ny+3,2:nz+3])))
    vf[2:nx+3,2:ny+3,2:nz+3,2] = (g*(a*(_q[3:nx+4,2:ny+3,2:nz+3,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                 + b*(_q[4:nx+5,2:ny+3,2:nz+3,0] + _q[1:nx+2,2:ny+3,2:nz+3,0])
                                 + c*(_q[5:nx+6,2:ny+3,2:nz+3,0] + _q[0:nx+1,2:ny+3,2:nz+3,0])))* \
                                 (g*(a*(u[3:nx+4,2:ny+3,2:nz+3] + u[2:nx+3,2:ny+3,2:nz+3])+ b*(u[4:nx+5,2:ny+3,2:nz+3]
                                 +u[1:nx+2,2:ny+3,2:nz+3])+c*(u[5:nx+6, 2:ny+3, 2:nz+3]+u[0:nx+1, 2:ny+3, 2:nz+3])))* \
                                 (g*(a*(v[3:nx+4,2:ny+3,2:nz+3]+v[2:nx+3,2:ny+3,2:nz+3]) + b*(v[4:nx+5,2:ny+3,2:nz+3]
                                 + v[1:nx+2,2:ny+3,2:nz+3])+ c*(v[5:nx+6,2:ny+3,2:nz+3] + v[0:nx+1,2:ny+3,2:nz+3])))
    vf[2:nx+3,2:ny+3,2:nz+3,3] = (g*(a*(_q[3:nx+4,2:ny+3,2:nz+3,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                 + b*(_q[4:nx+5,2:ny+3,2:nz+3,0] + _q[1:nx+2,2:ny+3,2:nz+3,0])
                                 + c*(_q[5:nx+6,2:ny+3,2:nz+3,0] + _q[0:nx+1,2:ny+3,2:nz+3,0])))* \
                                 (g*(a*(u[3:nx+4,2:ny+3,2:nz+3] + u[2:nx+3,2:ny+3,2:nz+3])+ b*(u[4:nx+5,2:ny+3,2:nz+3]
                                 +u[1:nx+2,2:ny+3,2:nz+3])+c*(u[5:nx+6, 2:ny+3, 2:nz+3]+u[0:nx+1, 2:ny+3, 2:nz+3])))* \
                                 (g*(a*(w[3:nx+4,2:ny+3,2:nz+3]+w[2:nx+3,2:ny+3,2:nz+3]) + b*(w[4:nx+5,2:ny+3,2:nz+3]
                                 + w[1:nx+2,2:ny+3,2:nz+3])+ c*(w[5:nx+6,2:ny+3,2:nz+3] + w[0:nx+1,2:ny+3,2:nz+3])))
    vf[2:nx+3,2:ny+3,2:nz+3,4] = ((g*(a*(_q[3:nx+4,2:ny+3,2:nz+3,4] + _q[2:nx+3,2:ny+3,2:nz+3,4])
                                 + b*(_q[4:nx+5,2:ny+3,2:nz+3,4] + _q[1:nx+2,2:ny+3,2:nz+3,4])
                                 + c*(_q[5:nx+6,2:ny+3,2:nz+3,4] + _q[0:nx+1,2:ny+3,2:nz+3,4]))) +
                                 (g*(a*(p[3:nx+4,2:ny+3,2:nz+3] +p[2:nx+3,2:ny+3,2:nz+3]) + b*(p[4:nx+5,2:ny+3,2:nz+3]
                                 + p[1:nx+2,2:ny+3,2:nz+3])+c*(p[5:nx+6,2:ny+3,2:nz+3] + p[0:nx+1,2:ny+3,2:nz+3]))))* \
                                 (g*(a*(u[3:nx+4,2:ny+3,2:nz+3]+u[2:nx+3,2:ny+3,2:nz+3]) + b*(u[4:nx+5,2:ny+3,2:nz+3]
                                 + u[1:nx+2,2:ny+3,2:nz+3])+ c*(u[5:nx+6,2:ny+3,2:nz+3] + u[0:nx+1,2:ny+3,2:nz+3])))

    vg[2:nx+3,2:ny+3,2:nz+3,0] = (g*(a*(_q[2:nx+3,3:ny+4,2:nz+3,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                 + b*(_q[2:nx+3,4:ny+5,2:nz+3,0] + _q[2:nx+3,1:ny+2,2:nz+3,0])
                                 + c*(_q[2:nx+3,5:ny+6,2:nz+3,0] + _q[2:nx+3,0:ny+1,2:nz+3,0])))* \
                                 (g*(a*(v[2:nx+3,3:ny+4,2:nz+3]+v[2:nx+3,2:ny+3,2:nz+3]) + b*(v[2:nx+3,4:ny+5,2:nz+3]
                                 + v[2:nx+3,1:ny+2,2:nz+3])+ c*(v[2:nx+3,5:ny+6,2:nz+3] + v[2:nx+3,0:ny+1,2:nz+3])))
    vg[2:nx+3,2:ny+3,2:nz+3,1] = (g*(a*(_q[2:nx+3,3:ny+4,2:nz+3,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                 + b*(_q[2:nx+3,4:ny+5,2:nz+3,0] + _q[2:nx+3,1:ny+2,2:nz+3,0])
                                 + c*(_q[2:nx+3,5:ny+6,2:nz+3,0] + _q[2:nx+3,0:ny+1,2:nz+3,0])))* \
                                (g*(a*(u[2:nx+3,3:ny+4,2:nz+3]+ u[2:nx+3,2:ny+3,2:nz+3]) + b*(u[2:nx+3,4:ny+5,2:nz+3]
                                + u[2:nx+3,1:ny+2,2:nz+3])+ c*(u[2:nx+3,5:ny+6,2:nz+3] + u[2:nx+3,0:ny+1,2:nz+3])))* \
                                (g*(a*(v[2:nx+3,3:ny+4,2:nz+3] +v[2:nx+3,2:ny+3,2:nz+3]) + b*(v[2:nx+3,4:ny+5,2:nz+3]
                                + v[2:nx+3,1:ny+2,2:nz+3])+ c*(v[2:nx+3,5:ny+6,2:nz+3] + v[2:nx+3,0:ny+1,2:nz+3])))
    vg[2:nx+3,2:ny+3,2:nz+3,2] = ((g*(a*(_q[2:nx+3,3:ny+4,2:nz+3,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                + b*(_q[2:nx+3,4:ny+5,2:nz+3,0] + _q[2:nx+3,1:ny+2,2:nz+3,0])
                                + c*(_q[2:nx+3,5:ny+6,2:nz+3,0] + _q[2:nx+3,0:ny+1,2:nz+3,0])))*
                                np.square(g*(a*(v[2:nx+3,3:ny+4,2:nz+3]+v[2:nx+3,2:ny+3,2:nz+3]) +
                                b*(v[2:nx+3,4:ny+5,2:nz+3] + v[2:nx+3,1:ny+2,2:nz+3])+ c*(v[2:nx+3,5:ny+6,2:nz+3] +
                                v[2:nx+3,0:ny+1,2:nz+3])))) + (g*(a*(p[2:nx+3,3:ny+4,2:nz+3]+p[2:nx+3,2:ny+3,2:nz+3])
                                + b*(p[2:nx+3,4:ny+5,2:nz+3]+ p[2:nx+3,1:ny+2,2:nz+3])+c*(p[2:nx+3,5:ny+6,2:nz+3] +
                                p[2:nx+3,0:ny+1,2:nz+3])))
    vg[2:nx+3,2:ny+3,2:nz+3,3] = (g*(a*(_q[2:nx+3,3:ny+4,2:nz+3,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                + b*(_q[2:nx+3,4:ny+5,2:nz+3,0] + _q[2:nx+3,1:ny+2,2:nz+3,0])
                                + c*(_q[2:nx+3,5:ny+6,2:nz+3,0] + _q[2:nx+3,0:ny+1,2:nz+3,0])))* \
                                (g*(a*(v[2:nx+3,3:ny+4,2:nz+3] + v[2:nx+3,2:ny+3,2:nz+3]) + b*(v[2:nx+3,4:ny+5,2:nz+3]
                                + v[2:nx+3,1:ny+2,2:nz+3])+ c*(v[2:nx+3,5:ny+6,2:nz+3] + v[2:nx+3,0:ny+1,2:nz+3])))* \
                                (g*(a*(w[2:nx+3,3:ny+4,2:nz+3]+w[2:nx+3,2:ny+3,2:nz+3]) + b*(w[2:nx+3,4:ny+5,2:nz+3]
                                + w[2:nx+3,1:ny+2,2:nz+3])+ c*(w[2:nx+3,5:ny+6,2:nz+3] + w[2:nx+3,0:ny+1,2:nz+3])))
    vg[2:nx+3,2:ny+3,2:nz+3,4] = ((g*(a*(_q[2:nx+3,3:ny+4,2:nz+3,4] + _q[2:nx+3,2:ny+3,2:nz+3,4])
                                + b*(_q[2:nx+3,4:ny+5,2:nz+3,4] + _q[2:nx+3,1:ny+2,2:nz+3,4])
                                + c*(_q[2:nx+3,5:ny+6,2:nz+3,4] + _q[2:nx+3,0:ny+1,2:nz+3,4]))) +
                                (g*(a*(p[2:nx+3,3:ny+4,2:nz+3] + p[2:nx+3,2:ny+3,2:nz+3]) + b*(p[2:nx+3,4:ny+5,2:nz+3]
                                + p[2:nx+3,1:ny+2,2:nz+3])+c*(p[2:nx+3,5:ny+6,2:nz+3] + p[2:nx+3,0:ny+1,2:nz+3]))))* \
                                (g*(a*(v[2:nx+3,3:ny+4,2:nz+3]+v[2:nx+3,2:ny+3,2:nz+3]) + b*(v[2:nx+3,4:ny+5,2:nz+3]
                                + v[2:nx+3,1:ny+2,2:nz+3])+ c*(v[2:nx+3,5:ny+6,2:nz+3] + v[2:nx+3,0:ny+1,2:nz+3])))

    vh[2:nx+3,2:ny+3,2:nz+3,0] = (g*(a*(_q[2:nx+3,2:ny+3,3:nz+4,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                +b*(_q[2:nx+3,2:ny+3,4:nz+5,0]+_q[2:nx+3,2:ny+3,1:nz+2,0])+c*(_q[2:nx+3,2:ny+3,5:nz+6,0]
                                + _q[2:nx+3,2:ny+3,0:nz+1,0])))*(g*(a*(w[2:nx+3,2:ny+3,3:nz+4] +
                                w[2:nx+3,2:ny+3,2:nz+3]) + b*(w[2:nx+3,2:ny+3,4:nz+5] + w[2:nx+3,2:ny+3,1:nz+2])
                                + c*(w[2:nx+3,2:ny+3,5:nz+6] + w[2:nx+3,2:ny+3,0:nz+1])))
    vh[2:nx+3,2:ny+3,2:nz+3,1] = (g*(a*(_q[2:nx+3,2:ny+3,3:nz+4,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                + b*(_q[2:nx+3,2:ny+3,4:nz+5,0] + _q[2:nx+3,2:ny+3,1:nz+2,0])
                                + c*(_q[2:nx+3,2:ny+3,5:nz+6,0] + _q[2:nx+3,2:ny+3,0:nz+1,0])))* \
                                (g*(a*(u[2:nx+3,2:ny+3,3:nz+4]+u[2:nx+3,2:ny+3,2:nz+3]) + b*(u[2:nx+3,2:ny+3,4:nz+5]
                                + u[2:nx+3,2:ny+3,1:nz+2])+ c*(u[2:nx+3,2:ny+3,5:nz+6] + u[2:nx+3,2:ny+3,0:nz+1])))* \
                                (g*(a*(w[2:nx+3,2:ny+3,3:nz+4]+w[2:nx+3,2:ny+3,2:nz+3]) + b*(w[2:nx+3,2:ny+3,4:nz+5]
                                + w[2:nx+3,2:ny+3,1:nz+2])+ c*(w[2:nx+3,2:ny+3,5:nz+6] + w[2:nx+3,2:ny+3,0:nz+1])))
    vh[2:nx+3,2:ny+3,2:nz+3,2] = (g*(a*(_q[2:nx+3,2:ny+3,3:nz+4,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                + b*(_q[2:nx+3,2:ny+3,4:nz+5,0] + _q[2:nx+3,2:ny+3,1:nz+2,0])
                                + c*(_q[2:nx+3,2:ny+3,5:nz+6,0] + _q[2:nx+3,2:ny+3,0:nz+1,0])))* \
                                (g*(a*(v[2:nx+3,2:ny+3,3:nz+4]+v[2:nx+3,2:ny+3,2:nz+3]) + b*(v[2:nx+3,2:ny+3,4:nz+5]
                                + v[2:nx+3,2:ny+3,1:nz+2])+ c*(v[2:nx+3,2:ny+3,5:nz+6] + v[2:nx+3,2:ny+3,0:nz+1])))* \
                                (g*(a*(w[2:nx+3,2:ny+3,3:nz+4]+w[2:nx+3,2:ny+3,2:nz+3]) + b*(w[2:nx+3,2:ny+3,4:nz+5]
                                + w[2:nx+3,2:ny+3,1:nz+2])+ c*(w[2:nx+3,2:ny+3,5:nz+6] + w[2:nx+3,2:ny+3,0:nz+1])))
    vh[2:nx+3,2:ny+3,2:nz+3,3] = ((g*(a*(_q[2:nx+3,2:ny+3,3:nz+4,0] + _q[2:nx+3,2:ny+3,2:nz+3,0])
                                + b*(_q[2:nx+3,2:ny+3,4:nz+5,0] + _q[2:nx+3,2:ny+3,1:nz+2,0])
                                + c*(_q[2:nx+3,2:ny+3,5:nz+6,0] + _q[2:nx+3,2:ny+3,0:nz+1,0])))*
                                np.square(g*(a*(w[2:nx+3,2:ny+3,3:nz+4]+w[2:nx+3,2:ny+3,2:nz+3]) +
                                b*(w[2:nx+3,2:ny+3,4:nz+5] + w[2:nx+3,2:ny+3,1:nz+2])+c*(w[2:nx+3,2:ny+3,5:nz+6] +
                                w[2:nx+3,2:ny+3,0:nz+1])))) + (g*(a*(p[2:nx+3,2:ny+3,3:nz+4]+p[2:nx+3,2:ny+3,2:nz+3])
                                + b*(p[2:nx+3,2:ny+3,4:nz+5] + p[2:nx+3,2:ny+3,1:nz+2]) +
                                c*(p[2:nx+3,2:ny+3,5:nz+6] + p[2:nx+3,2:ny+3,0:nz+1])))
    vh[2:nx+3,2:ny+3,2:nz+3,4] = ((g*(a*(_q[2:nx+3,2:ny+3,3:nz+4,4] + _q[2:nx+3,2:ny+3,2:nz+3,4])
                                + b*(_q[2:nx+3,2:ny+3,4:nz+5,4] + _q[2:nx+3,2:ny+3,1:nz+2,4])
                                + c*(_q[2:nx+3,2:ny+3,5:nz+6,4] + _q[2:nx+3,2:ny+3,0:nz+1,4]))) +
                                (g*(a*(p[2:nx+3,2:ny+3,3:nz+4] +p[2:nx+3,2:ny+3,2:nz+3]) + b*(p[2:nx+3,2:ny+3,4:nz+5]
                                + p[2:nx+3,2:ny+3,1:nz+2])+c*(p[2:nx+3,2:ny+3,5:nz+6] + p[2:nx+3,2:ny+3,0:nz+1]))))* \
                                (g*(a*(w[2:nx+3,2:ny+3,3:nz+4]+w[2:nx+3,2:ny+3,2:nz+3]) + b*(w[2:nx+3,2:ny+3,4:nz+5]
                                + w[2:nx+3,2:ny+3,1:nz+2])+ c*(w[2:nx+3,2:ny+3,5:nz+6] + w[2:nx+3,2:ny+3,0:nz+1])))

    _s[3:nx+3,3:ny+3,3:nz+3,0:5] = -(vf[3:nx+3,3:ny+3,3:nz+3,0:5]-vf[2:nx+2,3:ny+3,3:nz+3,0:5])/dx - \
                                   (vg[3:nx+3,3:ny+3,3:nz+3,0:5] - vg[3:nx+3,2:ny+2,3:nz+3,0:5])/dy - \
                                   (vh[3:nx+3,3:ny+3,3:nz+3,0:5]-vh[3:nx+3,3:ny+3,2:nz+2,0:5])/dz

    del u,v,w,p,vf,vg,vh

    return _s


def rhsvis(nx,ny,nz,dx,dy,dz,_q,_s,gamma,Tref,re,pr,ma,imu):
    g = 1./180.
    a = 245.
    b = -75.
    c = 10.

    dx1 = 1./dx
    dx3 = 1./(3.*dx)
    dx5 = 1./(5.*dx)

    dy1 = 1./dy
    dy3 = 1./(3.*dy)
    dy5 = 1./(5.*dy)

    dz1 = 1./dz
    dz3 = 1./(3.*dz)
    dz5 = 1./(5.*dz)

    # for interpolation (finite volume)

    gi = 1./60.
    ai = 37.
    bi = -8.
    ci = 1.

    # for cross derivatives

    gd = 1./10.
    ad = 15.
    bd = -6.
    cd = 1.

    dx2 = 1./(2.*dx)
    dx4 = 1./(4.*dx)
    dx6 = 1./(6.*dx)

    dy2 = 1./(2.*dy)
    dy4 = 1./(4.*dy)
    dy6 = 1./(6.*dy)

    dz2 = 1./(2.*dz)
    dz4 = 1./(4.*dz)
    dz6 = 1./(6.*dz)

    t = np.zeros(shape=(nx+6,ny+6,nz+6),dtype='double')
    mu = np.zeros(shape=(nx+6,ny+6,nz+6),dtype='double')
    vf = np.zeros(shape=(nx+3,ny+3,nz+3,5),dtype='double')
    vg = np.zeros(shape=(nx+3,ny+3,nz+3,5),dtype='double')
    vh = np.zeros(shape=(nx+3,ny+3,nz+3,5),dtype='double')
    cd1 = np.zeros(shape=(nx+6,ny,nz),dtype='double')
    cd2 = np.zeros(shape=(nx+6,ny,nz),dtype='double')
    cd3 = np.zeros(shape=(nx+6,ny,nz),dtype='double')
    cd4 = np.zeros(shape=(nx+6,ny,nz),dtype='double') # cd = 0:nx+6, 0:ny, 0:nz

    #sutherland's law
    if imu == 1:
        g1 = 1./re
        g2 = (2./3.)/re
        g3 = -1./(re*pr*ma*ma*(gamma-1.))
        g4 = (gamma - 1.)*gamma*ma*ma

        cc = 110.4/Tref
        t[:,:,:] = g4*(_q[:,:,:,4]/_q[:,:,:,0] - 0.5*(np.square(_q[:,:,:,1]/_q[:,:,:,0]) +
                   np.square(_q[:,:,:,2]/_q[:,:,:,0])+ np.square(_q[:,:,:,3]/_q[:,:,:,0]) ) )
        mu[:,:,:] = ((t[:,:,:])**(1.5))*(1.+ cc)/(t[:,:,:] + cc)
    else:
        g1 = 1./re
        g2 = (2./3.)/re
        g3 = -gamma/(re*pr)

        t[:,:,:] = _q[:,:,:,4]/_q[:,:,:,0] - 0.5*(np.square(_q[:,:,:,1]/_q[:,:,:,0]) +
                   np.square(_q[:,:,:,2]/_q[:,:,:,0])+ np.square(_q[:,:,:,3]/_q[:,:,:,0]))
        mu[:,:,:] = 1.

    # compute viscous fluxes

    #  x-direction
    #  uy
    cd1[:,:,:] = gd*(ad*dy2*(_q[:,4:ny+4,3:nz+3,1]/_q[:,4:ny+4,3:nz+3,0]-_q[:,2:ny+2,3:nz+3,1]/_q[:,2:ny+2,3:nz+3,0]) +
                 bd*dy4*(_q[:,5:ny+5,3:nz+3,1]/_q[:,5:ny+5,3:nz+3,0] - _q[:,1:ny+1,3:nz+3,1]/_q[:,1:ny+1,3:nz+3,0]) +
                 cd*dy6*(_q[:,6:ny+6,3:nz+3,1]/_q[:,6:ny+6,3:nz+3,0] - _q[:,0:ny,3:nz+3,1]/_q[:,0:ny,3:nz+3,0]))
    #  uz
    cd2[:,:,:] = gd*(ad*dz2*(_q[:,3:ny+3,4:nz+4,1]/_q[:,3:ny+3,4:nz+4,0]-_q[:,3:ny+3,2:nz+2,1]/_q[:,3:ny+3,2:nz+2,0]) +
                 bd*dz4*(_q[:,3:ny+3,5:nz+5,1]/_q[:,3:ny+3,5:nz+5,0] -_q[:,3:ny+3,1:nz+1,1]/_q[:,3:ny+3,1:nz+1,0]) +
                 cd*dz6*(_q[:,3:ny+3,6:nz+6,1]/_q[:,3:ny+3,6:nz+6,0] -_q[:,3:ny+3,0:nz,1]/_q[:,3:ny+3,0:nz,0]))

    #  vy
    cd3[:,:,:] = gd*(ad*dy2*(_q[:,4:ny+4,3:nz+3,2]/_q[:,4:ny+4,3:nz+3,0] -_q[:,2:ny+2,3:nz+3,2]/_q[:,2:ny+2,3:nz+3,0]) +
                 bd*dy4*(_q[:,5:ny+5,3:nz+3,2]/_q[:,5:ny+5,3:nz+3,0] -_q[:,1:ny+1,3:nz+3,2]/_q[:,1:ny+1,3:nz+3,0]) +
                 cd*dy6*(_q[:,6:ny+6,3:nz+3,2]/_q[:,6:ny+6,3:nz+3,0] -_q[:,0:ny,3:nz+3,2]/_q[:,0:ny,3:nz+3,0]))

    #  wz
    cd4[:,:,:] = gd*(ad*dz2*(_q[:,3:ny+3,4:nz+4,3]/_q[:,3:ny+3,4:nz+4,0] -_q[:,3:ny+3,2:nz+2,3]/_q[:,3:ny+3,2:nz+2,0]) +
                 bd*dz4*(_q[:,3:ny+3,5:nz+5,3]/_q[:,3:ny+3,5:nz+5,0] -_q[:,3:ny+3,1:nz+1,3]/_q[:,3:ny+3,1:nz+1,0]) +
                 cd*dz6*(_q[:,3:ny+3,6:nz+6,3]/_q[:,3:ny+3,6:nz+6,0] -_q[:,3:ny+3,0:nz,3]/_q[:,3:ny+3,0:nz,0]))

    vf[2:nx+3,3:ny+3,3:nz+3,0] = 0.
    vf[2:nx+3,3:ny+3,3:nz+3,1] = g2*(gi*(ai*(mu[3:nx+4,3:ny+3,3:nz+3] + mu[2:nx+3,3:ny+3,3:nz+3]) +
                                 bi*(mu[4:nx+5,3:ny+3,3:nz+3] + mu[1:nx+2,3:ny+3,3:nz+3]) +
                                 ci*(mu[5:nx+6,3:ny+3,3:nz+3] + mu[0:nx+1,3:ny+3,3:nz+3])))*\
                                 (2.*(g*(a*dx1*(_q[3:nx+4,3:ny+3,3:nz+3,1]/
                                 _q[3:nx+4,3:ny+3,3:nz+3,0] - _q[2:nx+3,3:ny+3,3:nz+3,1]/_q[2:nx+3,3:ny+3,3:nz+3,0]) +
                                 b*dx3*(_q[4:nx+5,3:ny+3,3:nz+3,1]/_q[4:nx+5,3:ny+3,3:nz+3,0] -
                                 _q[1:nx+2,3:ny+3,3:nz+3,1]/_q[1:nx+2,3:ny+3,3:nz+3,0]) + c*dx5*
                                 (_q[5:nx+6,3:ny+3,3:nz+3,1] /_q[5:nx+6,3:ny+3,3:nz+3,0] - _q[0:nx+1,3:ny+3,3:nz+3,1]/
                                 _q[0:nx+1,3:ny+3,3:nz+3,0])))-(gi*(ai*(cd3[3:nx+4,:,:] + cd3[2:nx+3,:,:])+
                                 bi*(cd3[4:nx+5,:,:] + cd3[1:nx+2,:,:]) + ci*(cd3[5:nx+6,:,:] +
                                 cd3[0:nx+1,:,:]))) - (gi*(ai*(cd4[3:nx+4,:,:] + cd4[2:nx+3,:,:]) +
                                 bi*(cd4[4:nx+5,:,:] + cd4[1:nx+2,:,:]) + ci*(cd4[5:nx+6,:,:] +
                                 cd4[0:nx+1,:,:]))))
    vf[2:nx+3,3:ny+3,3:nz+3,2] = g1*(gi*(ai*(mu[3:nx+4,3:ny+3,3:nz+3] + mu[2:nx+3,3:ny+3,3:nz+3]) +
                                 bi*(mu[4:nx+5,3:ny+3,3:nz+3] + mu[1:nx+2,3:ny+3,3:nz+3]) +
                                 ci*(mu[5:nx+6,3:ny+3,3:nz+3] + mu[0:nx+1,3:ny+3,3:nz+3])))*((gi*
                                 (ai*(cd1[3:nx+4,:,:] + cd1[2:nx+3,:,:])+bi*(cd1[4:nx+5,:,:] +
                                 cd1[1:nx+2,:,:]) + ci*(cd1[5:nx+6,:,:] + cd1[0:nx+1,:,:])))
                                 + (g*(a*dx1*(_q[3:nx+4,3:ny+3,3:nz+3,2]/_q[3:nx+4,3:ny+3,3:nz+3,0] -
                                 _q[2:nx+3,3:ny+3,3:nz+3,2]/_q[2:nx+3,3:ny+3,3:nz+3,0])
                                  + b*dx3*(_q[4:nx+5,3:ny+3,3:nz+3,2]/_q[4:nx+5,3:ny+3,3:nz+3,0] -
                                _q[1:nx+2,3:ny+3,3:nz+3,2]/_q[1:nx+2,3:ny+3,3:nz+3,0])
                                  + c*dx5*(_q[5:nx+6,3:ny+3,3:nz+3,2]/_q[5:nx+6,3:ny+3,3:nz+3,0] -
                                _q[0:nx+1,3:ny+3,3:nz+3,2]/_q[0:nx+1,3:ny+3,3:nz+3,0]))))
    vf[2:nx+3,3:ny+3,3:nz+3,3] = g1*(gi*(ai*(mu[3:nx+4,3:ny+3,3:nz+3] + mu[2:nx+3,3:ny+3,3:nz+3]) +
                                 bi*(mu[4:nx+5,3:ny+3,3:nz+3] + mu[1:nx+2,3:ny+3,3:nz+3]) +
                                 ci*(mu[5:nx+6,3:ny+3,3:nz+3] + mu[0:nx+1,3:ny+3,3:nz+3])))*\
                                 ((gi*(ai*(cd2[3:nx+4,:,:] + cd2[2:nx+3,:,:]) +
                                 bi*(cd2[4:nx+5,:,:] + cd2[1:nx+2,:,:]) + ci*(cd2[5:nx+6,:,:] +
                                 cd2[0:nx+1,:,:])))+(g*(a*dx1*(_q[3:nx+4,3:ny+3,3:nz+3,3]/_q[3:nx+4,3:ny+3,3:nz+3,0]
                                 - _q[2:nx+3,3:ny+3,3:nz+3,3]/_q[2:nx+3,3:ny+3,3:nz+3,0])
                                  + b*dx3*(_q[4:nx+5,3:ny+3,3:nz+3,3]/_q[4:nx+5,3:ny+3,3:nz+3,0] -
                                _q[1:nx+2,3:ny+3,3:nz+3,3]/_q[1:nx+2,3:ny+3,3:nz+3,0])+c*dx5*(_q[5:nx+6,3:ny+3,3:nz+3,3]
                                /_q[5:nx+6,3:ny+3,3:nz+3,0] - _q[0:nx+1,3:ny+3,3:nz+3,3]/_q[0:nx+1,3:ny+3,3:nz+3,0]))))
    vf[2:nx+3,3:ny+3,3:nz+3,4] = (gi*(ai*(_q[3:nx+4,3:ny+3,3:nz+3,1]/_q[3:nx+4,3:ny+3,3:nz+3,0] +
                                 _q[2:nx+3,3:ny+3,3:nz+3,1]/_q[2:nx+3,3:ny+3,3:nz+3,0])+ bi*(_q[4:nx+5,3:ny+3,3:nz+3,1]/
                                 _q[4:nx+5,3:ny+3,3:nz+3,0] + _q[1:nx+2,3:ny+3,3:nz+3,1]/_q[1:nx+2,3:ny+3,3:nz+3,0])
                                 + ci*(_q[5:nx+6,3:ny+3,3:nz+3,1]/_q[5:nx+6,3:ny+3,3:nz+3,0]+_q[0:nx+1,3:ny+3,3:nz+3,1]
                                 /_q[0:nx+1,3:ny+3,3:nz+3,0])))*(g2*(gi*(ai*(mu[3:nx+4,3:ny+3,3:nz+3] +
                                 mu[2:nx+3,3:ny+3,3:nz+3]) +bi*(mu[4:nx+5,3:ny+3,3:nz+3] + mu[1:nx+2,3:ny+3,3:nz+3]) +
                                 ci*(mu[5:nx+6,3:ny+3,3:nz+3] + mu[0:nx+1,3:ny+3,3:nz+3])))*\
                                 (2.*(g*(a*dx1*(_q[3:nx+4,3:ny+3,3:nz+3,1]/
                                 _q[3:nx+4,3:ny+3,3:nz+3,0] - _q[2:nx+3,3:ny+3,3:nz+3,1]/_q[2:nx+3,3:ny+3,3:nz+3,0]) +
                                 b*dx3*(_q[4:nx+5,3:ny+3,3:nz+3,1]/_q[4:nx+5,3:ny+3,3:nz+3,0] -
                                 _q[1:nx+2,3:ny+3,3:nz+3,1]/_q[1:nx+2,3:ny+3,3:nz+3,0]) + c*dx5*
                                 (_q[5:nx+6,3:ny+3,3:nz+3,1] /_q[5:nx+6,3:ny+3,3:nz+3,0] - _q[0:nx+1,3:ny+3,3:nz+3,1]/
                                 _q[0:nx+1,3:ny+3,3:nz+3,0])))-(gi*(ai*(cd3[3:nx+4,:,:] + cd3[2:nx+3,:,:])+
                                 bi*(cd3[4:nx+5,:,:] + cd3[1:nx+2,:,:]) + ci*(cd3[5:nx+6,:,:] +
                                 cd3[0:nx+1,:,:]))) - (gi*(ai*(cd4[3:nx+4,:,:] + cd4[2:nx+3,:,:]) +
                                 bi*(cd4[4:nx+5,:,:] + cd4[1:nx+2,:,:]) + ci*(cd4[5:nx+6,:,:] +
                                 cd4[0:nx+1,:,:])))))+(gi*(ai*(_q[3:nx+4,3:ny+3,3:nz+3,2]/
                                 _q[3:nx+4,3:ny+3,3:nz+3,0] + _q[2:nx+3,3:ny+3,3:nz+3,2]/_q[2:nx+3,3:ny+3,3:nz+3,0])
                                 + bi*(_q[4:nx+5,3:ny+3,3:nz+3,2]/_q[4:nx+5,3:ny+3,3:nz+3,0]+_q[1:nx+2,3:ny+3,3:nz+3,2]/
                                _q[1:nx+2,3:ny+3,3:nz+3,0])+ci*(_q[5:nx+6,3:ny+3,3:nz+3,2]/_q[5:nx+6,3:ny+3,3:nz+3,0] +
                                _q[0:nx+1,3:ny+3,3:nz+3,2]/_q[0:nx+1,3:ny+3,3:nz+3,0])))*(g1*(gi*(ai*
                                (mu[3:nx+4,3:ny+3,3:nz+3] + mu[2:nx+3,3:ny+3,3:nz+3]) +
                                 bi*(mu[4:nx+5,3:ny+3,3:nz+3] + mu[1:nx+2,3:ny+3,3:nz+3]) +
                                 ci*(mu[5:nx+6,3:ny+3,3:nz+3] + mu[0:nx+1,3:ny+3,3:nz+3])))*((gi*
                                 (ai*(cd1[3:nx+4,:,:] + cd1[2:nx+3,:,:])+bi*(cd1[4:nx+5,:,:] +
                                 cd1[1:nx+2,:,:]) + ci*(cd1[5:nx+6,:,:] + cd1[0:nx+1,:,:])))
                                 + (g*(a*dx1*(_q[3:nx+4,3:ny+3,3:nz+3,2]/_q[3:nx+4,3:ny+3,3:nz+3,0] -
                                 _q[2:nx+3,3:ny+3,3:nz+3,2]/_q[2:nx+3,3:ny+3,3:nz+3,0])
                                  + b*dx3*(_q[4:nx+5,3:ny+3,3:nz+3,2]/_q[4:nx+5,3:ny+3,3:nz+3,0] -
                                _q[1:nx+2,3:ny+3,3:nz+3,2]/_q[1:nx+2,3:ny+3,3:nz+3,0])
                                  + c*dx5*(_q[5:nx+6,3:ny+3,3:nz+3,2]/_q[5:nx+6,3:ny+3,3:nz+3,0] -
                                _q[0:nx+1,3:ny+3,3:nz+3,2]/_q[0:nx+1,3:ny+3,3:nz+3,0])))))+(gi*(ai*
                                (_q[3:nx+4,3:ny+3,3:nz+3,3]/_q[3:nx+4,3:ny+3,3:nz+3,0] + _q[2:nx+3,3:ny+3,3:nz+3,3]/
                                _q[2:nx+3,3:ny+3,3:nz+3,0])+ bi*(_q[4:nx+5,3:ny+3,3:nz+3,3]/_q[4:nx+5,3:ny+3,3:nz+3,0]+
                                 _q[1:nx+2,3:ny+3,3:nz+3,3]/_q[1:nx+2,3:ny+3,3:nz+3,0])+ ci*(_q[5:nx+6,3:ny+3,3:nz+3,3]/
                                _q[5:nx+6,3:ny+3,3:nz+3,0]+ _q[0:nx+1,3:ny+3,3:nz+3,3]/_q[0:nx+1,3:ny+3,3:nz+3,0])))*\
                                (g1*(gi*(ai*(mu[3:nx+4,3:ny+3,3:nz+3] + mu[2:nx+3,3:ny+3,3:nz+3]) +
                                 bi*(mu[4:nx+5,3:ny+3,3:nz+3] + mu[1:nx+2,3:ny+3,3:nz+3]) +
                                 ci*(mu[5:nx+6,3:ny+3,3:nz+3] + mu[0:nx+1,3:ny+3,3:nz+3])))*\
                                 ((gi*(ai*(cd2[3:nx+4,:,:] + cd2[2:nx+3,:,:]) +
                                 bi*(cd2[4:nx+5,:,:] + cd2[1:nx+2,:,:]) + ci*(cd2[5:nx+6,:,:] +
                                 cd2[0:nx+1,:,:])))+(g*(a*dx1*(_q[3:nx+4,3:ny+3,3:nz+3,3]/_q[3:nx+4,3:ny+3,3:nz+3,0]
                                 - _q[2:nx+3,3:ny+3,3:nz+3,3]/_q[2:nx+3,3:ny+3,3:nz+3,0])
                                  + b*dx3*(_q[4:nx+5,3:ny+3,3:nz+3,3]/_q[4:nx+5,3:ny+3,3:nz+3,0] -
                                _q[1:nx+2,3:ny+3,3:nz+3,3]/_q[1:nx+2,3:ny+3,3:nz+3,0])+c*dx5*(_q[5:nx+6,3:ny+3,3:nz+3,3]
                                /_q[5:nx+6,3:ny+3,3:nz+3,0] -_q[0:nx+1,3:ny+3,3:nz+3,3]/_q[0:nx+1,3:ny+3,3:nz+3,0])))))\
                                 - g3*(g*(a*dx1*(t[3:nx+4,3:ny+3,3:nz+3] -t[2:nx+3,3:ny+3,3:nz+3])+ b*dx3*
                                (t[4:nx+5,3:ny+3,3:nz+3] -t[1:nx+2,3:ny+3,3:nz+3])+ c*dx5*(t[5:nx+6,3:ny+3,3:nz+3] -
                                t[0:nx+1,3:ny+3,3:nz+3])))*(gi*(ai*(mu[3:nx+4,3:ny+3,3:nz+3]+mu[2:nx+3,3:ny+3,3:nz+3]) +
                                 bi*(mu[4:nx+5,3:ny+3,3:nz+3] + mu[1:nx+2,3:ny+3,3:nz+3]) +
                                 ci*(mu[5:nx+6,3:ny+3,3:nz+3] + mu[0:nx+1,3:ny+3,3:nz+3])))

    _s[3:nx+3,3:ny+3,3:nz+3,0:5] = _s[3:nx+3,3:ny+3,3:nz+3,0:5] + (vf[3:nx+3,3:ny+3,3:nz+3,0:5]
                                   - vf[2:nx+2,3:ny+3,3:nz+3,0:5])/dx
    del cd1,cd2,cd3,cd4

    cd1 = np.zeros(shape=(nx,ny+6,nz),dtype='double')
    cd2 = np.zeros(shape=(nx,ny+6,nz),dtype='double')
    cd3 = np.zeros(shape=(nx,ny+6,nz),dtype='double')
    cd4 = np.zeros(shape=(nx,ny+6,nz),dtype='double') # cd = 0:nx, 0:ny+6, 0:nz

    #  y-direction
    #  vx
    cd1[:,:,:] = gd*(ad*dx2*(_q[4:nx+4,:,3:nz+3,2]/_q[4:nx+4,:,3:nz+3,0] - _q[2:nx+2,:,3:nz+3,2]/_q[2:nx+2,:,3:nz+3,0])+
                 bd*dx4*(_q[5:nx+5,:,3:nz+3,2]/_q[5:nx+5,:,3:nz+3,0] - _q[1:nx+1,:,3:nz+3,2]/_q[1:nx+1,:,3:nz+3,0]) +
                 cd*dx6*(_q[6:nx+6,:,3:nz+3,2]/_q[6:nx+6,:,3:nz+3,0] -_q[0:nx,:,3:nz+3,2]/_q[0:nx,:,3:nz+3,0]))
    #  vz
    cd2[:,:,:] = gd*(ad*dz2*(_q[3:nx+3,:,4:nz+4,2]/_q[3:nx+3,:,4:nz+4,0] - _q[3:nx+3,:,2:nz+2,2]/_q[3:nx+3,:,2:nz+2,0])
                + bd*dz4*(_q[3:nx+3,:,5:nz+5,2]/_q[3:nx+3,:,5:nz+5,0] -_q[3:nx+3,:,1:nz+1,2]/_q[3:nx+3,:,1:nz+1,0]) +
                cd*dz6*(_q[3:nx+3,:,6:nz+6,2]/_q[3:nx+3,:,6:nz+6,0] -_q[3:nx+3,:,0:nz,2]/_q[3:nx+3,:,0:nz,0]))

    #  ux
    cd3[:,:,:] = gd*(ad*dx2*(_q[4:nx+4,:,3:nz+3,1]/_q[4:nx+4,:,3:nz+3,0] -_q[2:nx+2,:,3:nz+3,1]/_q[2:nx+2,:,3:nz+3,0])+
                bd*dx4*(_q[5:nx+5,:,3:nz+3,1] /_q[5:nx+5,:,3:nz+3,0] -_q[1:nx+1,:,3:nz+3,1]/_q[1:nx+1,:,3:nz+3,0]) +
                cd*dx6*(_q[6:nx+6,:,3:nz+3,1]/_q[6:nx+6,:,3:nz+3,0] -_q[0:nx,:,3:nz+3,1]/_q[0:nx,:,3:nz+3,0]))

    #  wz
    cd4[:,:,:] = gd*(ad*dz2*(_q[3:nx+3,:,4:nz+4,3]/_q[3:nx+3,:,4:nz+4,0] -_q[3:nx+3,:,2:nz+2,3]/_q[3:nx+3,:,2:nz+2,0])
                + bd*dz4*(_q[3:nx+3,:,5:nz+5,3]/_q[3:nx+3,:,5:nz+5,0] -_q[3:nx+3,:,1:nz+1,3]/_q[3:nx+3,:,1:nz+1,0]) +
                cd*dz6*(_q[3:nx+3,:,6:nz+6,3]/_q[3:nx+3,:,6:nz+6,0] -_q[3:nx+3,:,0:nz,3]/_q[3:nx+3,:,0:nz,0]))

    vg[3:nx+3,2:ny+3,3:nz+3,0] = 0.
    vg[3:nx+3,2:ny+3,3:nz+3,1] = g1*(gi*(ai*(mu[3:nx+3,3:ny+4,3:nz+3] + mu[3:nx+3,2:ny+3,3:nz+3]) +
                                 bi*(mu[3:nx+3,4:ny+5,3:nz+3] + mu[3:nx+3,1:ny+2,3:nz+3]) +
                                 ci*(mu[3:nx+3,5:ny+6,3:nz+3] + mu[3:nx+3,0:ny+1,3:nz+3])))*((gi*
                                 (ai*(cd1[:,3:ny+4,:] + cd1[:,2:ny+3,:])+bi*(cd1[:,4:ny+5,:] +
                                 cd1[:,1:ny+2,:]) + ci*(cd1[:,5:ny+6,:] + cd1[:,0:ny+1,:])))
                                 + (g*(a*dy1*(_q[3:nx+3,3:ny+4,3:nz+3,1]/_q[3:nx+3,3:ny+4,3:nz+3,0] -
                                 _q[3:nx+3,2:ny+3,3:nz+3,1]/_q[3:nx+3,2:ny+3,3:nz+3,0])
                                  + b*dy3*(_q[3:nx+3,4:ny+5,3:nz+3,1]/_q[3:nx+3,4:ny+5,3:nz+3,0] -
                                _q[3:nx+3,1:ny+2,3:nz+3,1]/_q[3:nx+3,1:ny+2,3:nz+3,0])
                                  + c*dy5*(_q[3:nx+3,5:ny+6,3:nz+3,1]/_q[3:nx+3,5:ny+6,3:nz+3,0] -
                                _q[3:nx+3,0:ny+1,3:nz+3,1]/_q[3:nx+3,0:ny+1,3:nz+3,0]))))
    vg[3:nx+3,2:ny+3,3:nz+3,2] = g2*(gi*(ai*(mu[3:nx+3,3:ny+4,3:nz+3] + mu[3:nx+3,2:ny+3,3:nz+3]) +
                                 bi*(mu[3:nx+3,4:ny+5,3:nz+3] + mu[3:nx+3,1:ny+2,3:nz+3]) +
                                 ci*(mu[3:nx+3,5:ny+6,3:nz+3] + mu[3:nx+3,0:ny+1,3:nz+3])))*\
                                 (2.*(g*(a*dy1*(_q[3:nx+3,3:ny+4,3:nz+3,2]/
                                 _q[3:nx+3,3:ny+4,3:nz+3,0] - _q[3:nx+3,2:ny+3,3:nz+3,2]/_q[3:nx+3,2:ny+3,3:nz+3,0]) +
                                 b*dy3*(_q[3:nx+3,4:ny+5,3:nz+3,2]/_q[3:nx+3,4:ny+5,3:nz+3,0] -
                                 _q[3:nx+3,1:ny+2,3:nz+3,2]/_q[3:nx+3,1:ny+2,3:nz+3,0]) + c*dy5*
                                 (_q[3:nx+3,5:ny+6,3:nz+3,2] /_q[3:nx+3,5:ny+6,3:nz+3,0] - _q[3:nx+3,0:ny+1,3:nz+3,2]/
                                 _q[3:nx+3,0:ny+1,3:nz+3,0])))-(gi*(ai*(cd3[:,3:ny+4,:] + cd3[:,2:ny+3,:])+
                                 bi*(cd3[:,4:ny+5,:] + cd3[:,1:ny+2,:]) + ci*(cd3[:,5:ny+6,:] +
                                 cd3[:,0:ny+1,:]))) - (gi*(ai*(cd4[:,3:ny+4,:] + cd4[:,2:ny+3,:]) +
                                 bi*(cd4[:,4:ny+5,:] + cd4[:,1:ny+2,:]) + ci*(cd4[:,5:ny+6,:] +
                                 cd4[:,0:ny+1,:]))))
    vg[3:nx+3,2:ny+3,3:nz+3,3] = g1*(gi*(ai*(mu[3:nx+3,3:ny+4,3:nz+3] + mu[3:nx+3,2:ny+3,3:nz+3]) +
                                 bi*(mu[3:nx+3,4:ny+5,3:nz+3] + mu[3:nx+3,1:ny+2,3:nz+3]) +
                                 ci*(mu[3:nx+3,5:ny+6,3:nz+3] + mu[3:nx+3,0:ny+1,3:nz+3])))*\
                                 ((gi*(ai*(cd2[:,3:ny+4,:] + cd2[:,2:ny+3,:]) +
                                 bi*(cd2[:,4:ny+5,:] + cd2[:,1:ny+2,:]) + ci*(cd2[:,5:ny+6,:] +
                                 cd2[:,0:ny+1,:])))+(g*(a*dy1*(_q[3:nx+3,3:ny+4,3:nz+3,3]/_q[3:nx+3,3:ny+4,3:nz+3,0]
                                 - _q[3:nx+3,2:ny+3,3:nz+3,3]/_q[3:nx+3,2:ny+3,3:nz+3,0])
                                  + b*dy3*(_q[3:nx+3,4:ny+5,3:nz+3,3]/_q[3:nx+3,4:ny+5,3:nz+3,0] -
                                _q[3:nx+3,1:ny+2,3:nz+3,3]/_q[3:nx+3,1:ny+2,3:nz+3,0])+c*dy5*(_q[3:nx+3,5:ny+6,3:nz+3,3]
                                /_q[3:nx+3,5:ny+6,3:nz+3,0] - _q[3:nx+3,0:ny+1,3:nz+3,3]/_q[3:nx+3,0:ny+1,3:nz+3,0]))))
    vg[3:nx+3,2:ny+3,3:nz+3,4] = (gi*(ai*(_q[3:nx+3,3:ny+4,3:nz+3,1]/_q[3:nx+3,3:ny+4,3:nz+3,0] +
                                 _q[3:nx+3,2:ny+3,3:nz+3,1]/_q[3:nx+3,2:ny+3,3:nz+3,0])+ bi*(_q[3:nx+3,4:ny+5,3:nz+3,1]/
                                 _q[3:nx+3,4:ny+5,3:nz+3,0] + _q[3:nx+3,1:ny+2,3:nz+3,1]/_q[3:nx+3,1:ny+2,3:nz+3,0])
                                 + ci*(_q[3:nx+3,5:ny+6,3:nz+3,1]/_q[3:nx+3,5:ny+6,3:nz+3,0]+_q[3:nx+3,0:ny+1,3:nz+3,1]
                                 /_q[3:nx+3,0:ny+1,3:nz+3,0])))*(g1*(gi*(ai*(mu[3:nx+3,3:ny+4,3:nz+3]+
                                 mu[3:nx+3,2:ny+3,3:nz+3]) +bi*(mu[3:nx+3,4:ny+5,3:nz+3] + mu[3:nx+3,1:ny+2,3:nz+3]) +
                                 ci*(mu[3:nx+3,5:ny+6,3:nz+3] + mu[3:nx+3,0:ny+1,3:nz+3])))*((gi*
                                 (ai*(cd1[:,3:ny+4,:] + cd1[:,2:ny+3,:])+bi*(cd1[:,4:ny+5,:] +
                                 cd1[:,1:ny+2,:]) + ci*(cd1[:,5:ny+6,:] + cd1[:,0:ny+1,:])))
                                 + (g*(a*dy1*(_q[3:nx+3,3:ny+4,3:nz+3,1]/_q[3:nx+3,3:ny+4,3:nz+3,0] -
                                 _q[3:nx+3,2:ny+3,3:nz+3,1]/_q[3:nx+3,2:ny+3,3:nz+3,0])
                                  + b*dy3*(_q[3:nx+3,4:ny+5,3:nz+3,1]/_q[3:nx+3,4:ny+5,3:nz+3,0] -
                                _q[3:nx+3,1:ny+2,3:nz+3,1]/_q[3:nx+3,1:ny+2,3:nz+3,0])
                                  + c*dy5*(_q[3:nx+3,5:ny+6,3:nz+3,1]/_q[3:nx+3,5:ny+6,3:nz+3,0] -
                                _q[3:nx+3,0:ny+1,3:nz+3,1]/_q[3:nx+3,0:ny+1,3:nz+3,0])))))+\
                                (gi*(ai*(_q[3:nx+3,3:ny+4,3:nz+3,2]/_q[3:nx+3,3:ny+4,3:nz+3,0]+
                                _q[3:nx+3,2:ny+3,3:nz+3,2]/_q[3:nx+3,2:ny+3,3:nz+3,0])
                                 + bi*(_q[3:nx+3,4:ny+5,3:nz+3,2]/_q[3:nx+3,4:ny+5,3:nz+3,0]+_q[3:nx+3,1:ny+2,3:nz+3,2]/
                                _q[3:nx+3,1:ny+2,3:nz+3,0])+ci*(_q[3:nx+3,5:ny+6,3:nz+3,2]/_q[3:nx+3,5:ny+6,3:nz+3,0] +
                                _q[3:nx+3,0:ny+1,3:nz+3,2]/_q[3:nx+3,0:ny+1,3:nz+3,0])))*(g2*(gi*(ai*
                                (mu[3:nx+3,3:ny+4,3:nz+3] + mu[3:nx+3,2:ny+3,3:nz+3]) +
                                 bi*(mu[3:nx+3,4:ny+5,3:nz+3] + mu[3:nx+3,1:ny+2,3:nz+3]) +
                                 ci*(mu[3:nx+3,5:ny+6,3:nz+3] + mu[3:nx+3,0:ny+1,3:nz+3])))*\
                                 (2.*(g*(a*dy1*(_q[3:nx+3,3:ny+4,3:nz+3,2]/
                                 _q[3:nx+3,3:ny+4,3:nz+3,0] - _q[3:nx+3,2:ny+3,3:nz+3,2]/_q[3:nx+3,2:ny+3,3:nz+3,0]) +
                                 b*dy3*(_q[3:nx+3,4:ny+5,3:nz+3,2]/_q[3:nx+3,4:ny+5,3:nz+3,0] -
                                 _q[3:nx+3,1:ny+2,3:nz+3,2]/_q[3:nx+3,1:ny+2,3:nz+3,0]) + c*dy5*
                                 (_q[3:nx+3,5:ny+6,3:nz+3,2] /_q[3:nx+3,5:ny+6,3:nz+3,0] - _q[3:nx+3,0:ny+1,3:nz+3,2]/
                                 _q[3:nx+3,0:ny+1,3:nz+3,0])))-(gi*(ai*(cd3[:,3:ny+4,:] + cd3[:,2:ny+3,:])+
                                 bi*(cd3[:,4:ny+5,:] + cd3[:,1:ny+2,:]) + ci*(cd3[:,5:ny+6,:] +
                                 cd3[:,0:ny+1,:]))) - (gi*(ai*(cd4[:,3:ny+4,:] + cd4[:,2:ny+3,:]) +
                                 bi*(cd4[:,4:ny+5,:] + cd4[:,1:ny+2,:]) + ci*(cd4[:,5:ny+6,:] +
                                 cd4[:,0:ny+1,:])))))+(gi*(ai*(_q[3:nx+3,3:ny+4,3:nz+3,3]/
                                _q[3:nx+3,3:ny+4,3:nz+3,0] + _q[3:nx+3,2:ny+3,3:nz+3,3]/_q[3:nx+3,2:ny+3,3:nz+3,0])
                                 + bi*(_q[3:nx+3,4:ny+5,3:nz+3,3]/_q[3:nx+3,4:ny+5,3:nz+3,0]+_q[3:nx+3,1:ny+2,3:nz+3,3]
                                /_q[3:nx+3,1:ny+2,3:nz+3,0])+ ci*(_q[3:nx+3,5:ny+6,3:nz+3,3]/_q[3:nx+3,5:ny+6,3:nz+3,0]
                                + _q[3:nx+3,0:ny+1,3:nz+3,3]/_q[3:nx+3,0:ny+1,3:nz+3,0])))*(g1*(gi*(ai*
                                (mu[3:nx+3,3:ny+4,3:nz+3] + mu[3:nx+3,2:ny+3,3:nz+3]) +
                                 bi*(mu[3:nx+3,4:ny+5,3:nz+3] + mu[3:nx+3,1:ny+2,3:nz+3]) +
                                 ci*(mu[3:nx+3,5:ny+6,3:nz+3] + mu[3:nx+3,0:ny+1,3:nz+3])))*\
                                 ((gi*(ai*(cd2[:,3:ny+4,:] + cd2[:,2:ny+3,:]) +
                                 bi*(cd2[:,4:ny+5,:] + cd2[:,1:ny+2,:]) + ci*(cd2[:,5:ny+6,:] +
                                 cd2[:,0:ny+1,:])))+(g*(a*dy1*(_q[3:nx+3,3:ny+4,3:nz+3,3]/_q[3:nx+3,3:ny+4,3:nz+3,0]
                                 - _q[3:nx+3,2:ny+3,3:nz+3,3]/_q[3:nx+3,2:ny+3,3:nz+3,0])
                                  + b*dy3*(_q[3:nx+3,4:ny+5,3:nz+3,3]/_q[3:nx+3,4:ny+5,3:nz+3,0] -
                                _q[3:nx+3,1:ny+2,3:nz+3,3]/_q[3:nx+3,1:ny+2,3:nz+3,0])+c*dy5*(_q[3:nx+3,5:ny+6,3:nz+3,3]
                                /_q[3:nx+3,5:ny+6,3:nz+3,0]-_q[3:nx+3,0:ny+1,3:nz+3,3]/_q[3:nx+3,0:ny+1,3:nz+3,0])))))\
                                 - g3*(g*(a*dy1*(t[3:nx+3,3:ny+4,3:nz+3] -t[3:nx+3,2:ny+3,3:nz+3])+
                                b*dy3*(t[3:nx+3,4:ny+5,3:nz+3]-t[3:nx+3,1:ny+2,3:nz+3])+ c*dy5*(t[3:nx+3,5:ny+6,3:nz+3]
                                -t[3:nx+3,0:ny+1,3:nz+3])))*(gi*(ai*(mu[3:nx+3,3:ny+4,3:nz+3]+mu[3:nx+3,2:ny+3,3:nz+3])+
                                 bi*(mu[3:nx+3,4:ny+5,3:nz+3] + mu[3:nx+3,1:ny+2,3:nz+3]) +
                                 ci*(mu[3:nx+3,5:ny+6,3:nz+3] + mu[3:nx+3,0:ny+1,3:nz+3])))

    _s[3:nx+3,3:ny+3,3:nz+3,0:5] = _s[3:nx+3,3:ny+3,3:nz+3,0:5] + (vg[3:nx+3,3:ny+3,3:nz+3,0:5]
                                   - vg[3:nx+3,2:ny+2,3:nz+3,0:5])/dy
    del cd1,cd2,cd3,cd4

    cd1 = np.zeros(shape=(nx,ny,nz+6),dtype='double')
    cd2 = np.zeros(shape=(nx,ny,nz+6),dtype='double')
    cd3 = np.zeros(shape=(nx,ny,nz+6),dtype='double')
    cd4 = np.zeros(shape=(nx,ny,nz+6),dtype='double') # cd = 0:nx, 0:ny, 0:nz+6

    #  z-direction
    #  wx
    cd1[:,:,:] = gd*(ad*dx2*(_q[4:nx+4,3:ny+3,:,3]/_q[4:nx+4,3:ny+3,:,0] -_q[2:nx+2,3:ny+3,:,3]/_q[2:nx+2,3:ny+3,:,0])
                 + bd*dx4*(_q[5:nx+5,3:ny+3,:,3]/_q[5:nx+5,3:ny+3,:,0] - _q[1:nx+1,3:ny+3,:,3]/_q[1:nx+1,3:ny+3,:,0]) +
                 cd*dx6*(_q[6:nx+6,3:ny+3,:,3]/_q[6:nx+6,3:ny+3,:,0] -_q[0:nx,3:ny+3,:,3]/_q[0:nx,3:ny+3,:,0]))
    #  wy
    cd2[:,:,:] = gd*(ad*dy2*(_q[3:nx+3,4:ny+4,:,3]/_q[3:nx+3,4:ny+4,:,0] - _q[3:nx+3,2:ny+2,:,3]/_q[3:nx+3,2:ny+2,:,0])
                 + bd*dy4*(_q[3:nx+3,5:ny+5,:,3] /_q[3:nx+3,5:ny+5,:,0] -_q[3:nx+3,1:ny+1,:,3]/_q[3:nx+3,1:ny+1,:,0]) +
                 cd*dy6*(_q[3:nx+3,6:ny+6,:,3]/_q[3:nx+3,6:ny+6,:,0] - _q[3:nx+3,0:ny,:,3]/_q[3:nx+3,0:ny,:,0]))

    #  ux
    cd3[:,:,:] = gd*(ad*dx2*(_q[4:nx+4,3:ny+3,:,1]/_q[4:nx+4,3:ny+3,:,0] -_q[2:nx+2,3:ny+3,:,1]/_q[2:nx+2,3:ny+3,:,0])
                 + bd*dx4*(_q[5:nx+5,3:ny+3,:,1]/_q[5:nx+5,3:ny+3,:,0] -_q[1:nx+1,3:ny+3,:,1]/_q[1:nx+1,3:ny+3,:,0]) +
                 cd*dx6*(_q[6:nx+6,3:ny+3,:,1]/_q[6:nx+6,3:ny+3,:,0] -_q[0:nx,3:ny+3,:,1]/_q[0:nx,3:ny+3,:,0]))

    #  vy
    cd4[:,:,:] = gd*(ad*dy2*(_q[3:nx+3,4:ny+4,:,2]/_q[3:nx+3,4:ny+4,:,0] -_q[3:nx+3,2:ny+2,:,2]/_q[3:nx+3,2:ny+2,:,0])+
                 bd*dy4*(_q[3:nx+3,5:ny+5,:,2]/_q[3:nx+3,5:ny+5,:,0] -_q[3:nx+3,1:ny+1,:,2]/_q[3:nx+3,1:ny+1,:,0]) +
                 cd*dy6*(_q[3:nx+3,6:ny+6,:,2]/_q[3:nx+3,6:ny+6,:,0] -_q[3:nx+3,0:ny,:,2]/_q[3:nx+3,0:ny,:,0]))

    vh[3:nx+3,3:ny+3,2:nz+3,0] = 0.
    vh[3:nx+3,3:ny+3,2:nz+3,1] = g1*(gi*(ai*(mu[3:nx+3,3:ny+3,3:nz+4] + mu[3:nx+3,3:ny+3,2:nz+3]) +
                                 bi*(mu[3:nx+3,3:ny+3,4:nz+5] + mu[3:nx+3,3:ny+3,1:nz+2]) +
                                 ci*(mu[3:nx+3,3:ny+3,5:nz+6] + mu[3:nx+3,3:ny+3,0:nz+1])))*((gi*
                                 (ai*(cd1[:,:,3:nz+4] + cd1[:,:,2:nz+3])+bi*(cd1[:,:,4:nz+5] +
                                 cd1[:,:,1:nz+2]) + ci*(cd1[:,:,5:nz+6] + cd1[:,:,0:nz+1])))
                                 + (g*(a*dz1*(_q[3:nx+3,3:ny+3,3:nz+4,1]/_q[3:nx+3,3:ny+3,3:nz+4,0] -
                                 _q[3:nx+3,3:ny+3,2:nz+3,1]/_q[3:nx+3,3:ny+3,2:nz+3,0])
                                  + b*dz3*(_q[3:nx+3,3:ny+3,4:nz+5,1]/_q[3:nx+3,3:ny+3,4:nz+5,0] -
                                _q[3:nx+3,3:ny+3,1:nz+2,1]/_q[3:nx+3,3:ny+3,1:nz+2,0])
                                  + c*dz5*(_q[3:nx+3,3:ny+3,5:nz+6,1]/_q[3:nx+3,3:ny+3,5:nz+6,0] -
                                _q[3:nx+3,3:ny+3,0:nz+1,1]/_q[3:nx+3,3:ny+3,0:nz+1,0]))))
    vh[3:nx+3,3:ny+3,2:nz+3,2] = g1*(gi*(ai*(mu[3:nx+3,3:ny+3,3:nz+4] + mu[3:nx+3,3:ny+3,2:nz+3]) +
                                 bi*(mu[3:nx+3,3:ny+3,4:nz+5] + mu[3:nx+3,3:ny+3,1:nz+2]) +
                                 ci*(mu[3:nx+3,3:ny+3,5:nz+6] + mu[3:nx+3,3:ny+3,0:nz+1])))*((gi*
                                 (ai*(cd2[:,:,3:nz+4] + cd2[:,:,2:nz+3])+bi*(cd2[:,:,4:nz+5] +
                                 cd2[:,:,1:nz+2]) + ci*(cd2[:,:,5:nz+6] + cd2[:,:,0:nz+1])))
                                 + (g*(a*dz1*(_q[3:nx+3,3:ny+3,3:nz+4,2]/_q[3:nx+3,3:ny+3,3:nz+4,0] -
                                 _q[3:nx+3,3:ny+3,2:nz+3,2]/_q[3:nx+3,3:ny+3,2:nz+3,0])
                                  + b*dz3*(_q[3:nx+3,3:ny+3,4:nz+5,2]/_q[3:nx+3,3:ny+3,4:nz+5,0] -
                                _q[3:nx+3,3:ny+3,1:nz+2,2]/_q[3:nx+3,3:ny+3,1:nz+2,0])
                                  + c*dz5*(_q[3:nx+3,3:ny+3,5:nz+6,2]/_q[3:nx+3,3:ny+3,5:nz+6,0] -
                                _q[3:nx+3,3:ny+3,0:nz+1,2]/_q[3:nx+3,3:ny+3,0:nz+1,0]))))
    vh[3:nx+3,3:ny+3,2:nz+3,3] = g2*(gi*(ai*(mu[3:nx+3,3:ny+3,3:nz+4] + mu[3:nx+3,3:ny+3,2:nz+3]) +
                                 bi*(mu[3:nx+3,3:ny+3,4:nz+5] + mu[3:nx+3,3:ny+3,1:nz+2]) +
                                 ci*(mu[3:nx+3,3:ny+3,5:nz+6] + mu[3:nx+3,3:ny+3,0:nz+1])))*\
                                 (2.*(g*(a*dz1*(_q[3:nx+3,3:ny+3,3:nz+4,3]/
                                 _q[3:nx+3,3:ny+3,3:nz+4,0] - _q[3:nx+3,3:ny+3,2:nz+3,3]/_q[3:nx+3,3:ny+3,2:nz+3,0]) +
                                 b*dz3*(_q[3:nx+3,3:ny+3,4:nz+5,3]/_q[3:nx+3,3:ny+3,4:nz+5,0] -
                                 _q[3:nx+3,3:ny+3,1:nz+2,3]/_q[3:nx+3,3:ny+3,1:nz+2,0]) + c*dz5*
                                 (_q[3:nx+3,3:ny+3,5:nz+6,3] /_q[3:nx+3,3:ny+3,5:nz+6,0] - _q[3:nx+3,3:ny+3,0:nz+1,3]/
                                 _q[3:nx+3,3:ny+3,0:nz+1,0])))-(gi*(ai*(cd3[:,:,3:nz+4] + cd3[:,:,2:nz+3])+
                                 bi*(cd3[:,:,4:nz+5] + cd3[:,:,1:nz+2]) + ci*(cd3[:,:,5:nz+6] +
                                 cd3[:,:,0:nz+1]))) - (gi*(ai*(cd4[:,:,3:nz+4] + cd4[:,:,2:nz+3])+
                                 bi*(cd4[:,:,4:nz+5] + cd4[:,:,1:nz+2]) + ci*(cd4[:,:,5:nz+6] +
                                 cd4[:,:,0:nz+1]))))
    vh[3:nx+3,3:ny+3,2:nz+3,4] = (gi*(ai*(_q[3:nx+3,3:ny+3,3:nz+4,1]/_q[3:nx+3,3:ny+3,3:nz+4,0] +
                                 _q[3:nx+3,3:ny+3,2:nz+3,1]/_q[3:nx+3,3:ny+3,2:nz+3,0])+ bi*(_q[3:nx+3,3:ny+3,4:nz+5,1]/
                                 _q[3:nx+3,3:ny+3,4:nz+5,0] + _q[3:nx+3,3:ny+3,1:nz+2,1]/_q[3:nx+3,3:ny+3,1:nz+2,0])
                                 + ci*(_q[3:nx+3,3:ny+3,5:nz+6,1]/_q[3:nx+3,3:ny+3,5:nz+6,0]+_q[3:nx+3,3:ny+3,0:nz+1,1]
                                 /_q[3:nx+3,3:ny+3,0:nz+1,0])))*(g1*(gi*(ai*(mu[3:nx+3,3:ny+3,3:nz+4] +
                                 mu[3:nx+3,3:ny+3,2:nz+3]) +
                                 bi*(mu[3:nx+3,3:ny+3,4:nz+5] + mu[3:nx+3,3:ny+3,1:nz+2]) +
                                 ci*(mu[3:nx+3,3:ny+3,5:nz+6] + mu[3:nx+3,3:ny+3,0:nz+1])))*((gi*
                                 (ai*(cd1[:,:,3:nz+4] + cd1[:,:,2:nz+3])+bi*(cd1[:,:,4:nz+5] +
                                 cd1[:,:,1:nz+2]) + ci*(cd1[:,:,5:nz+6] + cd1[:,:,0:nz+1])))
                                 + (g*(a*dz1*(_q[3:nx+3,3:ny+3,3:nz+4,1]/_q[3:nx+3,3:ny+3,3:nz+4,0] -
                                 _q[3:nx+3,3:ny+3,2:nz+3,1]/_q[3:nx+3,3:ny+3,2:nz+3,0])
                                  + b*dz3*(_q[3:nx+3,3:ny+3,4:nz+5,1]/_q[3:nx+3,3:ny+3,4:nz+5,0] -
                                _q[3:nx+3,3:ny+3,1:nz+2,1]/_q[3:nx+3,3:ny+3,1:nz+2,0])
                                  + c*dz5*(_q[3:nx+3,3:ny+3,5:nz+6,1]/_q[3:nx+3,3:ny+3,5:nz+6,0] -
                                _q[3:nx+3,3:ny+3,0:nz+1,1]/_q[3:nx+3,3:ny+3,0:nz+1,0])))))+(gi*(ai*
                                (_q[3:nx+3,3:ny+3,3:nz+4,2]/_q[3:nx+3,3:ny+3,3:nz+4,0]+_q[3:nx+3,3:ny+3,2:nz+3,2]/
                                _q[3:nx+3,3:ny+3,2:nz+3,0])+ bi*(_q[3:nx+3,3:ny+3,4:nz+5,2]/_q[3:nx+3,3:ny+3,4:nz+5,0]+
                                _q[3:nx+3,3:ny+3,1:nz+2,2]/_q[3:nx+3,3:ny+3,1:nz+2,0])+ci*(_q[3:nx+3,3:ny+3,5:nz+6,2]/
                                _q[3:nx+3,3:ny+3,5:nz+6,0]+_q[3:nx+3,3:ny+3,0:nz+1,2]/_q[3:nx+3,3:ny+3,0:nz+1,0])))*\
                                (g1*(gi*(ai*(mu[3:nx+3,3:ny+3,3:nz+4] + mu[3:nx+3,3:ny+3,2:nz+3]) +
                                 bi*(mu[3:nx+3,3:ny+3,4:nz+5] + mu[3:nx+3,3:ny+3,1:nz+2]) +
                                 ci*(mu[3:nx+3,3:ny+3,5:nz+6] + mu[3:nx+3,3:ny+3,0:nz+1])))*((gi*
                                 (ai*(cd2[:,:,3:nz+4] + cd2[:,:,2:nz+3])+bi*(cd2[:,:,4:nz+5] +
                                 cd2[:,:,1:nz+2]) + ci*(cd2[:,:,5:nz+6] + cd2[:,:,0:nz+1])))
                                 + (g*(a*dz1*(_q[3:nx+3,3:ny+3,3:nz+4,2]/_q[3:nx+3,3:ny+3,3:nz+4,0] -
                                 _q[3:nx+3,3:ny+3,2:nz+3,2]/_q[3:nx+3,3:ny+3,2:nz+3,0])
                                  + b*dz3*(_q[3:nx+3,3:ny+3,4:nz+5,2]/_q[3:nx+3,3:ny+3,4:nz+5,0] -
                                _q[3:nx+3,3:ny+3,1:nz+2,2]/_q[3:nx+3,3:ny+3,1:nz+2,0])
                                  + c*dz5*(_q[3:nx+3,3:ny+3,5:nz+6,2]/_q[3:nx+3,3:ny+3,5:nz+6,0] -
                                _q[3:nx+3,3:ny+3,0:nz+1,2]/_q[3:nx+3,3:ny+3,0:nz+1,0])))))+(gi*(ai*
                                (_q[3:nx+3,3:ny+3,3:nz+4,3]/
                                _q[3:nx+3,3:ny+3,3:nz+4,0] + _q[3:nx+3,3:ny+3,2:nz+3,3]/_q[3:nx+3,3:ny+3,2:nz+3,0])
                                 + bi*(_q[3:nx+3,3:ny+3,4:nz+5,3]/_q[3:nx+3,3:ny+3,4:nz+5,0]+_q[3:nx+3,3:ny+3,1:nz+2,3]
                                /_q[3:nx+3,3:ny+3,1:nz+2,0])+ ci*(_q[3:nx+3,3:ny+3,5:nz+6,3]/_q[3:nx+3,3:ny+3,5:nz+6,0]
                                + _q[3:nx+3,3:ny+3,0:nz+1,3]/_q[3:nx+3,3:ny+3,0:nz+1,0])))*\
                                 (g2*(gi*(ai*(mu[3:nx+3,3:ny+3,3:nz+4] + mu[3:nx+3,3:ny+3,2:nz+3]) +
                                 bi*(mu[3:nx+3,3:ny+3,4:nz+5] + mu[3:nx+3,3:ny+3,1:nz+2]) +
                                 ci*(mu[3:nx+3,3:ny+3,5:nz+6] + mu[3:nx+3,3:ny+3,0:nz+1])))*\
                                 (2.*(g*(a*dz1*(_q[3:nx+3,3:ny+3,3:nz+4,3]/
                                 _q[3:nx+3,3:ny+3,3:nz+4,0] - _q[3:nx+3,3:ny+3,2:nz+3,3]/_q[3:nx+3,3:ny+3,2:nz+3,0]) +
                                 b*dz3*(_q[3:nx+3,3:ny+3,4:nz+5,3]/_q[3:nx+3,3:ny+3,4:nz+5,0] -
                                 _q[3:nx+3,3:ny+3,1:nz+2,3]/_q[3:nx+3,3:ny+3,1:nz+2,0]) + c*dz5*
                                 (_q[3:nx+3,3:ny+3,5:nz+6,3] /_q[3:nx+3,3:ny+3,5:nz+6,0] - _q[3:nx+3,3:ny+3,0:nz+1,3]/
                                 _q[3:nx+3,3:ny+3,0:nz+1,0])))-(gi*(ai*(cd3[:,:,3:nz+4] + cd3[:,:,2:nz+3])+
                                 bi*(cd3[:,:,4:nz+5] + cd3[:,:,1:nz+2]) + ci*(cd3[:,:,5:nz+6] +
                                 cd3[:,:,0:nz+1]))) - (gi*(ai*(cd4[:,:,3:nz+4] + cd4[:,:,2:nz+3])+
                                 bi*(cd4[:,:,4:nz+5] + cd4[:,:,1:nz+2]) + ci*(cd4[:,:,5:nz+6] +
                                 cd4[:,:,0:nz+1])))))- g3*(g*(a*dz1*(t[3:nx+3,3:ny+3,3:nz+4] -t[3:nx+3,3:ny+3,2:nz+3])+
                                 b*dz3*(t[3:nx+3,3:ny+3,4:nz+5]-t[3:nx+3,3:ny+3,1:nz+2])+ c*dz5*(t[3:nx+3,3:ny+3,5:nz+6]
                                 -t[3:nx+3,3:ny+3,0:nz+1])))*(gi*(ai*(mu[3:nx+3,3:ny+3,3:nz+4]+mu[3:nx+3,3:ny+3,2:nz+3])
                                 + bi*(mu[3:nx+3,3:ny+3,4:nz+5] + mu[3:nx+3,3:ny+3,1:nz+2]) +
                                 ci*(mu[3:nx+3,3:ny+3,5:nz+6] + mu[3:nx+3,3:ny+3,0:nz+1])))
    _s[3:nx+3,3:ny+3,3:nz+3,0:5] = _s[3:nx+3,3:ny+3,3:nz+3,0:5] + (vh[3:nx+3,3:ny+3,3:nz+3,0:5]
                                   - vh[3:nx+3,3:ny+3,2:nz+2,0:5])/dz
    del cd1,cd2,cd3,cd4
    del t,mu,vf,vg,vh
    return _s


def rhs(nx,ny,nz,dx,dy,dz,q,s,gamma,Tref,re,pr,ma,imu,imodel):
    if imodel == 1:  #Euler
        s = rhscs(nx,ny,nz,dx,dy,dz,q,s,gamma)
    else:  # Navier-Stokes
        s = rhscs(nx,ny,nz,dx,dy,dz,q,s,gamma)
        s = rhsvis(nx,ny,nz,dx,dy,dz,q,s,gamma,Tref,re,pr,ma,imu)
    return s


def perbc_xyz(nx,ny,nz,m):

    m[3:nx+3,2,3:nz+3,0:5] = m[3:nx+3,ny+2,3:nz+3,0:5]
    m[3:nx+3,1,3:nz+3,0:5] = m[3:nx+3,ny+1,3:nz+3,0:5]
    m[3:nx+3,0,3:nz+3,0:5] = m[3:nx+3,ny,3:nz+3,0:5]
    m[3:nx+3,ny+3,3:nz+3,0:5] = m[3:nx+3,3,3:nz+3,0:5]
    m[3:nx+3,ny+4,3:nz+3,0:5] = m[3:nx+3,4,3:nz+3,0:5]
    m[3:nx+3,ny+5,3:nz+3,0:5] = m[3:nx+3,5,3:nz+3,0:5]

    m[2,0:ny+6,3:nz+3,0:5] = m[nx+2,0:ny+6,3:nz+3,0:5]
    m[1,0:ny+6,3:nz+3,0:5] = m[nx+1,0:ny+6,3:nz+3,0:5]
    m[0,0:ny+6,3:nz+3,0:5] = m[nx,0:ny+6,3:nz+3,0:5]
    m[nx+3,0:ny+6,3:nz+3,0:5] = m[3,0:ny+6,3:nz+3,0:5]
    m[nx+4,0:ny+6,3:nz+3,0:5] = m[4,0:ny+6,3:nz+3,0:5]
    m[nx+5,0:ny+6,3:nz+3,0:5] = m[5,0:ny+6,3:nz+3,0:5]

    m[0:nx+6,0:ny+6,2,0:5] = m[0:nx+6,0:ny+6,nz+2,0:5]
    m[0:nx+6,0:ny+6,1,0:5] = m[0:nx+6,0:ny+6,nz+1,0:5]
    m[0:nx+6,0:ny+6,0,0:5] = m[0:nx+6,0:ny+6,nz,0:5]
    m[0:nx+6,0:ny+6,nz+3,0:5] = m[0:nx+6,0:ny+6,3,0:5]
    m[0:nx+6,0:ny+6,nz+4,0:5] = m[0:nx+6,0:ny+6,4,0:5]
    m[0:nx+6,0:ny+6,nz+5,0:5] = m[0:nx+6,0:ny+6,5,0:5]

    return m


def timestep(nx,ny,nz,dx,dy,dz,cfl,_dt,q,imu,gamma,ma,nustab,aveeddy,prt):
    smx = 0.
    smy = 0.
    smz = 0.

    for k in range(3, nz+3):
        for j in range(3, ny+3):
            for i in range(3, nx+3):

                p = (gamma - 1.)*(q[i,j,k,4] - 0.5*(np.square(q[i,j,k,1])/q[i,j,k,0]
                                                       +np.square(q[i,j,k,2])/q[i,j,k,0]
                                                       +np.square(q[i,j,k,3])/q[i,j,k,0]))
                a = np.sqrt(gamma*p/q[i,j,k,0])

                l1 = abs(q[i,j,k,1]/q[i,j,k,0])
                l2 = abs(q[i,j,k,1]/q[i,j,k,0] + a)
                l3 = abs(q[i,j,k,1]/q[i,j,k,0] - a)
                radx = max(l1,l2,l3)

                l1 = abs(q[i,j,k,2]/q[i,j,k,0])
                l2 = abs(q[i,j,k,2]/q[i,j,k,0] + a)
                l3 = abs(q[i,j,k,2]/q[i,j,k,0] - a)
                rady = max(l1,l2,l3)

                l1 = abs(q[i,j,k,3]/q[i,j,k,0])
                l2 = abs(q[i,j,k,3]/q[i,j,k,0] + a)
                l3 = abs(q[i,j,k,3]/q[i,j,k,0] - a)
                radz = max(l1,l2,l3)

                if radx > smx:
                    smx = radx
                if rady > smy:
                    smy = rady
                if radz > smz:
                    smz = radz

    _dt = min(cfl*dx/smx,cfl*dy/smy,cfl*dz/smz)
    maxeddy = 0.
    if aveeddy >= 1e-11:
        if imu == 1:
            vis = max(maxeddy/(prt*ma*ma*(gamma-1.)),maxeddy)
        else:
            vis = np.max(maxeddy*gamma/prt,maxeddy)
        dtvis = (nustab/vis)*min(dx*dx,dy*dy,dz*dz)
        _dt = min(_dt,dtvis)

    return _dt



#  Output files: Tecplot
def outputTec(nx,ny,nz,x,y,z,q,t,ifile,gamma):

    # density
    f = open("a_density_"+str(ifile)+".plt", "w+")
    f.write('title="Zone_'+str(ifile)+'_Time_'+str(t)+'"''\n')
    f.write('variables = "x", "y", "z", "r"''\n')
    f.write('zone T=Zone_'+str(ifile)+' i=          '+str(nx+1)+' j=          '+str(ny+1)+' k=          '+str(nz+1)+
            ' f=point''\n')

    output_array = np.zeros(((nx+1)*(ny+1)*(nz+1),4),dtype='double')
    it = 0
    for k in range(2,nz+3):
        for j in range(2,ny+3):
            for i in range(2,nx+3):
                output_array[it, 0] = x[i,j,k]
                output_array[it, 1] = y[i,j,k]
                output_array[it, 2] = z[i,j,k]
                output_array[it, 3] = q[i,j,k,0]
                it = it + 1

    np.savetxt(f,output_array,fmt='%10.7e')
    f.close()
    del output_array, it

    # pressure
    f = open("a_pressure_"+str(ifile)+".plt", "w+")
    f.write('title="Zone_'+str(ifile)+'_Time_'+str(t)+'"''\n')
    f.write('variables = "x", "y", "z", "p"''\n')
    f.write('zone T=Zone_'+str(ifile)+' i=          '+str(nx+1)+' j=          '+str(ny+1)+' k=          '+str(nz+1)+
            ' f=point''\n')

    output_array = np.zeros(((nx+1)*(ny+1)*(nz+1),4),dtype='double')
    it = 0
    for k in range(2,nz+3):
        for j in range(2,ny+3):
            for i in range(2,nx+3):
                p = (gamma - 1.)*(q[i,j,k,4] - 0.5*(q[i,j,k,1]*q[i,j,k,1]/q[i,j,k,0] + q[i,j,k,2]*q[i,j,k,2]/q[i,j,k,0]
                    + q[i,j,k,3]*q[i,j,k,3]/q[i,j,k,0]))
                output_array[it, 0] = x[i,j,k]
                output_array[it, 1] = y[i,j,k]
                output_array[it, 2] = z[i,j,k]
                output_array[it, 3] = p
                it = it + 1

    np.savetxt(f,output_array,fmt='%10.7e')
    f.close()
    del output_array, it

    #  velocity
    f = open("a_velocity_"+str(ifile)+".plt", "w+")
    f.write('title="Zone_'+str(ifile)+'_Time_'+str(t)+'"''\n')
    f.write('variables = "x", "y", "z", "u", "v", "w"''\n')
    f.write('zone T=Zone_'+str(ifile)+' i=          '+str(nx+1)+' j=          '+str(ny+1)+' k=          '+str(nz+1)+
            ' f=point''\n')

    output_array = np.zeros(((nx+1)*(ny+1)*(nz+1),6),dtype='double')
    it = 0
    for k in range(2,nz+3):
        for j in range(2,ny+3):
            for i in range(2,nx+3):
                output_array[it, 0] = x[i,j,k]
                output_array[it, 1] = y[i,j,k]
                output_array[it, 2] = z[i,j,k]
                output_array[it, 3] = q[i,j,k,1]/q[i,j,k,0]
                output_array[it, 4] = q[i,j,k,2]/q[i,j,k,0]
                output_array[it, 5] = q[i,j,k,3]/q[i,j,k,0]
                it = it + 1

    np.savetxt(f,output_array,fmt='%10.7e')
    f.close()
    del output_array, it


# 6th order compact scheme for first degree derivative (_up) # periodic b.c.(0 = n), h = grid spacing
def c6dp(_u,_up,h,n):
    a = np.zeros(shape=(n+2),dtype='double')
    b = np.zeros(shape=(n+2),dtype='double')
    c = np.zeros(shape=(n+2),dtype='double')
    r = np.zeros(shape=(n+2),dtype='double')
    x = np.zeros(shape=(n+2),dtype='double')

    a[:] = 1. / 3.
    b[:] = 1.
    c[:] = 1./ 3.
    r[4:n+1] = 14./9.*(_u[5:n+2]-_u[3:n])/(2.*h) + 1./9.*(_u[6:n+3]-_u[2:n-1])/(4.*h)
    r[3] = 14./9.*(_u[4]-_u[2])/(2.*h) + 1./9.*(_u[5]-_u[n+1])/(4.*h)
    r[n+1] = 14./9.*(_u[n+2]-_u[n])/(2.*h) + 1./9.*(_u[3]-_u[n-1])/(4.*h)
    r[2] = 14./9.*(_u[3]-_u[n+1])/(2.*h) + 1./9.*(_u[4]-_u[n])/(4.*h)

    alpha = 1./3.
    beta = 1./3.

    x = ctdms(a,b,c,alpha,beta,r,x,0,n+1)

    _up[2:n+2] = x[2:n+2]
    _up[n+2] = _up[2]

    return _up


def tdms(_a,_b,_c,_r,_u,s,e):

    gam = np.zeros(shape=(e+1),dtype='double')

    bet = _b[s]
    _u[s] = _r[s]/bet
    gam[s+1:e+1] = _c[s:e]/bet
    bet = _b[s+1:e+1]-_a[s+1:e+1]*gam[s+1:e+1]
    _u[s+1:e+1] = (_r[s+1:e+1]-_a[s+1:e+1]*_u[s:e])/bet

    for j in range(e-1,s+1,-1):
        _u[j] = _u[j] - gam[j+1]*_u[j+1]

    return _u


def ctdms(a, b, c, alpha, beta, r, x, s, e):

    bb = np.zeros(shape=(e+1),dtype='double')
    u = np.zeros(shape=(e+1),dtype='double')
    z = np.zeros(shape=(e+1),dtype='double')

    gamma = - b[s]
    bb[s] = b[s]-gamma
    bb[e] = b[e] - alpha*beta/gamma
    bb[s+1:e] = b[s+1:e]
    x = tdms(a,bb,c,r,x,s,e)
    u[s] = gamma
    u[e] = alpha
    u[s+1:e] = 0.
    z = tdms(a,bb,c,u,z,s,e)
    fact = (x[s]+beta*x[e]/gamma)/(1.+z[s]+beta*z[e]/gamma)
    x[s:e+1] = x[s:e+1] - fact*z[s:e+1]

    return x


def spectrum3d(nx,ny,nz,_u,_v,_w,ifile):
    data1d = _u[2:nx+2,2:ny+2,2:nz+2].reshape(nx,ny,nz)
    data2d = _v[2:nx+2,2:ny+2,2:nz+2].reshape(nx,ny,nz)
    data3d = _w[2:nx+2,2:ny+2,2:nz+2].reshape(nx,ny,nz)

    data1d = (np.abs(np.fft.fftn(data1d)/data1d.size))**2
    data2d = (np.abs(np.fft.fftn(data2d)/data2d.size))**2
    data3d = (np.abs(np.fft.fftn(data3d)/data3d.size))**2

    data1d = np.fft.fftshift(data1d)
    data2d = np.fft.fftshift(data2d)
    data3d = np.fft.fftshift(data3d)

    kx = np.shape(data1d)[0]
    ky = np.shape(data2d)[1]
    kz = np.shape(data2d)[2]

    ne = int(np.ceil((np.sqrt((kx)**2+(ky)**2+(kz)**2))/2.)+1)

    ae_u = np.zeros(shape=ne, dtype='double')
    ae_v = np.zeros(shape=ne, dtype='double')
    ae_w = np.zeros(shape=ne, dtype='double')

    for k in range(kz):
        for j in range(ky):
            for i in range(kx):
                kr = int(np.round(np.sqrt((i-int(kx/2))**2+(j-int(ky/2))**2+(k-int(kz/2))**2)))
                ae_u[kr] = ae_u[kr] + data1d[i,j,k]
                ae_v[kr] = ae_v[kr] + data2d[i,j,k]
                ae_w[kr] = ae_w[kr] + data3d[i,j,k]

    ae = 0.5*(ae_u + ae_v + ae_w)

    f = open("a_spectrum_"+str(ifile)+".plt", "w+")
    f.write('variables = "k", "E(k)"''\n')
    output_array = np.zeros((len(ae),2),dtype='double')
    it = 0
    for k in range(0,len(ae)):
        output_array[it, 0] = k
        output_array[it, 1] = ae[k]
        it = it + 1
    np.savetxt(f,output_array,fmt='%10.7e')
    f.close()
    del output_array, it

    del data1d,data2d,data3d

    return


def Qcriterion(nx,ny,nz,dx,dy,dz,x,y,z,_q,t,ifile):

    u = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    v = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    w = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    qc = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    w1 = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    w2 = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    w3 = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    wa = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    ux = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    uy = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    uz = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    vx = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    vy = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    vz = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    wx = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    wy = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    wz = np.zeros(shape=(nx+3,ny+3,nz+3),dtype='double')
    u[2:nx+3,2:ny+3,2:nz+3] = _q[2:nx+3,2:ny+3,2:nz+3,1]/_q[2:nx+3,2:ny+3,2:nz+3,0]
    v[2:nx+3,2:ny+3,2:nz+3] = _q[2:nx+3,2:ny+3,2:nz+3,2]/_q[2:nx+3,2:ny+3,2:nz+3,0]
    w[2:nx+3,2:ny+3,2:nz+3] = _q[2:nx+3,2:ny+3,2:nz+3,3]/_q[2:nx+3,2:ny+3,2:nz+3,0]

    # u_x
    aa = np.zeros(shape=(nx+3),dtype='double')
    bb = np.zeros(shape=(nx+3),dtype='double')
    for k in range(2,nz+3):
        for j in range(2,ny+3):
            for i in range(2,nx+3):
                aa[i] = u[i,j,k]
            bb = c6dp(aa,bb,dx,nx)
            for i in range(2,nx+3):
                ux[i,j,k] = bb[i]
    del aa,bb

    # u_y
    aa = np.zeros(shape=(ny+3),dtype='double')
    bb = np.zeros(shape=(ny+3),dtype='double')
    for i in range(2,nx+3):
        for k in range(2,nz+3):
            for j in range(2,ny+3):
                aa[j] = u[i,j,k]
            bb = c6dp(aa,bb,dy,ny)
            for j in range(2,ny+3):
                uy[i,j,k] = bb[j]
    del aa,bb

    # u_z
    aa = np.zeros(shape=(nz+3),dtype='double')
    bb = np.zeros(shape=(nz+3),dtype='double')
    for j in range(2,ny+3):
        for i in range(2,nx+3):
            for k in range(2,nz+3):
                aa[k] = u[i,j,k]
            bb = c6dp(aa,bb,dz,nz)
            for k in range(2,nz+3):
                uz[i,j,k] = bb[k]
    del aa,bb

    # v_x
    aa = np.zeros(shape=(nx+3),dtype='double')
    bb = np.zeros(shape=(nx+3),dtype='double')
    for k in range(2,nz+3):
        for j in range(2,ny+3):
            for i in range(2,nx+3):
                aa[i] = v[i,j,k]
            bb = c6dp(aa,bb,dx,nx)
            for i in range(2,nx+3):
                vx[i,j,k] = bb[i]
    del aa,bb

    # v_y
    aa = np.zeros(shape=(ny+3),dtype='double')
    bb = np.zeros(shape=(ny+3),dtype='double')
    for i in range(2,nx+3):
        for k in range(2,nz+3):
            for j in range(2,ny+3):
                aa[j] = v[i,j,k]
            bb = c6dp(aa,bb,dy,ny)
            for j in range(2,ny+3):
                vy[i,j,k] = bb[j]
    del aa,bb

    # v_z
    aa = np.zeros(shape=(nz+3),dtype='double')
    bb = np.zeros(shape=(nz+3),dtype='double')
    for j in range(2,ny+3):
        for i in range(2,nx+3):
            for k in range(2,nz+3):
                aa[k] = v[i,j,k]
            bb = c6dp(aa,bb,dz,nz)
            for k in range(2,nz+3):
                vz[i,j,k] = bb[k]
    del aa,bb

    # w_x
    aa = np.zeros(shape=(nx+3),dtype='double')
    bb = np.zeros(shape=(nx+3),dtype='double')
    for k in range(2,nz+3):
        for j in range(2,ny+3):
            for i in range(2,nx+3):
                aa[i] = w[i,j,k]
            bb = c6dp(aa,bb,dx,nx)
            for i in range(2,nx+3):
                wx[i,j,k] = bb[i]
    del aa,bb

    # w_y
    aa = np.zeros(shape=(ny+3),dtype='double')
    bb = np.zeros(shape=(ny+3),dtype='double')
    for i in range(2,nx+3):
        for k in range(2,nz+3):
            for j in range(2,ny+3):
                aa[j] = w[i,j,k]
            bb = c6dp(aa,bb,dy,ny)
            for j in range(2,ny+3):
                wy[i,j,k] = bb[j]
    del aa,bb

    # w_z
    aa = np.zeros(shape=(nz+3),dtype='double')
    bb = np.zeros(shape=(nz+3),dtype='double')
    for j in range(2,ny+3):
        for i in range(2,nx+3):
            for k in range(2,nz+3):
                aa[k] = w[i,j,k]
            bb = c6dp(aa,bb,dz,nz)
            for k in range(2,nz+3):
                wz[i,j,k] = bb[k]
    del aa,bb

    # compute Q-C: qc
    # absolute value of vorticity: wa
    w1[2:nx+3,2:ny+3,2:nz+3] = wy[2:nx+3,2:ny+3,2:nz+3] - vz[2:nx+3,2:ny+3,2:nz+3]
    w2[2:nx+3,2:ny+3,2:nz+3] = uz[2:nx+3,2:ny+3,2:nz+3] - wx[2:nx+3,2:ny+3,2:nz+3]
    w3[2:nx+3,2:ny+3,2:nz+3] = vx[2:nx+3,2:ny+3,2:nz+3] - uy[2:nx+3,2:ny+3,2:nz+3]

    qc[2:nx+3,2:ny+3,2:nz+3] = -0.5*(np.square(ux[2:nx+3,2:ny+3,2:nz+3])+uy[2:nx+3,2:ny+3,2:nz+3]*
                               vx[2:nx+3,2:ny+3,2:nz+3] + uz[2:nx+3,2:ny+3,2:nz+3]*wx[2:nx+3,2:ny+3,2:nz+3]
                               +vx[2:nx+3,2:ny+3,2:nz+3]*uy[2:nx+3,2:ny+3,2:nz+3]+np.square(vy[2:nx+3,2:ny+3,2:nz+3])+
                               vz[2:nx+3,2:ny+3,2:nz+3]*wy[2:nx+3,2:ny+3,2:nz+3]+wx[2:nx+3,2:ny+3,2:nz+3]*
                               uz[2:nx+3,2:ny+3,2:nz+3]+wy[2:nx+3,2:ny+3,2:nz+3]*vz[2:nx+3,2:ny+3,2:nz+3]+
                               np.square(wz[2:nx+3,2:ny+3,2:nz+3]))

    wa[2:nx+3,2:ny+3,2:nz+3] = np.sqrt(np.square(w1[2:nx+3,2:ny+3,2:nz+3]) + np.square(w2[2:nx+3,2:ny+3,2:nz+3]) +
                               np.square(w3[2:nx+3,2:ny+3,2:nz+3]))

    #  vorticity
    f = open("a_vorticity_"+str(ifile)+".plt", "w+")
    f.write('title="Zone_'+str(ifile)+'_Time_'+str(t)+'"''\n')
    f.write('variables = "x", "y", "z", "w1", "w2", "w3"''\n')
    f.write('zone T=Zone_'+str(ifile)+' i=          '+str(nx+1)+' j=          '+str(ny+1)+' k=          '+str(nz+1)+
            ' f=point''\n')

    output_array = np.zeros(((nx+1)*(ny+1)*(nz+1),6),dtype='double')
    it = 0
    for k in range(2,nz+3):
        for j in range(2,ny+3):
            for i in range(2,nx+3):
                output_array[it, 0] = x[i,j,k]
                output_array[it, 1] = y[i,j,k]
                output_array[it, 2] = z[i,j,k]
                output_array[it, 3] = w1[i,j,k]
                output_array[it, 4] = w2[i,j,k]
                output_array[it, 5] = w3[i,j,k]
                it = it + 1

    np.savetxt(f,output_array,fmt='%10.7e')
    f.close()
    del output_array, it

    #  QC
    f = open("a_Qcriterion_"+str(ifile)+".plt", "w+")
    f.write('title="Zone_'+str(ifile)+'_Time_'+str(t)+'"''\n')
    f.write('variables = "x", "y", "z", "Q", "w"''\n')
    f.write('zone T=Zone_'+str(ifile)+' i=          '+str(nx+1)+' j=          '+str(ny+1)+' k=          '+str(nz+1)+
            ' f=point''\n')

    output_array = np.zeros(((nx+1)*(ny+1)*(nz+1),5),dtype='double')
    it = 0
    for k in range(2,nz+3):
        for j in range(2,ny+3):
            for i in range(2,nx+3):
                output_array[it, 0] = x[i,j,k]
                output_array[it, 1] = y[i,j,k]
                output_array[it, 2] = z[i,j,k]
                output_array[it, 3] = qc[i,j,k]
                output_array[it, 4] = wa[i,j,k]
                it = it + 1

    np.savetxt(f,output_array,fmt='%10.7e')
    f.close()
    del output_array, it

    spectrum3d(nx,ny,nz,u,v,w,ifile)

    del u,v,w,qc,wa,w1,w2,w3,ux,uy,uz,vx,vy,vz,wx,wy,wz


def history(nx,ny,nz,_q,_te):
    _te = 0.
    _te = np.sum(0.5*(np.square(_q[3:nx+3,3:ny+3,3:nz+3,1]/_q[3:nx+3,3:ny+3,3:nz+3,0]) +
                      np.square(_q[3:nx+3,3:ny+3,3:nz+3,2]/q[3:nx+3,3:ny+3,3:nz+3,0]) +
                      np.square(_q[3:nx+3,3:ny+3,3:nz+3,3]/_q[3:nx+3,3:ny+3,3:nz+3,0])))
    return _te


if __name__ == "__main__":
    input1 = open('input_integers.txt',"r")
    nx, ny, ipr, nsnap, imu, imodel, idt, iwriteF = np.loadtxt(input1, dtype="int64", unpack=True)
    input2 = open('input_floats.txt',"r")
    ukhi, denr, cfl, nustab, tmax, gamma, dt, prt, re, pr, ma, Tref, alpha, ampy = np.loadtxt(input2, unpack=True)

    aveeddy = 1e-12
    if ipr == 1:
        ma = ukhi/np.sqrt(gamma*2.5)

    if ipr == 1:
        nz = nx
        lx = 1.
        ly = 1.*float(ny)/float(nx)
        lz = 1.

        x0 = -lx/2.
        y0 = -ly/2.
        z0 = -lz/2.
    else:
        ny = nx
        nz = nx
        lx = 2.*np.pi
        ly = 2.*np.pi
        lz = 2.*np.pi

        x0 = 0.
        y0 = 0.
        z0 = 0.

    dx = lx/float(nx)
    dy = ly/float(ny)
    dz = lz/float(nz)

    ui = np.zeros(shape=(nx+6,ny+6,nz+6,5),dtype='double')
    s = np.zeros(shape=(nx+6,ny+6,nz+6,5),dtype='double')

    xx = x0 - 0.5*dx + dx*np.arange(0.,float(nx+6))
    yy = y0 - 0.5*dy + dy*np.arange(0.,float(ny+6))
    zz = z0 - 0.5*dz + dz*np.arange(0.,float(nz+6))

    x, y, z = np.meshgrid(xx, yy, zz, indexing='ij')

    q = initialize(nx,ny,nz,x,y,z,ma,alpha,denr,ukhi,gamma,ampy)

    if cfl >= 1.:
        cfl = 1.
    if nsnap < 1:
        nsnap = 1
    iend = 0
    t = 0.
    ifile = 0
    iout = 0
    dtout = tmax/float(nsnap)
    tout = dtout
    te = 0.

    if idt == 0:
        nt = int(tmax/dt)
        if (nt % nsnap) != 0:
            nt = nt - (nt % nsnap) + nsnap
        dt = tmax/float(nt)
        nf = int(nt/nsnap)
    elif idt == 1:
        dt = timestep(nx,ny,nz,dx,dy,dz,cfl,dt,q,imu,gamma,ma,nustab,aveeddy,prt)
        nt = int(tmax/dt)
        if (nt % nsnap) != 0:
            nt = nt - (nt % nsnap) + nsnap
        dt = tmax/float(nt)
        nf = int(nt/nsnap)
    else:
        nt = 1000000000

    te = history(nx,ny,nz,q,te)
    te = te / (nx*ny*nz)
    te0 = te

    ar1 = []
    ar2 = []
    a = 1./3.
    b = 2./3.
    nt1 = 0
    clock_time_init = time.time()

    # time integration
    for n in range(1,nt+1):

        if idt == 0:
            if (n % nf) == 0:
                iout = 1
        elif idt == 1:
            if (n % nf) == 0:
                iout = 1
        else:
            dt = timestep(nx,ny,nz,dx,dy,dz,cfl,dt,q,imu,gamma,ma,nustab,aveeddy,prt)
            if (t+dt) >= tout:
                dt = tout - t
                tout = tout + dtout
                iout = 1
            if (t+dt) >= tmax:
                dt = tmax-t
                iend = 1

        t = t + dt

        #  TVDRK3
        #  Step 1
        s = rhs(nx,ny,nz,dx,dy,dz,q,s,gamma,Tref,re,pr,ma,imu,imodel)
        ui[3:nx+3,3:ny+3,3:nz+3,0:5] = q[3:nx+3,3:ny+3,3:nz+3,0:5] + dt*s[3:nx+3,3:ny+3,3:nz+3,0:5]

        ui = perbc_xyz(nx,ny,nz,ui)

        #   step 2
        s = rhs(nx,ny,nz,dx,dy,dz,ui,s,gamma,Tref,re,pr,ma,imu,imodel)
        ui[3:nx+3,3:ny+3,3:nz+3,0:5] = 0.75*q[3:nx+3,3:ny+3,3:nz+3,0:5] + 0.25*ui[3:nx+3,3:ny+3,3:nz+3,0:5] + \
                                       0.25*dt*s[3:nx+3,3:ny+3,3:nz+3,0:5]
        ui = perbc_xyz(nx,ny,nz,ui)

        #  step 3
        s = rhs(nx,ny,nz,dx,dy,dz,ui,s,gamma,Tref,re,pr,ma,imu,imodel)
        q[3:nx+3,3:ny+3,3:nz+3,0:5] = a*q[3:nx+3,3:ny+3,3:nz+3,0:5] + b*ui[3:nx+3,3:ny+3,3:nz+3,0:5] + \
                                      b*dt*s[3:nx+3,3:ny+3,3:nz+3,0:5]
        q = perbc_xyz(nx,ny,nz,q)
        #  print(q)

        if iout == 1:
            ifile = ifile + 1
            if iwriteF == 1:
                outputTec(nx,ny,nz,x,y,z,q,t,ifile,gamma)
                Qcriterion(nx,ny,nz,dx,dy,dz,x,y,z,q,t,ifile)
            iout = 0

        te = history(nx,ny,nz,q,te)
        te = te/ (nx*ny*nz)
        rate = -(te-te0)/dt
        te0 = te
        ar11 = t - dt*0.5
        ar22 = rate
        ar1.append(ar11)
        ar2.append(ar22)
        nt1 = 1 + nt1
        if iend == 1:
            nt = nt1
            break

    total_clock_time = time.time() - clock_time_init
    print('Total clock time=', total_clock_time)
    ar1 = np.array(ar1)
    ar2 = np.array(ar2)
    ar = np.zeros((nt,2),dtype='double')
    ar[:,0] = ar1
    ar[:,1] = ar2

    tke = open("a_rate.plt", "w+")
    tke.write('variables = "t", "e"''\n')
    np.savetxt(tke,ar,fmt='%10.7e')
    tke.close()

    cpu = open("a_cpu.txt", "w+")
    cpu.write('cpu time in seconds = ')
    cpu.write(str(total_clock_time))
    cpu.write('\n')
    cpu.write('cpu time in hours = ')
    cpu.write(str(total_clock_time/3600.))
    cpu.write('\n')
    cpu.write('final time step = ')
    cpu.write(str(dt))
    cpu.write('\n')
    cpu.write('average time step = ')
    cpu.write(str(tmax/float(n)))
    cpu.write('\n')
    cpu.write('number of time iterations = ')
    cpu.write(str(n))
    cpu.write('\n')
    cpu.close()












