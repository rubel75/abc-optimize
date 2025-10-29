#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
# here -u is needed to skip buffering
"""
Total energy minimization
"""

import numpy as np
import subprocess
import os
import sys
import re
import scipy
from scipy.optimize import minimize
import numdifftools
from numdifftools import Jacobian, Hessian
import time

def senseDftCode():
    # sense DFT code ("VASP" or "WIEN2k") and case folder (for "WIEN2k")
    pathcase = os.getcwd() # get path to case folder (WIEN2k)
    case = os.path.basename(pathcase) # case directory name (WIEN2k)
    if os.path.exists(case+".struct"): # WIEN2k if case.struct file present
        dftCode = 'WIEN2k'
        return dftCode, case
    elif os.path.exists("POSCAR"): # VASP if POSCAR file present
        dftCode = 'VASP'
        return dftCode, case
    else: # case.struct not present
        print("Cannot find neither", case+".struct", "nor POSCAR file")
        sys.exit(1)
    return None, case
# END def senseDftCode

def Etot_jac(x):
    # compute Jacobian of Etot(x)
    global finite_diff_rel_step # finite difference relative step size
    h = abs(x)*finite_diff_rel_step # finite difference absolute step
    jac = Jacobian(lambda x: Etot(x), step=h, method='central')(x).ravel()
    print("jac =", jac)
    return jac
# END def Etot_jac

def Etot_hess(x):
    # compute Hessian of Etot(x)
    global finite_diff_rel_step # finite difference relative step size
    h = abs(x)*finite_diff_rel_step # finite difference absolute step
    hess = Hessian(lambda x: Etot(x), step=h, method='central')(x)
    print("hess =", hess)
    return hess
# END def Etot_jac

def readE0FromHistory(x):
    # detrmine if
    global xhistory # store all values of [x] and associeated Etot
    global E0history
    N = len(E0history)
    for i in range(0, N):
        if all(xhistory[i,:] == x): # match [x] vector found
            print("The total energy for x=", x, "vas previosly calculated")
            print("in step", i+1, "and will be used once again")
            return E0history[i] # return previously calculated total energy
    return None # if match [x] is not found in the oprevious history
# END def readE0FromHistory

def readInitLatw2k(case):
    # check if case.struct is present
    global mode
    if os.path.exists(case+".struct"): # is file present?
        if os.stat(case+".struct").st_size == 0: # size is not zero?
            print(case+".struct file has 0 size.")
            sys.exit(1)
        pass # contunue
    else: # case.struct not present
        print(case+".struct file does not exist.")
        sys.exit(1)
    #read struct file
    fin = open(case+".struct", "rt")
    data = fin.readlines()
    fin.close()
    # read lattice parameters from line 4
    # compatible with the Fortran format FORMAT(6F10.7)
    a0 = data[3][0:10] # angles from the 4th line in struct file
    b0 = data[3][11:20]
    c0 = data[3][21:30]
    alpha0 = data[3][30:40]
    beta0 = data[3][40:50]
    gamma0 = data[3][50:60]
    if mode == 'aaa000': # cubic a=b=c=x[0], do not change angles
        x0 = [float(a0)]
    elif mode == 'aac000': # hexagonal or tetragonal
        # a=b=x[0], c=x[1], do not change angles
        x0 = [float(a0), float(c0)]
    elif mode == 'aaaAAA': # rhombohedral a=b=c=x[0], alpha=beta=gamma=x[1]
        x0 = [float(a0), float(alpha0)]
    elif mode == 'abc000': # orthorombic
        # a=x[0], b=x[1], c=x[2], do not change angles
        x0 = [float(a0), float(b0), float(c0)]
    elif mode == 'abc0B0': # monoclinic
        # a=x[0], b=x[1], c=x[2], beta=x[3], do not change other angles
        x0 = [float(a0), float(b0), float(c0), float(beta0)]
    elif mode == 'abcABG': # triclinic
        # a=x[0], b=x[1], c=x[2], alpha=x[3], beta=x[4], gamma=x[5]
        x0 = [float(a0), float(b0), float(c0), float(alpha0), float(beta0), \
              float(gamma0)]
    elif mode == 'ab000G': # specialty 2DM
        # a=x[0], b=x[1], gamma=x[2], do not change angles
        x0 = [float(a0), float(b0), float(gamma0)]
    else: # unknown mode
        print("mode =", mode, "is not implemented")
        sys.exit(1)
    return x0
# END def readInitLatw2k

def readInitLatVasp():
    # check if case.struct is present
    global mode
    #read struct file from previous calculation
    fname = "POSCAR"
    fin = open(fname, "rt")
    data = fin.readlines()
    fin.close() # POSCAR
    #old lattice vectors
    scale_factor = float(data[1])
    latv0 = np.zeros((3,3)) # init lattice vactor array
    for i in range(3):
        line = data[2+i] # lineas 3, 4, 5 in POSCAR/CONTCAR files
        line = line.split() # split lines
        if len(line) != 3:
            print("ERROR reading file", fname)
            print("line", 2+i, "=", line)
            print("expected 3 values separated by space(s)")
            sys.exit(1)
        for j in range(3):
            latv0[i,j] = float(line[j])
    #apply scale factor
    latv0 = latv0*scale_factor
    a0 = np.linalg.norm(latv0[0,:])
    b0 = np.linalg.norm(latv0[1,:])
    c0 = np.linalg.norm(latv0[2,:])
    cosalpha0 = np.dot(latv0[1,:], latv0[2,:])/(b0*c0)
    alpha0 = np.arccos(cosalpha0) # radians
    alpha0 = np.degrees(alpha0) # rad -> deg
    cosbeta0 = np.dot(latv0[0,:], latv0[2,:])/(a0*c0)
    beta0 = np.arccos(cosbeta0) # radians
    beta0 = np.degrees(beta0) # rad -> deg
    cosgamma0 = np.dot(latv0[0,:], latv0[1,:])/(a0*b0)
    gamma0 = np.arccos(cosgamma0) # radians
    gamma0 = np.degrees(gamma0) # rad -> deg
    if mode == 'aaa000': # cubic a=b=c=x[0], do not change angles
        x0 = [float(a0)]
    elif mode == 'aac000': # hexagonal or tetragonal
        # a=b=x[0], c=x[1], do not change angles
        x0 = [float(a0), float(c0)]
    elif mode == 'aaaAAA': # rhombohedral a=b=c=x[0], alpha=beta=gamma=x[1]
        x0 = [float(a0), float(alpha0)]
    elif mode == 'abc000': # orthorombic
        # a=x[0], b=x[1], c=x[2], do not change angles
        x0 = [float(a0), float(b0), float(c0)]
    elif mode == 'abc0B0': # monoclinic
        # a=x[0], b=x[1], c=x[2], beta=x[3], do not change other angles
        x0 = [float(a0), float(b0), float(c0), float(beta0)]
    elif mode == 'abcABG': # triclinic
        # a=x[0], b=x[1], c=x[2], alpha=x[3], beta=x[4], gamma=x[5]
        x0 = [float(a0), float(b0), float(c0), float(alpha0), float(beta0), \
              float(gamma0)]
    elif mode == 'ab000G': # specialty 2DM
        # a=x[0], b=x[1], gamma=x[2], do not change angles
        x0 = [float(a0), float(b0), float(gamma0)]
    else: # unknown mode
        print("mode =", mode, "is not implemented")
        sys.exit(1)
    return x0
# END def readInitLatVasp

def runDFT():
    if dftCode == "VASP":
        # command to be executed
        cmd = ["srun",
               "/home/rubel/VASP-intelmpi-2018-3-222/vasp.5.4.4/bin/vasp_std"]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError: # in case of error
            # try again
            time.sleep(5) # pause for 5 sec before executing the command again
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError: # stop if failed 2nd time
                sys.exit(1)
    elif dftCode == "WIEN2k":
        # interpolate charge density
        cmd = ["clmextrapol_lapw"]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError: # stop in case of error
            sys.exit(1)
        # minimize atomic positions
        # shell = T is needed to pass double quotes
        #cmd = ['min -I -j "run_lapw -I -fc 0.1 -i 60 -p"'] # min forces
        cmd = ['min -I -j "runsp_lapw -I -fc 0.1 -i 60 -eece -p"'] # min forces
        #cmd = ['run_lapw -I -fc 0.50 -min -p'] # min forces on fly
        #cmd = ['run_lapw -I -ec 0.00001 -cc 0.0001 -i 60 -p']
        #cmd = ['runsp_lapw -I -ec 0.00001 -cc 0.0001 -i 60 -p']
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError: # stop in case of error
            sys.exit(1)
        if 'min' in cmd[0]: # atomic positions were minimized?
            # calculate superposed density for relaxed structure
            # for clmextrapol of the next structure
            #cmd = ['x dstart -super -p']
            cmd = ['x dstart -super -p -up; x dstart -super -p -dn']
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError: # stop in case of error
                sys.exit(1)
    else:
        print("Unknown DFT code")
        print("dftCode =", dftCode)
        sys.exit(1)
    return
# END runDFT

def readE0vasp():
    fin = open("OSZICAR", "rt")
    allLines = fin.readlines()
    fin.close()
    lastLine = allLines[len(allLines)-1]
    # remove F=, E0=, d E =
    lastLine = lastLine.replace('F=', '')
    lastLine = lastLine.replace('E0=', '')
    lastLine = lastLine.replace('d E =', '')
    lastLine = lastLine.replace('\n', '')
    E0 = lastLine.split()
    E0 = float(E0[2])
    print("E0=",E0)
    return E0
# END def readE0vasp

def readE0w2k(case):
    # check if case.scf present
    if os.path.exists(case+".scf"): # is file present?
        if os.stat(case+".scf").st_size == 0: # size is not zero?
            print(case+".scf file has 0 size.")
            sys.exit(1)
        pass # contunue
    else: # case.struct not present
        print(case+".scf file does not exist.")
        sys.exit(1)
    # read case.scf file
    fin = open(case+".scf", "rt")
    allLines = fin.readlines()
    fin.close()
    # go through the file line-by-line
    for line in allLines:
        if re.match(":ENE", line): # find lines with ":ENE" pattern
            # e.g.
            # :ENE  : ********** TOTAL ENERGY IN Ry =        -9099.81489700
            line = line.split(sep='TOTAL ENERGY IN Ry =') # split into 2 parts
            E0 = line[1] # get 2nd part
            E0 = E0.replace('\n', '') # clean up end of line
            E0 = float(E0) # only last occurance E0 will remain
    print("E0=",E0)
    return E0
# END def readE0w2k

def prepStructvasp(x):
    #read struct file from previous calculation
    global mode
    if os.path.exists("CONTCAR"): # is file present?
        if os.stat("CONTCAR").st_size != 0: # size is not zero?
            fname = "CONTCAR"
        else: # use POSCAR if CONTCAR cannot be used
            print("CONTCAR file has 0 size. POSCAR will be used")
            fname = "POSCAR"
    else: # use POSCAR if CONTCAR cannot be used
        print("CONTCAR file is not present. POSCAR will be used")
        fname = "POSCAR"
    fin = open(fname, "rt")
    data = fin.readlines()
    fin.close() # CONTCAR/POSCAR
    #old lattice vectors
    scale_factor = float(data[1])
    latv0 = np.zeros((3,3)) # init lattice vactor array
    latv = np.zeros((3,3)) # new lattice vector array
    for i in range(3):
        line = data[2+i] # lineas 3, 4, 5 in POSCAR/CONTCAR files
        line = line.split() # split lines
        if len(line) != 3:
            print("ERROR reading file", fname)
            print("line", 2+i, "=", line)
            print("expected 3 values separated by space(s)")
            sys.exit(1)
        for j in range(3):
            latv0[i,j] = float(line[j])
    #apply scale factor
    latv0 = latv0*scale_factor
    # set scale factor to 1
    scale_factor = 1.0
    data[1] = '1.0\n ' # update scale factor = 1
    #determine orininal length of lattice vectors (norm)
    a0 = np.linalg.norm(latv0[0,:])
    b0 = np.linalg.norm(latv0[1,:])
    c0 = np.linalg.norm(latv0[2,:])
    # determine original angles
    cosalpha0 = np.dot(latv0[1,:], latv0[2,:])/(b0*c0)
    alpha0 = np.arccos(cosalpha0) # radians
    alpha0 = np.degrees(alpha0) # rad -> deg
    cosbeta0 = np.dot(latv0[0,:], latv0[2,:])/(a0*c0)
    beta0 = np.arccos(cosbeta0) # radians
    beta0 = np.degrees(beta0) # rad -> deg
    cosgamma0 = np.dot(latv0[0,:], latv0[1,:])/(a0*b0)
    gamma0 = np.arccos(cosgamma0) # radians
    gamma0 = np.degrees(gamma0) # rad -> deg
    #determine new lattice vectors using old ones as a basis
    if mode == 'aaa000': # cubic a=b=c=x[0], do not change angles
        if len(x) != 1: # check x == 1
            print("mode =", mode, "expected 1 variables in [x] =", x)
            sys.exit(1)
        a = x[0]
        b = x[0]
        c = x[0]
        #scale lattice vectors, do not change angles
        latv[0,:] = latv0[0,:]*a/a0
        latv[1,:] = latv0[1,:]*b/b0
        latv[2,:] = latv0[2,:]*c/c0
    elif mode == 'aac000': # hexagonal or tetragonal
        # a=b=x[0], c=x[1], do not change angles
        if len(x) != 2: # check x == 2
            print("mode =", mode, "expected 2 variables in [x] =", x)
            sys.exit(1)
        a = x[0]
        b = x[0]
        c = x[1]
        #scale lattice vectors, do not change angles
        latv[0,:] = latv0[0,:]*a/a0
        latv[1,:] = latv0[1,:]*b/b0
        latv[2,:] = latv0[2,:]*c/c0
    elif mode == 'aaaAAA': # rhombohedral a=b=c=x[0], alpha=beta=gamma=x[1]
        if len(x) != 2: # check x == 2
            print("mode =", mode, "expected 2 variables in [x] =", x)
            sys.exit(1)
        a = x[0]
        b = x[0]
        c = x[0]
        alpha = x[1]
        beta = x[1]
        gamma = x[1]
        cosalpha = np.cos(np.radians(alpha))
        cosbeta = np.cos(np.radians(beta))
        cosgamma = np.cos(np.radians(gamma))
        singamma = np.sin(np.radians(gamma))
        latv[0,0] = a # [a] vector // X
        latv[0,1] = 0
        latv[0,2] = 0
        latv[1,0] = b*cosgamma # [b] vector in X-Y plane
        latv[1,1] = b*singamma
        latv[1,2] = 0
        latv[2,0] = c*cosbeta # [c] vector
        latv[2,1] = c*(cosalpha-cosgamma*cosbeta)/singamma
        latv[2,2] = c*np.sqrt(1 - cosbeta**2 - \
                              ((cosalpha-cosgamma*cosbeta)/singamma)**2)
    elif mode == 'abc000': # orthorombic
        # a=x[0], b=x[1], c=x[2], do not change angles
        if len(x) != 3: # check x == 3
            print("mode =", mode, "expected 3 variables in [x] =", x)
            sys.exit(1)
        #new length (norm) of lattice vectors
        a = x[0]
        b = x[1]
        c = x[2]
        #scale lattice vectors, do not change angles
        latv[0,:] = latv0[0,:]*a/a0
        latv[1,:] = latv0[1,:]*b/b0
        latv[2,:] = latv0[2,:]*c/c0
    elif mode == 'abc0B0': # monoclinic
        # a=x[0], b=x[1], c=x[2], beta=x[3], do not change other angles
        if len(x) != 4: # check x == 4
            print("mode =", mode, "expected 4 variables in [x] =", x)
            sys.exit(1)
        pass
        #new length (norm) of lattice vectors and angle between [a] and [c]
        a = x[0]
        b = x[1]
        c = x[2]
        beta = x[3]
        #scale lattice vectors [a] and [b] (angle between them does not change)
        latv[0,:] = latv0[0,:]*a/a0
        latv[1,:] = latv0[1,:]*b/b0
        if latv[0,1] == 0 and latv[0,2] == 0 and \
                latv[1,0] == 0 and latv[1,2] == 0:
            # [a] is alinged with X and [b] is aligned with Y
            latv[2,0] = c*np.cos(np.radians(beta))
            latv[2,1] = 0
            latv[2,2] = c*np.sin(np.radians(beta))
        else:
            print("[a] vector", latv[0,:]," is not alligner with X-axis")
            print("[b] vector", latv[1,:]," is not alligner with Y-axis")
            sys.exit(1)
    elif mode == 'abcABG': # triclinic
        # a=x[0], b=x[1], c=x[2], alpha=x[3], beta=x[4], gamma=x[5]
        if len(x) != 6: # check x == 6
            print("mode =", mode, "expected 6 variables in [x] =", x)
            sys.exit(1)
        a = x[0]
        b = x[1]
        c = x[2]
        alpha = x[3]
        beta = x[4]
        gamma = x[5]
        cosalpha = np.cos(np.radians(alpha))
        cosbeta = np.cos(np.radians(beta))
        cosgamma = np.cos(np.radians(gamma))
        singamma = np.sin(np.radians(gamma))
        latv[0,0] = a # [a] vector // X
        latv[0,1] = 0
        latv[0,2] = 0
        latv[1,0] = b*cosgamma # [b] vector in X-Y plane
        latv[1,1] = b*singamma
        latv[1,2] = 0
        latv[2,0] = c*cosbeta # [c] vector
        latv[2,1] = c*(cosalpha-cosgamma*cosbeta)/singamma
        latv[2,2] = c*np.sqrt(1 - cosbeta**2 - \
                              ((cosalpha-cosgamma*cosbeta)/singamma)**2)
    elif mode == 'ab000G': # specialty 2DM
        # a=x[0], b=x[1], gamma=x[2], do not change angles
        if len(x) != 3: # check x == 3
            print("mode =", mode, "expected 3 variables in [x] =", x)
            sys.exit(1)
        a = x[0]
        b = x[1]
        gamma = x[2]
        # compute c, alpha, beta
        singamma = np.sin(np.radians(gamma))
        cosgamma = np.cos(np.radians(gamma))
        latv[0,0] = a # [a] vector // X
        latv[0,1] = 0
        latv[0,2] = 0
        latv[1,0] = b*cosgamma # [b] vector in X-Y plane
        latv[1,1] = b*singamma
        latv[1,2] = 0
        latv[2,0] = c0*cosbeta0 # [c] vector
        latv[2,1] = c0*(cosalpha0-cosgamma*cosbeta0)/singamma
        latv[2,2] = c0*np.sqrt(1 - cosbeta0**2 - \
                              ((cosalpha0-cosgamma*cosbeta0)/singamma)**2)
    else:
        print("VASP mode =", mode, "is not implemented")
        sys.exit(1)
    #replace lattice vectors in lines 3-5
    for i in range(3):
        data[2+i] = str(latv[i,0]) +" "+ str(latv[i,1]) +" "+ \
            str(latv[i,2]) + "\n"
    #write the struct file
    fin = open("POSCAR", "wt")
    fin.writelines(data)
    fin.close()
    return
# END def prepStructvasp

def prepStructw2k(x, case):
    global mode
    #read struct file
    fin = open(case+".struct", "rt")
    data = fin.readlines()
    fin.close()
    # prepare string with lattice parameters compatible with the Fortran
    # format FORMAT(6F10.7)
    a = data[3][0:10] # lattice parameters from the 4th line in struct file
    b = data[3][11:20]
    c = data[3][21:30]
    alpha = data[3][30:40] # angles from the 4th line in struct file
    beta = data[3][40:50]
    gamma = data[3][50:60]
    if mode == 'aaa000': # cubic a=b=c=x[0], do not change angles
        if len(x) != 1: # check x == 1
            print("mode =", mode, "expected 1 variables in [x] =", x)
            sys.exit(1)
        abc = '{:10.6f}{:10.6f}{:10.6f}'.format(x[0], x[0], x[0])
    elif mode == 'aac000': # hexagonal or tetragonal
        # a=b=x[0], c=x[1], do not change angles
        if len(x) != 2: # check x == 2
            print("mode =", mode, "expected 2 variables in [x] =", x)
            sys.exit(1)
        abc = '{:10.6f}{:10.6f}{:10.6f}'.format(x[0], x[0], x[1])
    elif mode == 'aaaAAA': # rhombohedral a=b=c=x[0], alpha=beta=gamma=x[1]
        if len(x) != 2: # check x == 2
            print("mode =", mode, "expected 2 variables in [x] =", x)
            sys.exit(1)
        abc = '{:10.6f}{:10.6f}{:10.6f}'.format(x[0], x[0], x[0])
        alpha = '{:10.6f}'.format(x[1])
        beta = '{:10.6f}'.format(x[1])
        gamma = '{:10.6f}'.format(x[1])
    elif mode == 'abc000': # orthorombic
        # a=x[0], b=x[1], c=x[2], do not change angles
        if len(x) != 3: # check x == 3
            print("mode =", mode, "expected 3 variables in [x] =", x)
            sys.exit(1)
        abc = '{:10.6f}{:10.6f}{:10.6f}'.format(x[0], x[1], x[2])
    elif mode == 'abc0B0': # monoclinic
        # a=x[0], b=x[1], c=x[2], beta=x[3], do not change other angles
        if len(x) != 4: # check x == 4
            print("mode =", mode, "expected 4 variables in [x] =", x)
            sys.exit(1)
        abc = '{:10.6f}{:10.6f}{:10.6f}'.format(x[0], x[1], x[2])
        beta = '{:10.6f}'.format(x[3])
    elif mode == 'abcABG': # triclinic
        # a=x[0], b=x[1], c=x[2], alpha=x[3], beta=x[4], gamma=x[5]
        if len(x) != 6: # check x == 6
            print("mode =", mode, "expected 6 variables in [x] =", x)
            sys.exit(1)
        abc = '{:10.6f}{:10.6f}{:10.6f}'.format(x[0], x[1], x[2])
        alpha = '{:10.6f}'.format(x[3])
        beta = '{:10.6f}'.format(x[4])
        gamma = '{:10.6f}'.format(x[5])
    elif mode == 'ab000G': # specialty 2DM
        # a=x[0], b=x[1], gamma=x[3]
        if len(x) != 3: # check x == 3
            print("mode =", mode, "expected 3 variables in [x] =", x)
            sys.exit(1)
        abc = '{:10.6f}{:10.6f}'.format(x[0], x[1]) + c
        gamma = '{:10.6f}'.format(x[2])
    ang = alpha + beta + gamma # join angle strings
    line4 = abc + ang + '\n' # join lattice parameters and angles
    #replace 4th line with new lattice parameters
    data[3] = line4
    #write the struct file
    fin = open(case+".struct", "wt")
    fin.writelines(data)
    fin.close()
    return
# END def prepStructw2k

def Etot(x):
    """
    Parameters
    ----------
    x[0:2] : array float64
        [0] a
        [1] b
        [2] c

    Returns
    -------
    E0 : float64
        Calculated DFT total energy (sigma -> 0)
    """
    global xhistory # store all values of [x] and associeated Etot
    global E0history
    print("x=", x)
    # check if Etot was previously calculated for the vector [x]
    E0 = readE0FromHistory(x) # E0 = None means no match found in the history
    if E0 != None: # match in the fistory found
        return E0 # reuse previous result and do not run DFT
    if dftCode == "VASP":
        prepStructvasp(x)
    elif dftCode == "WIEN2k":
        prepStructw2k(x, case)
    else:
        print("Unknown DFT code")
        print("dftCode =", dftCode)
        sys.exit(1)
    #execute DFT on Linux platform
    if os.name == 'posix': # Linux
        runDFT()
    #read total energy from DFT output & clean up (WIEN2k only)
    if dftCode == "VASP":
        E0 = readE0vasp()
    elif dftCode == "WIEN2k":
        E0 = readE0w2k(case)
        if os.name == 'posix': # Linux
            # save calculation to clean up
            # (-f will force to overide the previous save)
            subprocess.run(["save_lapw -f -d last_scf_mini"], shell=True)
            if os.path.exists(case+".scf_mini"):
                # if case.scf_mini file present
                # mv case.scf_mini last_scf_mini/
                subprocess.run(["mv "+case+".scf_mini last_scf_mini/"], \
                               shell=True)
            # clean up force mini history
            if os.path.exists(".minrestart"):
                subprocess.run(["rm", ".minrestart"])
            if os.path.exists(".min_hess"):
                subprocess.run(["rm", ".min_hess"])
            if os.path.exists(".minpair"):
                subprocess.run(["rm", ".minpair"])
            if os.path.exists(case+".scf_mini1"):
                subprocess.run(["rm "+case+".scf_mini1"], shell=True)
            if os.path.exists(case+".struct_last_min"):
                subprocess.run(["rm "+case+".struct_last_min"], shell=True)
            if os.path.exists(case+".tmpM*"):
                subprocess.run(["rm "+case+".tmpM*"], shell=True)
            if os.path.exists(case+".finM"):
                subprocess.run(["rm "+case+".finM"], shell=True)
    else:
        print("Unknown DFT code")
        print("dftCode =", dftCode)
        sys.exit(1)
    # store result in a file (append)
    fin = open("data.csv", "a+")
    #line = print(*[x, E0], sep=',') #str(a)+','+str(b)+','+str(c)+','+str(E0)+'\n'
    line = [str(element) for element in x] # conver [x] vector into a string
    line.append(str(E0)) # append total energy
    line = ",".join(line) # convert to CSV
    line = line + '\n' # add end of line
    fin.write(line)
    fin.close()
    # store history of argumets [x] and function [Etot] as a global variable
    if np.isnan(np.min(xhistory)) or np.isnan(np.min(E0history)):
        # xhistory and E0history are initialized with NaN values
        # if NaN's are found, this is the first function evaluation
        xhistory[0,:] = x
        E0history[0] = E0
    else: # subsequent function evaluations
        xhistory = np.append(xhistory, [x], axis=0)
        E0history = np.append(E0history, [E0], axis=0)
    return E0
# END def Etot

# MAIN ========================================================================

print("Python version:", sys.version)
print("Scipy version:", scipy.version.version)
print("Numpy version:", np.__version__)
print("Numdifftools version:", numdifftools.__version__)

# Modes:
# 'aaa000' -- cubic a=b=c=x[0], do not change angles
# 'aac000' -- hexagonal or tetragonal a=b=x[0], c=x[1], do not change angles
# 'aaaAAA' -- rhombohedral a=b=c=x[0], alpha=beta=gamma=x[1]
# 'abc000' -- orthorombic a=x[0], b=x[1], c=x[2], do not change angles
# 'abc0B0' -- monoclinic a=x[0], b=x[1], c=x[2], beta=x[3], do not change
#             other angles
# 'abcABG' -- triclinic a=x[0], b=x[1], c=x[2], alpha=x[3], beta=x[4],
#             gamma=x[5]
# Special modes:
# 'ab000G' -- for 2D materials a=x[0], b=x[1], gamma=x[2],
#             fixed: c, alpha, beta
mode = 'ab000G' # GLOBAL

# constants
eV_to_J = 1.602176634e-19 # 1eV = 1.602176634e-19 J
ang3_to_m3 = 1e-30 # 1 ang3 = 1e-30 m3
Pa_to_kbar = 1e-8 # 1 Pa = 1e-8 kbar
Ry_to_eV = 13.6056980659 # 1 Ry = 13.6056980659 eV
ang_to_bohr = 1.889725989 # 1 Ang = 1.889725989 bohr

# sense DFT code ("VASP" or "WIEN2k") and case folder (for "WIEN2k")
dftCode, case = senseDftCode()
print("DFT code =", dftCode)

# create empty file "data.csv"
if os.path.exists("data.csv"): # if file exists?
    print("Removing", "data.csv")
    os.remove("data.csv")
if os.name == 'posix': # Linux
    subprocess.run(["touch", "data.csv"])

# Init lattice param: x0 = [a[0], b[0], c[0]]
if dftCode == "VASP":
    x0 = readInitLatVasp() # Ang
elif dftCode == "WIEN2k":
    x0 = readInitLatw2k(case) # read lattice parameters from case.struct
else:
    print("Unknown DFT code")
    print("dftCode =", dftCode)
    sys.exit(1)
# minimize total energy starting with x0
# "xatol" sets the tolerance for lattice parameters
# "fatol" as a tollerance for the Etot optimization
#         (0.001 Ry = 0.014 eV = 160 K)
N = len(x0) # number of optimization parameters
print("Initial lattice parameters:", x0)
print("Optimization mode:", mode)
# initialize global vatiables to store intermediate argument and
# function values with None's
xhistory = np.full((1,N), None, dtype='float') # GLOBAL
E0history = np.full((1), None, dtype='float') # GLOBAL
# Minimize Etot with respect to [x] using one of the following methods

# Option (1): Nelder-Mead siplex algorithm
# fatol - function tollerance
# xatol - parameter tollerance
# initial simplex
strain0 = 0.01
initial_simplex = np.zeros((N+1, N)) # allocate initial simplex matrix
initial_simplex[0,:] = x0
for i in range(1, N+1):
    initial_simplex[i,:] = x0
    initial_simplex[i,i-1] = x0[i-1]*(1+strain0)
res = minimize(Etot, x0, method='Nelder-Mead', \
                options={'disp': True, 'xatol': 0.01, 'fatol': 0.005, \
                        'initial_simplex': initial_simplex, 'adaptive': True})
print("res =",res)


# Option (2): Conjugate gradient algorithm (needs Jacobian)
# res = minimize(Etot, x0, method='CG', jac='3-point', \
#                 options={'disp': True, 'eps': 0.01, \
#                          'finite_diff_rel_step': 0.01})

# # Option (3): Nearly exact trust-region algorithm (needs Jacobian & Hessian)
# # gtol - gradient tolerance
# finite_diff_rel_step = 0.01 # GLOBAL finite difference relative step 0.01 = 1%
# max_stress = 2 # kbar
# print("Max target stress =", max_stress, "kbar")
# x0sort = x0[0:3] # a, b, c only
# # sort array to get smalles lattice parameters to compute smallest area
# x0sort.sort()
# # stress = energy gradient/smallest area
# # min energy gradient = stress * smallest area
# gtol = max_stress*x0sort[0]*x0sort[1] # kbar*length2
# # units conversion
# if dftCode == "VASP":
#     # kbar -> J/m3 -> eV/Ang3
#     # units for [x0] = Ang
#     # final units for [gtol] = eV/Ang
#     gtol = (gtol/Pa_to_kbar)*(ang3_to_m3/eV_to_J)
#     print("Gradient tollerance [gtol] =", gtol, "eV/Ang")
# elif dftCode == "WIEN2k":
#     # kbar -> J/m3 -> eV/Ang3 -> Ry/bohr3
#     # units for [x0] = bohr
#     # final units for [gtol] = Ry/bohr
#     gtol = (gtol/Pa_to_kbar)*(ang3_to_m3/eV_to_J)/(Ry_to_eV*ang_to_bohr**3)
#     print("Gradient tollerance [gtol] =", gtol, "Ry/bohr")
# else:
#     print("Unknown DFT code")
#     print("dftCode =", dftCode)
#     sys.exit(1)
# res = minimize(Etot, x0, method='trust-exact', jac=Etot_jac, hess=Etot_hess, \
#                 options={'disp': True, 'initial_trust_radius': 0.3, \
#                         'max_trust_radius': 0.5, 'gtol': gtol})
# print("res =",res)
# # stress in the final structure
# a0 = res.x[0]
# b0 = res.x[1]
# c0 = res.x[2]
# volume = a0*b0*c0
# stress = -res.jac # -dE/dx -> force
# stress[0] = stress[0]/(b0*c0) # (dE/dx)/area -> stress
# stress[1] = stress[1]/(a0*c0)
# stress[2] = stress[2]/(a0*b0)
# if dftCode == "VASP":
#     stress = stress*eV_to_J/ang3_to_m3
#     print("Stress (Pa) =", stress)
#     stress = stress*Pa_to_kbar
#     print("Stress (kbar) =", stress)
# else:
#     print("Stress (energy/lenght3) =", stress)
