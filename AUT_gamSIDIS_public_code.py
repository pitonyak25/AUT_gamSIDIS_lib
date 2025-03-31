import os,sys
import numpy as np
import matplotlib.pyplot as plt
import pylab as py
import pandas as pd
from scipy.integrate import quad, fixed_quad, dblquad
from scipy import stats
import matplotlib.colors as colz
import matplotlib.cm as cms
import matplotlib.gridspec as gridspec
import matplotlib.colors as colz
import matplotlib.cm as cms
import itertools
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lhapdf

import matplotlib
from matplotlib import rcParams
# #matplotlib.use('Agg')
matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rcParams["text.usetex"] = True
matplotlib.rc('text.latex',preamble=r"\usepackage{amsmath}")


#LHAPDF --> JAM flavor conversion
#21: 0 (g)    2: 1 (u)   -2: 2 (ubar)   1: 3 (d)   -1: 4 (dbar)    3: 5 (s)    -3: 6 (sbar)
#                                       4: 7 (c)    -4: 8 (cbar)   5: 9 (b)    -5: 10 (bbar)
f1 = lhapdf.mkPDFs("CT18NLO")
f1Tperp = lhapdf.mkPDFs("JAM22-sivers_proton_lo_qbar") # Note: u, d, ubar=dbar flavors available

def p2n(p):
    p=np.copy(p[[0,3,4,1,2,5,6,7,8,9,10]])
    return p

def get_f1(x,Q2,target,rep): # This returns an array giving proton f1(x) for the various quark flavors
    f1arr=[]
    for i in [21,2,-2,1,-1,3,-3,4,-4,5,-5]:
        f1arr.append(f1[rep].xfxQ2(i,x,Q2)/x)
    if target == 'n': f1arr=p2n(np.array(f1arr))
    if target == 'He3' : f1arr=(2./3.) * np.array(f1arr) + (1./3.) * p2n(np.array(f1arr))
    return np.array(f1arr)

def get_f1Tperp(x,Q2,rep,target): # This returns an array giving proton f1Tperp(x) for the various quark flavors
    f1Tperparr=[]
    for i in [21,2,-2,1,-1,3,-3,4,-4,5,-5]:
        f1Tperparr.append(f1Tperp[rep].xfxQ2(i,x,Q2)/x)
    if target == 'n': f1Tperparr=p2n(np.array(f1Tperparr))
    return np.array(f1Tperparr)

def get_df1Tperp(x,Q2,rep,target,res=1e-6): # Returns an array with the df1Tperp(x)/dx
    # Computes x-values considering function domain for gradient calculation
    x_v = [x-res if x-res >=0 else 0, x, x+res if x+res <= 1 else 1]

    # Computes f1Tperp values, an ndarray containing the respective values for each quark at x
    y_v = np.array([get_f1Tperp(xx,Q2,rep,target) for xx in x_v])

    # Computes the gradient for the up and down quark and returns them with traditional JAM convention
    u = np.gradient(y_v[:,1],res)[1]
    ub = np.gradient(y_v[:,2],res)[1]
    d = np.gradient(y_v[:,3],res)[1]
    db = np.gradient(y_v[:,4],res)[1]
    s = np.gradient(y_v[:,5],res)[1]
    sb = np.gradient(y_v[:,6],res)[1]
    c = np.gradient(y_v[:,7],res)[1]
    cb = np.gradient(y_v[:,8],res)[1]
    b = np.gradient(y_v[:,9],res)[1]
    bb = np.gradient(y_v[:,10],res)[1]


    df1Tperparr = np.array([0,u,ub,d,db,s,sb,c,cb,b,bb])

    return df1Tperparr

def rad_polar(x,xp):
    radius = np.sqrt((x**2)+(xp**2))
    return radius

def phi_polar(x,xp):
    if x==xp and xp>0: _phi=0
    elif xp>x and x>0: _phi = -np.pi/4 + np.arctan(xp/x)
    elif x==0 and xp>0: _phi = np.pi/4
    elif x<0 and xp>=0: _phi = 3*np.pi/4 + np.arctan(xp/x)
    elif x<0 and xp<0: _phi = 3*np.pi/4 + np.arctan(xp/x)
    elif x==0 and xp<0: _phi = 5*np.pi/4
    elif x>0 and xp<0: _phi = 7*np.pi/4 + np.arctan(xp/x)
    elif x>0 and xp==0: _phi = 7*np.pi/4
    elif x>xp and xp>0: _phi = 7*np.pi/4 + np.arctan(xp/x)

    return _phi

def hside(input):
    if input >= 0:
        return 1
    elif input < 0:
        return 0

def f_tild(x,xp,delt,eps):
    term1 = ((1-(x**2))*(1-(xp**2))/((1-x*xp)**2))**delt
    term2 = (1-((x-xp)**2))**eps
    term3 = hside(1-np.abs(x))*hside(1-np.abs(xp))*hside(1-np.abs(x-xp))
    final = term1 * term2 * term3

    return final

def e(x,xp):
    factor1 = 2/(1+np.exp(-50*(1-x**2)**3))-1
    factor2 = 2/(1+np.exp(-50*(1-xp**2)**3))-1
    factor3 = 2/(1+np.exp(-50*(1-(x-xp)**2)**3))-1
    factor4 = np.heaviside(1-np.abs(x),1)*np.heaviside(1-np.abs(xp),1)*np.heaviside(1-np.abs(x-xp),1)

    final = factor1*factor2*factor3*factor4

    return final

def new_FFT_quark(x,xp,Q,delt,eps,A,rep,target,q): #note that q refers to the quark index in the original array
    a2 = A[0]
    a3 = A[1]
    a4 = A[2]
    a5 = A[3]
    a6 = A[4]
    a7 = A[5]
    r = rad_polar(x,xp)/np.sqrt(2)
    phip = phi_polar(x,xp)
    Q2 = Q**2

    term11 = (1/(2*np.pi))*(get_f1Tperp(r,Q2,rep,target)[q] + get_f1Tperp(r,Q2,rep,target)[q+1])
    term12 = 1 + a2[q]*(np.cos(2*phip)-1) + a4[q]*(np.cos(4*phip)-1) + a6[q]*(np.cos(6*phip)-1)
    term21 = (1/(2*np.pi))*(get_f1Tperp(r,Q2,rep,target)[q] - get_f1Tperp(r,Q2,rep,target)[q+1])
    term22 = np.cos(phip) + a3[q]*(np.cos(3*phip)-np.cos(phip)) + a5[q]*(np.cos(5*phip)-np.cos(phip)) + a7[q]*(np.cos(7*phip)-np.cos(phip))
    final = ((term11 * term12) + (term21 * term22))*e(x,xp)
    return final

def new_FFT_antiquark(x,xp,Q,delt,eps,A,rep,target,q): #note that q refers to the quark index in the original array
    final = new_FFT_quark(-xp,-x,Q,delt,eps,A,rep,target,q-1)

    return final

def new_FFT(x,xp,Q,delt,eps,A,rep,target):
    array = np.zeros(11)
    for i in range(5):
        array[2*i + 1] = new_FFT_quark(x,xp,Q,delt,eps,A,rep,target,(2*i +1))
    for i in range(6):
        array[2*i] = new_FFT_antiquark(x,xp,Q,delt,eps,A,rep,target,(2*i))
    return array

def get_FFT(x,xp,Q2,rep,target,delt, eps, A): # This returns an array giving proton F_FT(x,x') for the various quark flavors
    Q = np.sqrt(Q2)
    _FFT = new_FFT(x,xp,Q,delt,eps,A,rep,target)
    return _FFT

def new_GFT_quark(x,xp,Q,delt,eps,B,rep,target,q): #note that q refers to the quark index in the original array
    b1 = B[0]
    b2 = B[1]
    b3 = B[2]
    b4 = B[3]
    b5 = B[4]
    b6 = B[5]
    r = rad_polar(x,xp)/np.sqrt(2)
    phip = phi_polar(x,xp)
    Q2 = Q**2

    term11 = -(1/(np.pi))*(get_f1Tperp(r,Q2,rep,target)[q] + get_f1Tperp(r,Q2,rep,target)[q+1])
    term12 = b1[q]*np.sin(phip) + b3[q]*np.sin(3*phip) + b5[q]*np.sin(5*phip)
    term21 = -(1/(np.pi))*(get_f1Tperp(r,Q2,rep,target)[q] + get_f1Tperp(r,Q2,rep,target)[q+1])
    term22 = b2[q]*np.sin(2*phip) + b4[q]*np.sin(4*phip) + b6[q]*np.sin(6*phip)
    final = ((term11 * term12) + (term21 * term22))*e(x,xp)
    return final

def new_GFT_antiquark(x,xp,Q,delt,eps,B,rep,target,q): #note that q refers to the quark index in the original array
    final = -new_GFT_quark(-xp,-x,Q,delt,eps,B,rep,target,q-1)

    return final

def new_GFT(x,xp,Q,delt,eps,B,rep,target):
    array = np.zeros(11)
    for i in range(5):
        array[2*i + 1] = new_GFT_quark(x,xp,Q,delt,eps,B,rep,target,(2*i+1))
    for i in range(6):
        array[2*i] = new_GFT_antiquark(x,xp,Q,delt,eps,B,rep,target,(2*i))
    return array

def get_GFT(x,xp,Q2,rep,target,delt,eps,B): # This returns an array giving proton G_FT(x,x') for the various quark flavors
    Q = np.sqrt(Q2)
    _GFT = new_GFT(x,xp,Q,delt,eps,B,rep,target)
    return _GFT

# em coupling
alpha_em = 1/137.036

#array of (quark charges)^2
def eq2(mu):
    mc = 1.28
    mb = 4.18

    if mu > mc and mu < mb: eq2=np.array([0,4./9.,4./9.,1./9.,1./9.,1./9.,1./9.,4./9.,4./9.,(10**(-9)),(10**(-9))])
    elif mu > mc and mu > mb: eq2=np.array([0,4./9.,4./9.,1./9.,1./9.,1./9.,1./9.,4./9.,4./9.,1./9,1./9.])
    else: eq2=np.array([0,4./9.,4./9.,1./9.,1./9.,1./9.,1./9.,(10**(-9)),(10**(-9)),(10**(-9)),(10**(-9))])

    return eq2

#array of (quark charges)^4
def eq4(mu):
    return eq2(mu)**2

#array of (quark charges)^3
def eq3(mu):
    return eq2(mu)**1.5

def sigma_u_BH(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)

    sigma = (4/(beta*beta_p))*(2*(1-beta)+(beta**2)+(beta_p**2)+2*beta_p-2*xb*((alpha*(1+beta_p))-alpha_p*(1-beta))+2*(xb**2)*(alpha**2 +alpha_p**2))
    return sigma

def sigma_u_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    sigmaC = ((beta * beta_p)/(d1*d2*d3))*sigma_u_BH(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam)
    return sigmaC

def au_fxn(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    au = 4*d1*(beta + beta_p)*(2-beta*(2-beta)+beta_p*(2+beta_p))
    return au

def bu_fxn(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    bu1 = (beta-1)*beta*((beta-2)*beta +4)
    bu2 = -beta_p*(beta-2)*((beta-3)*beta -2) + (beta_p**2)*((beta-3)*beta -6) - (beta_p**3)*(beta+3)
    bu3 = beta*(3*(beta-2)*beta + 4)-beta_p*(beta-2)*((beta-1)*beta +2)+(beta_p)**2*(beta-3)*(beta-2)-(beta_p**3)*(beta-3)+(beta_p**4)
    bu = 4*alpha*(bu1 +bu2) + 4*alpha_p*bu3
    return bu

def cu_fxn(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    cu1 = 2*(1-beta)*beta + beta_p*(2+beta*(2-beta))+(beta_p**2)*(2+beta)
    cu2 = 2*(1-beta)*beta + beta_p*(2-beta*(2-beta)) + (beta_p**2)*(2-beta)
    cu3 = -beta*(2-beta*(2-beta))-beta_p*(2-(beta**2))-(beta_p**2)*(2-beta)-(beta_p**3)
    cu = (8*alpha**2)*cu1 + (8*alpha_p**2)*cu2 + (8*alpha*alpha_p)*cu3
    return cu

def du_fxn(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    du1 = alpha_p*(beta+beta_p*(1-beta)+(beta_p**2))- alpha*(beta*(1-beta)+beta_p*(1+beta))
    du = 8*((alpha**2)+(alpha_p**2))*du1
    return du

def sigma_u_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    au = au_fxn(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam)
    bu = bu_fxn(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam)
    cu = cu_fxn(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam)
    du = du_fxn(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam)
    sigmaI = -(au + bu*xb + cu*(xb**2) + du*(xb**3))/(beta*beta_p*d1*d2*d3)
    return sigmaI

def fBH(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    mu = Q
    f1h = get_f1(xb,mu**2,'p',0)
    array = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] + f1h[2*(i)+2]
        quark = sub * eq2(mu)[2*(i)+1]
        array[i-1] = quark
    return np.sum(array)

def fC(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    mu = np.sqrt((1-beta+beta_p)*(Q**2))
    f1h = get_f1(xb,mu**2,'p',0)
    array = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] + f1h[2*(i)+2]
        quark = sub * eq4(mu)[2*(i)+1]
        array[i-1] = quark
    return np.sum(array)

def fI(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    Qtild = np.sqrt((1-beta+beta_p)*(Q**2))
    mu = np.sqrt(Q*Qtild)
    f1h = get_f1(xb,mu**2,'p',0)
    array = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] - f1h[2*(i)+2]
        quark = sub * eq3(mu)[2*(i)+1]
        array[i-1] = quark
    return np.sum(array)

def equation_9(s,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    Q2 = (np.sqrt(s)*p_prime*np.exp(eta_prime))+(np.sqrt(s)*p_gam*np.exp(eta_gam)) - (2*p_prime*p_gam)*(np.cosh(eta_prime - eta_gam) - np.cos(phi_prime - phi_gam))
    Q = np.sqrt(Q2)
    prelim = (alpha_em**3)/(4*(np.pi**2)*s*(Q**4))
    BH = (sigma_u_BH(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam))*(fBH(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))
    C = (sigma_u_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam))*(fC(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))
    I = (sigma_u_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam))*(fI(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))
    final = prelim * (BH + C + I)
    return final

def Qtild2(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    qutild = ((1-beta+beta_p)*(Q**2))
    return qutild


def sigma_uthpp_phip_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (-16*(xb**2)*(alpha**2 + alpha_p**2))/(d1*(d3**2)*alpha)
    return final

def sigma_uthpp_phip_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (16*xb**2*(alpha**2 + alpha_p**2)*(d1*beta_p + d1*xb*alpha_p*beta_p + xb*alpha*(beta**2 - (1 + beta)*beta_p)))/(d1*d3**2*alpha*beta*beta_p)
    return final


def sigma_uthpp_phigam_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (-16*xb**3*(alpha - alpha_p)*(alpha**2 + alpha_p**2))/(d1*d3**2*alpha*(-1 + xb*(alpha - alpha_p)))
    return final

def sigma_uthpp_phigam_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (-16*xb**3*(alpha**2 + alpha_p**2)*(alpha*beta - alpha_p*beta_p))/(d3**2*alpha*beta*beta_p)
    return final

def sigma_uthpm_phip_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (-16*(d1**2 + 2*d1*xb*alpha_p*(1 + beta_p) + 2*xb**2*(alpha**2*(-1 + beta)*beta + alpha_p**2*(1 + beta_p)**2 - alpha*alpha_p*beta*(1 + 2*beta_p))))/((d1**2)*(d1 - d3)*(d3**2)*alpha)
    return final

def sigma_uthpm_phip_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    FINAL1 = (2*alpha*beta**2 - (-6*alpha_p + alpha*(2 + xb**2*alpha*alpha_p) + 2*(alpha + alpha_p)*beta)*beta_p +  6*alpha_p*beta_p**2)
    FINAL2 = (4*(-1 + beta)*beta**2 + xb*alpha_p*(1 + beta*(-1 + 12*beta))*beta_p + xb*alpha_p*beta_p**2)
    FINAL3 = (2 - beta + 2*beta_p + xb*alpha_p*(1 + beta_p))
    FINAL4 = (1 + beta + beta_p + 2*beta*beta_p + xb*alpha_p*(1 + 2*beta + beta_p + 3*beta*beta_p))
    final = (8*(2*d1**2*beta_p + d1*xb*FINAL1 + xb**2*(-4*xb*(alpha**3)*(-1 + beta)* (beta**2) + (alpha**2)*FINAL2 + 4*(alpha_p**2)*beta_p*(1 + beta_p)*FINAL3 - 4*alpha*alpha_p*beta_p*FINAL4)))/(d1*(d1 - d3)*d3**2*alpha*beta*beta_p)
    return final

def sigma_uthpm_phigam_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (16*xb*(d1**2*(-alpha + alpha_p) + 2*d1*xb*alpha_p**2*(1 + beta_p) + 2*xb*(-(xb*alpha**3*(-1 + beta)**2) + xb*alpha_p**3* (1 + beta_p)**2 + alpha**2*((-1 + beta)*(-1 + beta + xb*alpha_p*beta) + beta_p - beta*beta_p + xb*alpha_p*(-1 + 2*beta)*beta_p) - xb*alpha*alpha_p**2* (beta + beta_p + 2*beta*beta_p + beta_p**2))))/ (d1**2*(-1 + d1 - d3)*(d1 - d3)*d3**2*alpha)
    final1 = (16*xb*(d1**2*(-alpha + alpha_p) + 2*d1*xb*alpha_p**2*(1 + beta_p) + 2*xb*(-(xb*alpha**3*(-1 + beta)**2) + xb*alpha_p**3* (1 + beta_p)**2 + alpha**2*((-1 + beta)*(-1 + beta + xb*alpha_p*beta) + beta_p - beta*beta_p + xb*alpha_p*(-1 + 2*beta)*beta_p) - xb*alpha*alpha_p**2*(beta + beta_p + 2*beta*beta_p + beta_p**2))))/(d1**2*(-1 + d1 - d3)*(d1 - d3)*d3**2*alpha)
    return final


def sigma_uthpm_phigam_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    FINAL1 = (alpha*(1 + 2*xb*alpha*(-1 + xb*alpha))*(-1 + beta)**2*beta)
    FINAL2 = (alpha_p*(1 + 2*xb*alpha_p*(1 + xb*alpha_p)) - (alpha + alpha_p)*(2 - d1 + d3 + xb*(-alpha + alpha_p + 4*xb*alpha*alpha_p))*beta + (alpha_p + 2*alpha*(1 + xb*alpha*(-1 + 3*xb*alpha_p)))*beta**2)
    final = (16*xb*(-FINAL1 + FINAL2*beta_p + (-(alpha*beta) + 2*alpha_p*(1 - beta + xb*alpha_p*(2 + 2*xb*alpha_p - beta - 3*xb*alpha*beta)))*beta_p**2 + alpha_p*(1 + 2*xb*alpha_p*(1 + xb*alpha_p))*beta_p**3))/(d1*(d1 - d3)*d3**2*alpha*beta*beta_p)
    return final

def sigma_utSFPp_phip_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (-16*((-1 + beta)**2 + beta_p**2 - 2*xb*(alpha_p*(-1 + beta + beta_p) + beta_p*gamma) + xb**2*(2*alpha_p**2 + 2*alpha_p*gamma + gamma**2)))/(d1*gamma*(-1 + d1 - xb*gamma)*(1 + xb*(alpha_p + gamma)))
    return final

def sigma_utSFPp_phip_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (16*((-1 + beta)**2 + beta_p**2 - 2*xb*(alpha_p*(-1 + beta + beta_p) + beta_p*gamma) + xb**2*(2*alpha_p**2 + 2*alpha_p*gamma + gamma**2)))/(d1*beta_p*gamma*(1 + xb*(alpha_p + gamma)))
    return final

def sigma_utSFPp_phigam_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (-16*(1 + xb*gamma)*((-1 + beta)**2 + beta_p**2 - 2*xb*(alpha_p*(-1 + beta + beta_p) + beta_p*gamma) + xb**2*(2*alpha_p**2 + 2*alpha_p*gamma + gamma**2)))/(d1*xb*gamma**2*(-1 + d1 - xb*gamma)*(1 + xb*(alpha_p + gamma)))
    return final


def sigma_utSFPp_phigam_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (-16*(alpha_p*beta - alpha*beta_p)*(2 + xb**2*(alpha**2 + alpha_p**2) + (-2 + beta)*beta + beta_p*(2 + beta_p) - 2*xb*(alpha_p*(-1 + beta) + alpha*(1 + beta_p))))/(d1*beta*beta_p*gamma**2*(1 + xb*(alpha_p + gamma)))
    return final

def sigma_utSFPm_phip_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (16*(-(d1*(beta**2 + (1 + beta_p)**2)) + beta**2*(1 + xb*gamma) + (1 + beta_p)**2*(1 + xb*gamma) - 2*beta*(1 + xb*(alpha_p + gamma))))/(d1**2*gamma*(-1 + d1 - xb*gamma)*(1 + xb*(alpha_p + gamma)))
    return final

def sigma_utSFPm_phip_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (16*(-(((-1 + beta - beta_p)*(beta**2 + beta_p**2))/(beta_p*gamma)) - (xb*(beta**2 + (1 + beta_p)**2))/(1 + xb*(alpha_p + gamma))))/(d1*beta)
    return final

def sigma_utSFPm_phigam_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (16*(1 - 2*beta + beta**2 - 4*xb*alpha_p*beta_p + beta_p**2 + 2*xb*((-1 + beta)**2 + beta_p**2)*gamma - d1*(1 - 2*beta + beta**2 + beta_p**2 - 2*xb*alpha_p*(beta + beta_p) + xb*((-1 + beta)**2 + beta_p**2)*gamma) + xb**2*(2*alpha_p**2*(beta - beta_p) - 4*alpha_p*beta_p*gamma + ((-1 + beta)**2 + beta_p**2)*gamma**2)))/(d1**2*xb*gamma**2*(-1 + d1 - xb*gamma)*(1 + xb*(alpha_p + gamma)))
    return final

def sigma_utSFPm_phigam_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    d1 = 1 - beta + beta_p
    d2 = 1 - xb*(alpha - alpha_p)
    d3 = 1 - beta + beta_p -xb*(alpha - alpha_p)
    final = (-16*xb*(alpha*beta*((-1 + beta)**2 + beta_p**2) - alpha_p*beta_p*(beta**2 + (1 + beta_p)**2)))/(d1*beta*beta_p*gamma*(1 + xb*(alpha_p + gamma)))
    return final

def FNplusxxpC(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    xbp = (1 - beta + beta_p)/(alpha - alpha_p)
    mu = np.sqrt((1-beta+beta_p)*(Q**2))

    f1h = get_FFT(xb,xbp,mu**2,rep,'p',delt,eps,A)
    array1 = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] + f1h[2*(i)+2]
        quark = sub * eq4(mu)[2*(i)+1]
        array1[i-1] = quark
    g1h = get_GFT(xb,xbp,mu**2,rep,'p',delt,eps,B)
    array2 = np.empty(5)
    for i in range(5):
        sub = g1h[2*(i)+1] + g1h[2*(i)+2]
        quark = sub * eq4(mu)[2*(i)+1]
        array2[i-1] = quark
    return np.sum(array1) + np.sum(array2)

def FNminusxxpC(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    xbp = (1 - beta + beta_p)/(alpha - alpha_p)
    mu = np.sqrt((1-beta+beta_p)*(Q**2))
    f1h = get_FFT(xb,xbp,mu**2,rep,'p',delt,eps,A) - get_GFT(xb,xbp,mu**2,rep,'p',delt,eps,B)
    array = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] + f1h[2*(i)+2]
        quark = sub * eq4(mu)[2*(i)+1]
        array[i-1] = quark
    return np.sum(array)

def FNplusx0C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    xbp = (1 - beta + beta_p)/(alpha - alpha_p)
    mu = np.sqrt((1-beta+beta_p)*(Q**2))
    f1h = get_FFT(xb,(10**(-9)),mu**2,rep,'p',delt,eps,A) + get_GFT(xb,(10**(-9)),mu**2,rep,'p',delt,eps,B)
    array = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] + f1h[2*(i)+2]
        quark = sub * eq4(mu)[2*(i)+1]
        array[i-1] = quark
    return np.sum(array)

def FNminusx0C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    xbp = (1 - beta + beta_p)/(alpha - alpha_p)
    mu = np.sqrt((1-beta+beta_p)*(Q**2))
    f1h = get_FFT(xb,(10**(-9)),mu**2,rep,'p',delt,eps,A) - get_GFT(xb,(10**(-9)),mu**2,rep,'p',delt,eps,B)
    array = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] + f1h[2*(i)+2]
        quark = sub * eq4(mu)[2*(i)+1]
        array[i-1] = quark
    return np.sum(array)

def FNplusxxpI(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    xbp = (1 - beta + beta_p)/(alpha - alpha_p)
    Qtild = np.sqrt((1-beta+beta_p)*(Q**2))
    mu = np.sqrt(Q*Qtild)
    f1h = get_FFT(xb,xbp,mu**2,rep,'p',delt,eps,A) + get_GFT(xb,xbp,mu**2,rep,'p',delt,eps,B)
    array = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] - f1h[2*(i)+2]
        quark = sub * eq3(mu)[2*(i)+1]
        array[i-1] = quark
    return np.sum(array)

def FNminusxxpI(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    xbp = (1 - beta + beta_p)/(alpha - alpha_p)
    Qtild = np.sqrt((1-beta+beta_p)*(Q**2))
    mu = np.sqrt(Q*Qtild)
    f1h = get_FFT(xb,xbp,mu**2,rep,'p',delt,eps,A) - get_GFT(xb,xbp,mu**2,rep,'p',delt,eps,B)
    array = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] - f1h[2*(i)+2]
        quark = sub * eq3(mu)[2*(i)+1]
        array[i-1] = quark
    return np.sum(array)

def FNplusx0I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    xbp = (1 - beta + beta_p)/(alpha - alpha_p)
    Qtild = np.sqrt((1-beta+beta_p)*(Q**2))
    mu = np.sqrt(Q*Qtild)
    f1h = get_FFT(xb,(10**(-9)),mu**2,rep,'p',delt,eps,A) + get_GFT(xb,(10**(-9)),mu**2,rep,'p',delt,eps,B)
    array = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] - f1h[2*(i)+2]
        quark = sub * eq3(mu)[2*(i)+1]
        array[i-1] = quark
    return np.sum(array)

def FNminusx0I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    xbp = (1 - beta + beta_p)/(alpha - alpha_p)
    Qtild = np.sqrt((1-beta+beta_p)*(Q**2))
    mu = np.sqrt(Q*Qtild)
    f1h = get_FFT(xb,(10**(-9)),mu**2,rep,'p',delt,eps,A) - get_GFT(xb,(10**(-9)),mu**2,rep,'p',delt,eps,B)
    array = np.empty(5)
    for i in range(5):
        sub = f1h[2*(i)+1] - f1h[2*(i)+2]
        quark = sub * eq3(mu)[2*(i)+1]
        array[i-1] = quark
    return np.sum(array)

def sigma_ut_phip_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    t1 = (sigma_uthpp_phip_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNplusxxpC(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t2 = (sigma_uthpm_phip_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNminusxxpC(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t3 = (sigma_utSFPp_phip_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNplusx0C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t4 = (sigma_utSFPm_phip_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNminusx0C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    final = t1 + t2 + t3 + t4
    return final

def sigma_ut_phip_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    t1 = (sigma_uthpp_phip_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNplusxxpI(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t2 = (sigma_uthpm_phip_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNminusxxpI(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t3 = (sigma_utSFPp_phip_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNplusx0I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t4 = (sigma_utSFPm_phip_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNminusx0I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    final = t1 + t2 + t3 + t4
    return final

def sigma_ut_phigam_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    t1 = (sigma_uthpp_phigam_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNplusxxpC(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t2 = (sigma_uthpm_phigam_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNminusxxpC(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t3 = (sigma_utSFPp_phigam_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNplusx0C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t4 = (sigma_utSFPm_phigam_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNminusx0C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    final = t1 + t2 + t3 + t4
    return final

def sigma_ut_phigam_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    t1 = (sigma_uthpp_phigam_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNplusxxpI(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t2 = (sigma_uthpm_phigam_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNminusxxpI(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t3 = (sigma_utSFPp_phigam_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNplusx0I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    t4 = (sigma_utSFPm_phigam_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep))*(FNminusx0I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B))
    final = t1 + t2 + t3 + t4
    return final

def sigma_ut_phip(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    final = sigma_ut_phip_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B) + sigma_ut_phip_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B)
    return final

def sigma_ut_phigam(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    final = sigma_ut_phigam_C(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B) + sigma_ut_phigam_I(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B)
    return final

def ePLLsQ3(s,Q,phi_s,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    final = (1/2)*np.sqrt(alpha*alpha_p*(1-beta+beta_p))*np.sin(phi_s - phi_prime)
    return final

def ePLPsQ3(s,Q,phi_s,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep):
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    final = (1/2)*np.sqrt(alpha*beta*gamma)*np.sin(phi_s - phi_gam)
    return final

def equation21(s,phi_s,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B):
    M = 0.938
    Q2 = (np.sqrt(s)*p_prime*np.exp(eta_prime))+(np.sqrt(s)*p_gam*np.exp(eta_gam)) - (2*p_prime*p_gam)*(np.cosh(eta_prime - eta_gam) - np.cos(phi_prime - phi_gam))
    Q = np.sqrt(Q2)
    prelim = ((alpha_em**3)/(4*(np.pi**2)*s*(Q**4)))*((np.pi * M)/Q)
    f1 = ePLLsQ3(s,Q,phi_s,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep) * sigma_ut_phip(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B)
    f2 = ePLPsQ3(s,Q,phi_s,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep) * sigma_ut_phigam(s,Q,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B)
    final = prelim * (f1 + f2)
    return final

def AUTSIDIS(s,phi_s,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,A,B,delt=1,eps=1):
    M = 0.938
    Q2 = (np.sqrt(s)*p_prime*np.exp(eta_prime))+(np.sqrt(s)*p_gam*np.exp(eta_gam)) - (2*p_prime*p_gam)*(np.cosh(eta_prime - eta_gam) - np.cos(phi_prime - phi_gam))
    Q = np.sqrt(Q2)
    alpha = s/Q**2
    alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
    beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
    beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
    gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
    xb = 1/(alpha - alpha_p - gamma)
    final = equation21(s,phi_s,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep,delt,eps,A,B)/equation_9(s,p_prime,p_gam,eta_prime,eta_gam,phi_prime,phi_gam,rep)
    return final

def QGQ_Plot(func,coeffs,scen,Q2,delt,eps,rep,target,grain,cmap,levels):

    N = grain
    x = np.linspace(-0.8, 0.8, N)
    xp = np.linspace(-0.8, 0.8, N)
    X, XP = np.meshgrid(x, xp)

    #fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12, 10),dpi=150)
    fig, ((ax1,ax2)) = plt.subplots(1,2,figsize=(12, 5),dpi=150)
    fig.subplots_adjust(hspace=0.25)

    for flav in ['u','d']: #,'ubar','dbar']:
        if flav=='u':
            iflav=1
            if func=='FFT': label=r'$\boldsymbol{F_{FT}^u(x,x^{\prime})}$'
            elif func=='GFT': label=r'$\boldsymbol{G_{FT}^u(x,x^{\prime})}$'
            ax=ax1
        elif flav=='ubar':
            iflav=2
            if func=='FFT': label=r'$\boldsymbol{F_{FT}^{\bar{u}}(x,x^{\prime})}$'
            elif func=='GFT': label=r'$\boldsymbol{G_{FT}^{\bar{u}}(x,x^{\prime})}$'
            ax=ax3
        elif flav=='d':
            iflav=3
            if func=='FFT': label=r'$\boldsymbol{F_{FT}^d(x,x^{\prime})}$'
            elif func=='GFT': label=r'$\boldsymbol{G_{FT}^d(x,x^{\prime})}$'
            ax=ax2
        elif flav=='dbar':
            iflav=4
            if func=='FFT': label=r'$\boldsymbol{F_{FT}^{\bar{d}}(x,x^{\prime})}$'
            elif func=='GFT': label=r'$\boldsymbol{G_{FT}^{\bar{d}}(x,x^{\prime})}$'
            ax=ax4

        z = np.zeros_like(X)
        for i in range(N):
            for j in range(len(X[0])):
                if func=='FFT': z[i][j] = get_FFT(X[i][j],XP[i][j],Q2,rep,target,delt,eps,coeffs)[iflav]
                elif func=='GFT': z[i][j] = get_GFT(X[i][j],XP[i][j],Q2,rep,target,delt,eps,coeffs)[iflav]

        cs = ax.contourf(X, XP, z, cmap=cmap,levels=levels)
        ax.contour(X, XP, z, colors='k',linestyles='solid',linewidths=0.5,levels=levels)
        ax.axhline(0, linestyle='-',color='magenta',linewidth=0.35)
        ax.axvline(0, linestyle='-',color='magenta',linewidth=0.35)
        #ax.plot(x,xp,linestyle='-',color='red',linewidth=0.3)
        #ax.plot(-x,xp,linestyle='-',color='red',linewidth=0.3)

        ax.set_xlim([-0.8,0.8])
        ax.set_xticks([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8])
        ax.set_ylim([-0.8,0.8])

        ax.set_xlabel(r'\boldmath{$x$}', size = 14)
        ax.set_ylabel(r'\boldmath{$x^{\prime}$}', size = 14)
        ax.set_title(label)

        if flav=='u' and func=='FFT': ax.text(1,0.9,f'Scenario {scen}',fontsize=14,weight='bold')

        cbar = plt.colorbar(cs,ax=ax)

    plt.savefig(f'gallery/{func}_scen{scen}.pdf',bbox_inches='tight')
    plt.show()

def FFT_model_params(flav,Q2,rep,target,delt,eps,a46357,d2u,d2d,n):

    if flav=='u':iflav=1
    if flav=='d':iflav=3

    a3_model=np.insert(np.zeros(11),iflav,a46357[2])
    a4_model=np.insert(np.zeros(11),iflav,a46357[0])
    a5_model=np.insert(np.zeros(11),iflav,a46357[3])
    a6_model=np.insert(np.zeros(11),iflav,a46357[1])
    a7_model=np.insert(np.zeros(11),iflav,a46357[4])

    #calc A0
    a2=np.zeros(11)
    a3=np.zeros(11)
    a4=np.zeros(11)
    a5=np.zeros(11)
    a6=np.zeros(11)
    a7=np.zeros(11)

    coeffs=np.array([a2,a3,a4,a5,a6,a7])

    FFTxxp = np.vectorize(lambda xp,x: -get_FFT(x,xp,Q2,rep,target,delt,eps,coeffs)[iflav])
    FFTx = np.vectorize(lambda x: fixed_quad(lambda xp: FFTxxp(x,xp), -1., 1., n=n)[0])
    A0 = fixed_quad(FFTx, -1., 1., n=n)[0]

    #calc Ai
    Ai = np.zeros(8)
    for i in range(2,8):
        a2=np.zeros(11)
        a3=np.zeros(11)
        a4=np.zeros(11)
        a5=np.zeros(11)
        a6=np.zeros(11)
        a7=np.zeros(11)

        if i==2: a2[iflav]=1.
        elif i==3: a3[iflav]=1.
        elif i==4: a4[iflav]=1.
        elif i==5: a5[iflav]=1.
        elif i==6: a6[iflav]=1.
        elif i==7: a7[iflav]=1.

        coeffs=np.array([a2,a3,a4,a5,a6,a7])

        FFTxxp = np.vectorize(lambda xp,x: -get_FFT(x,xp,Q2,rep,target,delt,eps,coeffs)[iflav])
        FFTx = np.vectorize(lambda x: fixed_quad(lambda xp: FFTxxp(x,xp), -1., 1., n=n)[0])
        Ai[i] = -A0 + fixed_quad(FFTx, -1., 1., n=n)[0]

    if flav=='u': d2=d2u
    if flav=='d': d2=d2d

    a2flav=(d2 - (A0 + Ai[4]*a46357[0] + Ai[6]*a46357[1] + Ai[3]*a46357[2] + Ai[5]*a46357[3] + Ai[7]*a46357[4]))/Ai[2]

    a2_model=np.insert(np.zeros(11),iflav,a2flav)

    As_model = np.array([a2_model,a3_model,a4_model,a5_model,a6_model,a7_model])

    return As_model

def cartesianator(s,phi_s,pprim,etaprim,phi_prime,phi_gam,pgam_grain,etagam_grain,rep,pgamin,pgamax,etaga_min,etaga_max,As,Bs):
    pgam = np.linspace(pgamin,pgamax,pgam_grain)
    etagam = np.linspace(etaga_min,etaga_max,etagam_grain)
    masterlist = [pgam,etagam] ## ptprime, pt gamma, eta prime, eta gamma
    finallists = [list(tup) for tup in itertools.product(*masterlist)]
    points1 = []
    pgamfinals = []
    etagamfinals = []
#     xb_array=[]
#     xbp_array=[]
    for i in range(len(finallists)):

        p_prime = pprim
        p_gam = finallists[i][0]
        eta_prime = etaprim
        eta_gam = finallists[i][1]
        Q2 = (np.sqrt(s)*p_prime*np.exp(eta_prime))+(np.sqrt(s)*p_gam*np.exp(eta_gam)) - (2*p_prime*p_gam)*(np.cosh(eta_prime - eta_gam) - np.cos(phi_prime - phi_gam))
        Q = np.sqrt(Q2)

        alpha = s/Q**2
        alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
        beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
        beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
        gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
        xb = 1/(alpha - alpha_p - gamma)
        xbp = (1 - beta + beta_p)/(alpha - alpha_p)
        Qtild2 = ((1-beta+beta_p)*(Q**2))

        try:
            goodpoint= AUTSIDIS(s,phi_s,pprim,p_gam,etaprim,eta_gam,phi_prime,phi_gam,rep,As,Bs)
            if abs(goodpoint) <= 1 and Q2 > 1 and Qtild2 > 1 and (Q2 - Qtild2) > 1:
                points1.append(goodpoint)
                pgamfinals.append(finallists[i][0])
                etagamfinals.append(finallists[i][1])
#                 xb_array.append(xb)
#                 xbp_array.append(xbp)
            elif abs(goodpoint) > 1:
                pass
            elif math.isnan(goodpoint) == True:
                pass
        except:
            pass
#     plt.scatter(xb_array,xbp_array)
    return [etagamfinals, pgamfinals, points1]

def AUTSIDIS_plot(s,phi_s,phi_prime,phi_gam,pgam_grain,etagam_grain,rep,pgamin,pgamax,etaga_min,etaga_max,index,pprimslist,etaprimslist,vmin,vmax,As,Bs,scen):

        dictofpprim = {}
        pprims = pprimslist
        etaprims = etaprimslist
        masterlistp = [pprims,etaprims]
        primes = [list(tup) for tup in itertools.product(*masterlistp)]
        primedict = {}
        for i in range(len(primes)):
            etap = primes[i][1]
            ptp = primes[i][0]
            primedict[ptp,etap] = cartesianator(s,phi_s,ptp,etap,phi_prime,phi_gam,pgam_grain,etagam_grain,rep,pgamin,pgamax,etaga_min,etaga_max,As,Bs)

        fig = plt.figure(figsize=(12, 15),dpi=500)
        outer_outer_grid = fig.add_gridspec(3,2, height_ratios=[0.25,1.2,0.1], width_ratios=[0.1,1.2])
        outer_grid = outer_outer_grid[1,1].subgridspec(len(pprimslist), len(etaprimslist), wspace=0, hspace=0)
        outax = outer_outer_grid.subplots()
        outax[2,1].set_xticklabels(['$-2$','$-1$','$0$','$1$'],minor=True)
        etapaxis_bnd=0.935
        outax[2,1].set_xticks([etapaxis_bnd/4/2+j*(etapaxis_bnd/4) for j in range(4)],minor=True)
        outax[2,1].set_xticklabels(['','','','',''])
        outax[2,1].set_xticks([j*(etapaxis_bnd/4) for j in range(4+1)])
        outax[2,1].tick_params(axis='x',length=5)
        outax[2,1].tick_params(axis='x',length=0,which='minor')
        titleeta = outax[2,1].set_title(r'\boldmath{$\rm \eta^{ \prime }$}', y = 0.1, x = 0.5, size=16)

        pTpaxis_bnd=1
        outax[1,0].set_yticklabels([r'${pTprim}$'.format(pTprim=round(_pTprim,2)) for _pTprim in pprimslist],minor=True)
        outax[1,0].set_yticklabels(['' for j in range(len(pprimslist)+1)])
        outax[1,0].set_yticks([pTpaxis_bnd/len(pprimslist)/2+j*(pTpaxis_bnd/len(pprimslist)) for j in range(len(pprimslist))],minor=True)
        outax[1,0].set_yticks([j*(pTpaxis_bnd/len(pprimslist)) for j in range(len(pprimslist)+1)])
        outax[1,0].tick_params(axis='y',length=5)
        outax[1,0].tick_params(axis='y',length=0,which='minor')
        titlepT = outax[1,0].set_title(r'\boldmath{$ p_{T}^\prime \, (\rm GeV)$}', y = 0.5, x = 0.1, size = 16, rotation=90)


        outax[1,0].set_xticks([])
        outax[2,1].set_yticks([])
        fig.delaxes(outax[0,0])
        fig.delaxes(outax[0,1])
        outax[1,1].set_axis_off()
        outax[2,0].set_axis_off()

        outax[2,1].xaxis.set_tick_params(width=1, direction="in")
        outax[1,0].yaxis.set_tick_params(width=1, direction="in")

        outax[2,1].spines['top'].set_visible(False)
        outax[2,1].spines['right'].set_visible(False)
        outax[2,1].spines['left'].set_visible(False)
        outax[2,1].spines['bottom'].set_linewidth(1)
        outax[2,1].spines['bottom'].set_position(('outward',-40))
        outax[2,1].spines['bottom'].set_bounds(0,etapaxis_bnd)

        outax[1,1].spines['top'].set_visible(False)
        outax[1,1].spines['right'].set_visible(False)
        outax[1,1].spines['bottom'].set_visible(False)
        outax[1,1].spines['left'].set_visible(False)

        outax[1,0].spines['top'].set_visible(False)
        outax[1,0].spines['right'].set_visible(False)
        outax[1,0].spines['bottom'].set_visible(False)
        outax[1,0].spines['left'].set_linewidth(1)
        outax[1,0].spines['left'].set_position(('outward',-40))
        outax[1,0].spines['bottom'].set_bounds(0,pTpaxis_bnd)

        axs = outer_grid.subplots()
        axisindx = [tup for tup in itertools.product(range(len(pprimslist)),range(len(etaprimslist)))]

        norm = colz.Normalize(vmin = vmin,vmax = vmax)
        cm = cms.get_cmap("rainbow")
        for i in range(len(primedict)):
            etap = primes[i][1]
            ptp = primes[i][0]
            x,y,c= primedict[ptp,etap][0], primedict[ptp,etap][1], primedict[ptp,etap][2]
            axs[axisindx[i]].set_xticks([-1.5,0,1.5,3])
            axs[axisindx[i]].tick_params('x', direction='in')
            axs[axisindx[i]].set_xlim(-2.25,3.5)
            axs[axisindx[i]].set_ylim(-0.5,pgamax+2.5)
            im = None
            if x and y and c:
                im = axs[axisindx[i]].scatter(x,y, s=250/pgam_grain, c=np.abs(c), cmap = cm, norm=norm)
            else:
                axs[axisindx[i]].set_axis_off()

        rs = np.sqrt(s)
        for _indx in axisindx:
            if _indx in [(1,0),(2,0),(3,0),(0,1),(4,1),(5,1)] and rs<30: axs[_indx].set_yticks(np.linspace(1,pgamax,num = 4).tolist())
            elif _indx in [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0)] and rs>100: axs[_indx].set_yticks(np.linspace(1,pgamax,num = 4).tolist())
            elif _indx in [(1,0),(2,0),(3,0),(4,0),(5,0)] and rs>30 and rs<100: axs[_indx].set_yticks(np.linspace(1,pgamax,num = 4).tolist())
            else:
                axs[_indx].set_yticks(np.linspace(1,pgamax,num = 4).tolist())
                axs[_indx].set_yticklabels(['','','',''])



        axs[(3,0)].set_xticks([-1.5,0,1.5,3])
        axs[(3,0)].set_xticklabels([r'$-1.5$',r'$0$',r'$1.5$',r'$3.0$'])

        cbar=plt.colorbar(im, ax=axs, fraction=0.044, pad=0.025)
        cbar.set_label(r'\boldmath{$|A_{UT}|  $}', size = 18,labelpad=-35, y=1.05, rotation=0)

        if rs<30 or rs>100: axs[(5,2)].set_xlabel(r'\boldmath{$ \eta^{ \gamma } $}',fontsize=16)
        elif rs>20 and rs<100: axs[(5,2)].set_xlabel(r'\boldmath{$ \eta^{ \gamma } $}',y = 0, x = 0,fontsize=16)
        if rs<30 or rs>100: axs[(2,0)].set_ylabel(r'\boldmath{$ p_{T}^{\gamma}\,(\rm GeV)$}', fontsize=16)
        elif rs>20 and rs<100: axs[(3,0)].set_ylabel(r'\boldmath{$ p_{T}^{\gamma}\,(\rm GeV)$}', fontsize=16)

        rs = np.sqrt(s)

        if phi_prime == np.pi:
            fiprime = '\pi'
        else:
            fiprime = phi_prime

        if phi_gam == np.pi:
            figam = '\pi'
        else:
            figam = phi_gam

        sqrts = int(np.sqrt(s))
        text = r'$\boldsymbol{{ \sqrt{{s}} = {sz} \, \rm GeV,\ \phi ^{{\prime}} = {phip},\ \phi^{{\gamma}} = {phigam},\ \rm Scenario \ {zcen} }}$'.format(sz=sqrts,phip=fiprime,phigam=figam,zcen=scen)

        outax[1,0].text(4.35,1.02,text,size=16)

        plt.savefig(f"gallery/AUT_gamSIDIS_rs{round(rs,0)}_phigam{round(phi_gam,2)}_phip{round(phi_prime,2)}_scen{scen}.pdf",bbox_inches='tight')

def xBcartesianator(s,phi_s,pprim,etaprim,phi_prime,phi_gam,pgam_grain,etagam_grain,rep,pgamin,pgamax,etaga_min,etaga_max,As,Bs):
    pgam = np.linspace(pgamin,pgamax,pgam_grain)
    etagam = np.linspace(etaga_min,etaga_max,etagam_grain)
    masterlist = [pgam,etagam] ## ptprime, pt gamma, eta prime, eta gamma
    finallists = [list(tup) for tup in itertools.product(*masterlist)]
    points1 = []
    pgamfinals = []
    etagamfinals = []
    xb_arr=[]
    xbp_arr=[]
    for i in range(len(finallists)):

        p_prime = pprim
        p_gam = finallists[i][0]
        eta_prime = etaprim
        eta_gam = finallists[i][1]
        Q2 = (np.sqrt(s)*p_prime*np.exp(eta_prime))+(np.sqrt(s)*p_gam*np.exp(eta_gam)) - (2*p_prime*p_gam)*(np.cosh(eta_prime - eta_gam) - np.cos(phi_prime - phi_gam))
        Q = np.sqrt(Q2)

        alpha = s/Q**2
        alpha_p = (np.sqrt(s)*p_prime*np.exp(-eta_prime))/(Q**2)
        beta = (np.sqrt(s)*p_gam*np.exp(eta_gam))/(Q**2)
        beta_p = (2*p_gam*p_prime*(np.cosh(eta_gam - eta_prime) - np.cos(phi_gam - phi_prime)))/(Q**2)
        gamma = (np.sqrt(s)*p_gam*np.exp(-eta_gam))/(Q**2)
        xb = 1/(alpha - alpha_p - gamma)
        xbp = (1 - beta + beta_p)/(alpha - alpha_p)
        Qtild2 = ((1-beta+beta_p)*(Q**2))

        try:
            goodpoint= AUTSIDIS(s,phi_s,pprim,p_gam,etaprim,eta_gam,phi_prime,phi_gam,rep,As,Bs)
            if abs(goodpoint) <= 1 and Q2 > 1 and Qtild2 > 1 and (Q2 - Qtild2) > 1 and xb>0 and xb<1 and xbp>0 and xbp<1:
                points1.append(goodpoint)
                pgamfinals.append(finallists[i][0])
                etagamfinals.append(finallists[i][1])
                xb_arr.append(xb)
                xbp_arr.append(xbp)
            elif abs(goodpoint) > 1:
                pass
            elif math.isnan(goodpoint) == True:
                pass
        except:
            pass
#     plt.scatter(xb_arr,xbp_arr)
    return [etagamfinals, pgamfinals, xb_arr, xbp_arr, points1]

def xBxBtilde(s,phi_s,phi_prime,phi_gam,pgam_grain,etagam_grain,rep,pgamin,pgamax,etaga_min,etaga_max,index,pprimslist,etaprimslist,vmin,vmax,As,Bs,scen,binsize):

        dictofpprim = {}
        pprims = pprimslist
        etaprims = etaprimslist
        masterlistp = [pprims,etaprims]
        primes = [list(tup) for tup in itertools.product(*masterlistp)]
        primedict = {}
        for i in range(len(primes)):
            etap = primes[i][1]
            ptp = primes[i][0]
            primedict[ptp,etap] = xBcartesianator(s,phi_s,ptp,etap,phi_prime,phi_gam,pgam_grain,etagam_grain,rep,pgamin,pgamax,etaga_min,etaga_max,As,Bs)

        #print(primedict)

        fig, ((axs)) = plt.subplots(1,1,figsize=(4, 3),dpi=150)

        norm = colz.Normalize(vmin = vmin,vmax = vmax)
        cm = cms.get_cmap("rainbow")
        for i in range(len(primedict)):
            etap = primes[i][1]
            ptp = primes[i][0]
            x,y,c= primedict[ptp,etap][2], primedict[ptp,etap][3], primedict[ptp,etap][4]
            axs.set_xlim(0,0.8)
            axs.set_ylim(0,0.8)
            axs.set_yticks([0,0.2,0.4,0.6,0.8])
            im = plt.scatter(x,y, s=50/pgam_grain, c=np.abs(c), cmap = cm, norm=norm)
            plt.scatter(y,x, s=50/pgam_grain, c=np.abs(c), cmap = cm, norm=norm)


        #outax[1,2].set_axis_off()

        cbar=plt.colorbar(im, ax=axs, fraction=0.044, pad=0.025)
        cbar.set_label(r'\boldmath{$|A_{UT}|  $}', size = 12,labelpad=-25, y=1.1, rotation=0)

        rs = np.sqrt(s)

        if phi_prime == np.pi:
            fiprime = '\pi'
        else:
            fiprime = phi_prime

        if phi_gam == np.pi:
            figam = '\pi'
        else:
            figam = phi_gam

        sqrts = int(np.sqrt(s))

        axs.set_xlabel(r'\boldmath{$x_B$}',fontsize=14)
        axs.set_ylabel(r'\boldmath{$\tilde{x}_B$}',fontsize=14)
        text = r'$\boldsymbol{{ \sqrt{{s}} = {sz} \, \rm GeV,\ \phi ^{{\prime}} = {phip},\ \phi^{{\gamma}} = {phigam},\ \rm Scenario \ {zcen} }}$'.format(sz=sqrts,phip=fiprime,phigam=figam,zcen=scen)

        axs.text(-0.05,0.905,text,size=12)

        xB_lin = np.linspace(0,0.8,10)

        plt.plot(xB_lin,xB_lin,color='black',ls=':')

        plt.savefig(f"gallery/xB_xBtilde_rs{round(rs,0)}_phigam{round(phi_gam,2)}_phip{round(phi_prime,2)}_scen{scen}_bin{binsize}.pdf",bbox_inches='tight')
