import matplotlib.pyplot as plt
import numpy
import numpy as np
import matplotlib
from matplotlib import rcParams
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.linalg import solve
matplotlib.rcParams['text.usetex'] = True
np.set_printoptions(suppress=True)
from sympy import *
from pylab import *
import math
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)

#计算r值
def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    print("r：", SSR / SST, "r-squared：", (SSR / SST) ** 2)
    return


print(matplotlib.matplotlib_fname())
def runplt(size=None):
    plt.figure(figsize=(8,8))
    plt.title(r'The Composition diagram of Ethanol(Full reflux)')
    plt.ylabel(r'$y$',fontsize=15)
    plt.xlabel(r'$x$',fontsize=15)
    plt.axis([0,1,0,1])
    # plt.axis([])
    tick_params(direction='in')
    tick_params(top='on', bottom='on', left='on', right='on')
    return plt
print(matplotlib.matplotlib_fname())
runplt()
x = np.array([0,1.9,7.20,9.60,12.38,16.61,23.37,26.08,32.73,39.65,50.79,51.98,57.32,67.63,74.72,89.43,100])
x = x / 100
y = np.array([0,17,38.91,43.75,47.04,50.89,54.45,55.80,58.26,61.22,65.64,65.99,68.41,73.85,78.15,89.43,100])
y = y / 100
A = np.polyfit(x,y,8)
B = np.poly1d(A)
# plt.scatter(x,y,c='purple',marker='o',label='$Original \ Datas$',zorder=3)
plt.plot(np.arange(0,1,0.001),B(np.arange(0,1,0.001)),ls='-',linewidth=1.0,c='black',label=r'Balanced relationship',zorder=2)
plt.plot(np.arange(0,1,0.001),np.arange(0,1,0.001),ls='-',linewidth=1.0,c='dimgrey',label=r'Distillation section operation line',zorder=2)

def xw(nw):
    Ww = 25.555*nw - 34.0378
    xw = (Ww/46)/((Ww)/(46)+(1 - Ww)/(18))
    return xw


plt.vlines(xw(1.3610),0, xw(1.3610), colors = 'royalblue',linewidth=1.0)
plt.vlines(xw(1.3342),0, xw(1.3342), colors = 'royalblue',linewidth=1.0)
Ix = np.roots(B-xw(1.3610))
plt.text(xw(1.3610),0, r'$x_{\rm{D}}$',ha='left', va='top', fontsize=13)
plt.text(xw(1.3342),0, r'$x_{\rm{W}}$',ha='left', va='top', fontsize=13)

print(Ix)
plt.hlines(xw(1.3610),Ix[5].real, xw(1.3610), colors = 'r',linewidth=1.0)
plt.vlines(Ix[5].real,Ix[5].real,xw(1.3610), colors = 'r',linewidth=1.0)
Ix2 = np.roots(B-Ix[5].real)
plt.text((Ix[5].real+xw(1.3610))/2, xw(1.3610), '$1$',ha='center', va='bottom', fontsize=13)
print(Ix2)
plt.hlines(Ix[5].real,Ix2[7].real, Ix[5].real, colors = 'r',linewidth=1.0)
plt.vlines(Ix2[7].real,Ix2[7].real,Ix[5].real, colors = 'r',linewidth=1.0)
Ix3 = np.roots(B-Ix2[7].real)
plt.text((Ix2[7].real+Ix[5].real)/2, Ix[5].real, '$2$',ha='center', va='bottom', fontsize=13)

print(Ix3)
plt.hlines(Ix2[7].real,Ix3[7].real, Ix2[7].real, colors = 'r',linewidth=1.0)
plt.vlines(Ix3[7].real,Ix3[7].real,Ix2[7].real, colors = 'r',linewidth=1.0)
plt.text((Ix3[7].real+Ix2[7].real)/2,Ix2[7].real, '$3$',ha='center', va='bottom', fontsize=13)

d1 = Ix3[7].real-Ix2[7].real
d1 = -d1
d2 = Ix2[7].real - xw(1.3342)
print(1-(d1-d2)/d1 + 2)
plt.scatter(xw(1.3610),xw(1.3610),c='purple',marker='o',zorder=3)
plt.text(xw(1.3610)+0.01,xw(1.3610)+0.005, '$a$',ha='left', va='top', fontsize=13)

plt.scatter(xw(1.3342),xw(1.3342),c='purple',marker='o',zorder=3)
plt.text(xw(1.3342)+0.01,xw(1.3342)+0.005, '$c$',ha='left', va='top', fontsize=13)


plt.legend(loc='upper left')
plt.savefig('全回流.pdf',bbox_inches='tight')
plt.show()