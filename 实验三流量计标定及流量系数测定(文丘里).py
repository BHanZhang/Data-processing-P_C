import matplotlib.pyplot as plt
import numpy
import numpy as np
import matplotlib
from matplotlib import rcParams
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.linalg import solve
np.set_printoptions(suppress=True)
from sympy import *
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
    plt.figure(figsize=(10,6))
    plt.title(r'The flowmeter calibration curve of $\ln{Vs} - \ln{\Delta p}$')
    plt.ylabel(r'$\ln{Vs} \ / \  \rm{m}^{3}\cdot \rm{s}^{-1}$')
    plt.xlabel(r'$\ln{\Delta p} \ / \rm{kPa}$')
    # plt.axis([0, 4.5,0.03, 0.07])
    # plt.axis([])
    return plt
print(matplotlib.matplotlib_fname())

def runplt2(size=None):
    plt.figure(figsize=(10,6))
    plt.title(r'The Relationship of $C - \ln{Re}$')
    plt.xlabel(r'$\ln{Re}$')
    plt.ylabel(r'$C$')
    plt.axis([9, 12,0, 1])
    # plt.axis([])
    return plt
#压差(单位是kPa)
p = [150*0.00981,220*0.00981,310*0.00981,420*0.00981,5.9,8.4,11.8,16.7,23.5,33.2,46.8,59.3]
p = np.array(p)

#时间(单位是s)
t1 = [89,73,61,51,42,36,30,25,21,17,15,13]
t1 = np.array(t1)
t2 = [90,74,62,51,43,36,30,25,21,18,15,13]
t2 = np.array(t2)
t = (t1+t2)/2

#高度(单位是mm)
h = 200.0

#文丘里缩脉处的面积(单位是m^2)
A0 = (math.pi * (15/1000) * (15/1000))/4
# print(A0)
#管道的面积
A1 = (math.pi * (26/1000) * (26/1000))/4
# print(A1)
#体积流量(单位是m^3/s)
Vs = (h * 0.1100) / t
Vs = Vs / 1000

# print(Vs)
#流速(单位是m/s)
u = Vs / A1
# print(u)

#雷诺数(纯数)
Re = (26.0/1000) * u * 1000 /(0.8937/1000)

# print(Re)
#流量计系数()
q = A0 *  np.sqrt((2 * p))
C = Vs / (q)
print(C)


#fig1的拟合
A = np.polyfit(np.log(p),np.log(Vs),1)
B = np.poly1d(A)
print(B)

#fig2的拟合
D = np.polyfit(np.log(Re),C,1)
E = np.poly1d(D)
print(E)



#fig1
plt=runplt()
plt.grid(zorder=0)
plt.scatter(np.log(p),np.log(Vs),c='purple',marker='o',label='original datas',zorder=3)
plt.plot(np.log(p),B(np.log(p)),ls='-',c='orange',label=r'$ \ln{V_{s}} =  0.5196 \ln{\Delta p} - 8.504$',zorder=2)
plt.legend(loc='upper left')
plt.savefig('文丘里1.pdf')
plt.show()

#fig2,3
plt2 = runplt2()
plt2.grid(zorder=0)
plt2.scatter(np.log(Re),C,c='purple',marker='o',label='original datas',zorder=3)
plt2.plot(np.log(Re),E(np.log(Re)),ls='-',c='orange',label=r'$ C =  0.03205 \ln{R_{e}} + 0.5121$',zorder=2)
plt2.legend(loc='lower left')
plt2.savefig('文丘里3.pdf',bbox_inches='tight')
plt2.show()

