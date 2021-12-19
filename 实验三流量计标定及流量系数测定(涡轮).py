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
    plt.title(r'The flowmeter calibration curve of $\ln{Vs} - \ln{f}$')
    plt.ylabel(r'$\ln{Vs} \ / \  \rm{m}^{3}\cdot \rm{s}^{-1}$')
    plt.xlabel(r'$\ln{f} \ / \rm{Hz}$')
    # plt.axis([0, 4.5,0.03, 0.07])
    # plt.axis([])
    return plt
print(matplotlib.matplotlib_fname())

def runplt2(size=None):
    plt.figure(figsize=(10,6))
    plt.title(r'The Relationship of $K - \ln{Re}$')
    plt.xlabel(r'$\ln{Re}$')
    plt.ylabel(r'$K \ / \ \rm{m}^{-3}$')
    plt.axis([8.5, 11.5,0, 330000])
    # plt.axis([])
    return plt
#频率计读数(单位是Hz)
f = [55,101,150,199,250,300,345,400,420,230]
f = np.array(f)

#时间(单位是s)
t1 = [120,66,44,33,27,22,19,16,15,28]
t1 = np.array(t1)
t2 = [120,68,45,34,27,23,19,16,16,28]
t2 = np.array(t2)
t = (t1+t2)/2
print(t)
#高度(单位是mm)
h = 200.00

#管道的面积
A1 = (math.pi * (26/1000) * (26/1000))/4
# print(A1)
#体积流量(单位是m^3/s)
Vs = (h * 0.1090) / t
Vs = Vs / 1000
print('体积流量')
print(Vs)

#流速(单位是m/s)
u = Vs / A1
print('流速')
print(u)

#雷诺数(纯数)
Re = (26.0/1000) * u * 1000 /(0.8937/1000)
print('雷诺数')
print(Re)

#仪表常数(单位是m^{-3})
K = f / Vs
print('仪表常数')
print(K)


#fig1的拟合
A = np.polyfit(np.log(f),np.log(Vs),1)
B = np.poly1d(A)
print(B)

#fig2的拟合
D = np.polyfit(np.log(Re),K,1)
E = np.poly1d(D)
print(E)


#fig1
plt=runplt()
plt.grid(zorder=0)
plt.scatter(np.log(f),np.log(Vs),c='purple',marker='o',label='original datas',zorder=3)
plt.plot(np.log(f),B(np.log(f)),ls='-',c='orange',label=r'$ \ln{V_{s}} =  1.012 \ln{f} - 12.69$',zorder=2)
plt.legend(loc='upper left')
plt.savefig('涡轮1.pdf')
plt.show()

# fig2
plt2 = runplt2()
plt.grid(zorder=0)
plt2.scatter(np.log(Re),K,c='purple',marker='o',label='original datas',zorder=3)
plt.plot(np.log(Re),E(np.log(Re)),ls='-',c='orange',label=r'$ \ln{R_{e}} =-3853 \ln{f} + 3.439\times 10^{5}$',zorder=2)
plt.legend(loc='lower left')
plt2.savefig('涡轮3.pdf',bbox_inches='tight')
plt2.show()

