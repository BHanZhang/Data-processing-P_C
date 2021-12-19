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
    plt.title(r'The Relationship Of $\frac{Nu}{Pr^{0.4}} - Re$')
    plt.ylabel(r'$\lg\frac{Nu}{Pr^{0.4}}$')
    plt.xlabel(r'$\lg{Re}$')
    # plt.axis([np.log10(400), np.log10(100000),-1.80, 0])
    # plt.axis([])
    return plt
print(matplotlib.matplotlib_fname())
runplt()
#压差计读数导入(Pa)
p = np.array([0.20,0.28,0.38,0.52,0.72,0.99,1.36,1.87,2.57,3.44])
p  = p + 0.01
# p = p * 1000
#温度导入(摄氏度)
t1 = np.array([39.4,39.5,40,40.4,40.9,41.5,42.2,43,44.2,45.5])
t2 = np.array([75.2,75.0,74.8,74.5,74.3,74.3,74.3,74.2,74.3,74.4])
tw = np.array([98.2,98.3,98.2,98.3,98.2,98.2,98.2,98.1,98.2,98.1])

tm = (t2 - t1)/np.log((tw - t1)/(tw - t2))
print(tm)
#换热面积的计算(m^2)
A = math.pi * 0.01925 * 1

#物理常数
rho35 = 1.145
rho40 = 1.127

Cp = 1007

lamda35 = 0.02625
lamda40 = 0.02662

mu35 = 1.895 * 0.1 * 0.1 * 0.1 * 0.1 * 0.1
mu40 = 1.918 * 0.1 * 0.1 * 0.1 * 0.1 * 0.1

rho = (rho40-rho35)*(tm-35)/(5) + rho35
rho1 = (rho40-rho35)*(t1-35)/(5) + rho35
lamda = (lamda40-lamda35)*(tm-35)/(5) + lamda35
mu = (mu40-mu35)*(tm-35)/(5) + mu35
Cp = 1007



#流量计算
#m^3 / h
Vt1 = 23.8*np.sqrt(p/rho1)
print('流量')
print(Vt1)
Vi = Vt1 *(273.15+tm)/(273.15+t1)
print(Vi)
u = Vi / 3600
print(Vi)
u = u / (0.25 * math.pi *0.01925 *0.01925)
Wi =    Vi * rho / 3600

#管内传热速率计算(W)
Q = Wi * Cp * (t2-t1)

#对流传热系数
alpha = Q/(tm*A)

#量纲1数群
Pr = Cp * mu /lamda
Nu = alpha * 0.01925 / lamda
Re = 0.01925 * u * rho / mu
Nu0 = Nu

print(u)
#压降
pj0 = np.array([0.25,0.31,0.40,0.50,0.66,0.86,1.12,1.49,1.98,2.58])
u0 = u
plt.scatter(np.log10(Re),np.log10(Nu/pow(Pr,0.4)),c='purple',marker='o',zorder=3)
A = np.polyfit(np.log10(Re),np.log10(Nu/pow(Pr,0.4)),1)
B = np.poly1d(A)
print(B)
plt.plot(np.log10(Re),B(np.log10(Re)),ls='-',c='orange',label=r'General Pipe : $\lg\frac{Nu}{Pr^{0.4}}  = 0.8859 \lg {Re} - 1.966$ ',zorder=2)



#压差计读数导入(Pa)
p = np.array([0.20,0.27,0.36,0.49,0.66,0.90,1.21,1.63,2.21,2.99])
p  = p + 0.01
# p = p * 1000
#温度导入(摄氏度)
t1 = np.array([41.3,41.0,40.9,41.0,41.2,41.5,42,42.6,43.3,44.5])
t2 = np.array([76.2,76.3,76.5,76.9,77.1,77.3,77.2,77.6,77.7,78])
tw = np.array([98.4,98.4,98.3,98.4,98.3,98.3,98.3,98.3,98.3,98.2])

tm = (t2 - t1)/np.log((tw - t1)/(tw - t2))
print(tm)
#换热面积的计算(m^2)
A = math.pi * 0.01925 * 1

#物理常数
rho35 = 1.145
rho40 = 1.127

Cp = 1007

lamda35 = 0.02625
lamda40 = 0.02662

mu35 = 1.895 * 0.1 * 0.1 * 0.1 * 0.1 * 0.1
mu40 = 1.918 * 0.1 * 0.1 * 0.1 * 0.1 * 0.1

rho = (rho40-rho35)*(tm-35)/(5) + rho35
rho1 = (rho40-rho35)*(t1-35)/(5) + rho35
lamda = (lamda40-lamda35)*(tm-35)/(5) + lamda35
mu = (mu40-mu35)*(tm-35)/(5) + mu35
Cp = 1007



#流量计算
#m^3 / h
Vt1 = 23.8*np.sqrt(p/rho1)
Vi = Vt1 *(273.15+tm)/(273.15+t1)
u = Vi / 3600
u = u / (0.25 * math.pi *0.01925 *0.01925)
Wi =    Vi * rho / 3600

#管内传热速率计算(W)
Q = Wi * Cp * (t2-t1)

#对流传热系数
alpha = Q/(tm*A)

#量纲1数群
Pr = Cp * mu /lamda
Nu = alpha * 0.01925 / lamda
Re = 0.01925 * u * rho / mu

pj = np.array([0.25,0.32,0.43,0.57,0.75,1.02,1.34,1.77,2.32,3.11])
plt.scatter(np.log10(Re),np.log10(Nu/pow(Pr,0.4)),c='blue',marker='o',zorder=3)
A = np.polyfit(np.log10(Re),np.log10(Nu/pow(Pr,0.4)),1)
B = np.poly1d(A)
print(B)
plt.plot(np.log10(Re),B(np.log10(Re)),ls='-',c='green',label=r'Strengthened Pipe : $\lg\frac{Nu}{Pr^{0.4}} =  1.026 \lg {Re} - 2.523$',zorder=2)

plt.legend(loc='upper left')
plt.savefig('强化普通.pdf')
plt.show()

plt.figure(figsize=(10,6))
plt.title(r'The Relationship Of $\frac{Nu}{Nu_{0}} - Re$')
plt.ylabel(r'$\lg\frac{Nu}{Nu_{0}}$')
plt.xlabel(r'$\lg{Re}$')
plt.scatter(np.log10(Re),np.log10(Nu/Nu0),c='purple',marker='o',zorder=3)
A = np.polyfit(np.log10(Re),np.log10(Nu/Nu0),1)
B = np.poly1d(A)
print(B)
plt.plot(np.log10(Re),B(np.log10(Re)),ls='-',c='orange',label=r'Fortification ratio : $\lg\frac{Nu}{Nu_{0}} = 0.0958 \lg {Re} - 0.3816$',zorder=2)
print(Nu/Nu0)
plt.legend(loc='upper left')
plt.savefig('强化比.pdf')
plt.show()

plt.figure(figsize=(10,6))
plt.title(r'The Relationship Of $\Delta p  - u$')
plt.ylabel(r'$\lg\Delta p$')
plt.xlabel(r'$\lg{u}$')

plt.scatter(np.log10(u),np.log10(pj),c='blue',marker='o',zorder=3)
A = np.polyfit(np.log10(u),np.log10(pj),1)
B = np.poly1d(A)
print(B)
plt.plot(np.log10(u),B(np.log10(u)),ls='-',c='green',label=r'Strengthened Pipe : $\lg\Delta p = 1.923 \lg {u}- 2.502$',zorder=2)

plt.scatter(np.log10(u0),np.log10(pj0),c='purple',marker='o',zorder=3)
A = np.polyfit(np.log10(u0),np.log10(pj0),1)
B = np.poly1d(A)
print(B)
plt.plot(np.log10(u0),B(np.log10(u0)),ls='-',c='orange',label=r'General Pipe : $\lg\Delta p = 1.697 \lg {u} - 2.305$',zorder=2)

plt.legend(loc='upper left')
plt.savefig('压降.pdf')
plt.show()
q = Nu/Nu0
