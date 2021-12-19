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
    plt.figure(figsize=(12.5,8))
    plt.title(r'The relationship of $\Delta p- u$',fontsize=15)
    plt.ylabel(r'$\ln \Delta p \ / \rm{mmH}_{2}\rm{O}$',fontsize=15)
    plt.xlabel(r'$\ln u \ / \rm{m}\cdot\rm{s}^{-1}$',fontsize=15)
    # plt.axis([0,1,0,1])
    # plt.axis([])
    tick_params(direction='in')
    tick_params(top='on', bottom='on', left='on', right='on')
    return plt
print(matplotlib.matplotlib_fname())
runplt()

#流量计读数
V1 = np.array([2.5,5,7.5,10,12.5,15,17.5,20,22.5,23.5])
V2 = np.array([2.5,3,4,5,6,7,7.5,8,8.5,9])
V3 = np.array([2.5,3,4,5,5.5,6,6.5,7,7.5,8])

#压降读数
p1 = np.array([6,19,41,72,109,150,205,259,319,345])
p2 = np.array([16,20,33,53,78,130,150,337,406,442])
p3 = np.array([19,31,51,85,106,264,298,342,376,362])

#空气温度读数
t11 = np.array([23.5,26.2,29.5,31,32.4,33.8,34.6,36,37,38.6])
t12 = np.array([43.6,43.4,43.8,44.6,45.4,44,43.4,42.4,41.4,40.2])
t13 = np.array([46.2,46.8,47,47.5,47.9,48.1,47.9,47.8,47.5,47])

#水温度
t22 = np.array([13.3,13.3,13.4,13.4,13.4,13.3,13.3,13.6,13.7,14.4])
t23 = np.array([13.3,13.3,13.4,13.4,13.3,13.4,13.4,13.4,13.2,13.2])

def V2b(x,t):
    x = x / 3600
    t = t + 273.15
    V0B = x * (273.17/101) * np.sqrt(((101 * 101)/(t * 293.15)))
    return V0B

def u(x,t):
    V0B = V2b(x,t)
    omega = math.pi * 0.25 * 0.08 * 0.08
    u = V0B / omega
    return u


plt.scatter(np.log(u(V1,t11)),np.log(p1),c='tab:brown',label = 'Dry Tower Scatters',marker='o',zorder=3)
plt.scatter(np.log(u(V2,t12)),np.log(p2),c='royalblue',label = r'Water flow rate : $60\rm{L}\cdot\rm{h}^{-1}$',marker='o',zorder=3)
plt.scatter(np.log(u(V3,t13)),np.log(p3),c='orangered',label = r'Water flow rate : $80\rm{L}\cdot\rm{h}^{-1}$',marker='o',zorder=3)
A = np.polyfit(np.log(u(V1,t11)),np.log(p1),1)
B = np.poly1d(A)
print(B)
x = np.log(u(V1,t11))
y_err = x.std() * np.sqrt(1/len(x) +
                          (x - x.mean())**2 / np.sum((x - x.mean())**2))
plt.fill_between(x, B(x) - y_err, B(x) + y_err, alpha=0.2)
plt.plot(x,B(x),ls='-',label=r'Dry Tower : $\ln \Delta p  = 1.852 \ln u + 5.54$',linewidth=1.0,zorder=2)

tick_params(direction='in')
tick_params(top='on',bottom='on',left='on',right='on')


A = np.polyfit(np.log(u(V2,t12))[0:7],np.log(p2)[0:7],1)
B = np.poly1d(A)
print(B)
x = np.log(u(V2,t12))[0:7]
plt.plot(x,B(x),c='saddlebrown',ls='-',label=r'Before loading point : $\ln \Delta p  = 2.071 \ln u + 6.964 \  (60\rm{L}\cdot\rm{h}^{-1})$',linewidth=1.0,zorder=2)

A = np.polyfit(np.log(u(V2,t12))[6:10],np.log(p2)[6:10],1)
B = np.poly1d(A)
print(B)
x = np.log(u(V2,t12))[6:10]
plt.plot(np.log(u(V2,t12))[5:10],B(np.log(u(V2,t12))[5:10]),c='chocolate',ls='-',label=r'After loading point : $\ln \Delta p  = 5.553 \ln u + 10.7\  (60\rm{L}\cdot\rm{h}^{-1})$',linewidth=1.0,zorder=2)


A = np.polyfit(np.log(u(V3,t13))[0:5],np.log(p3)[0:5],1)
B = np.poly1d(A)
print(B)
x = np.log(u(V2,t12))[0:5]
plt.plot(x,B(x),ls='-',c='forestgreen',label=r'Before loading point : $\ln \Delta p  =2.125 \ln u + 7.426 \ (80\rm{L}\cdot\rm{h}^{-1})$',linewidth=1.0,zorder=2)

A = np.polyfit(np.log(u(V3,t13))[4:10],np.log(p3)[4:10],1)
B = np.poly1d(A)
print(B)
x = np.log(u(V3,t13))[4:10]
plt.plot(np.log(u(V3,t13))[3:10],B(np.log(u(V3,t13))[3:10]),c='limegreen',ls='-',label=r'After loading point : $\ln \Delta p  =2.877 \ln u + 8.792\ (80\rm{L}\cdot\rm{h}^{-1})$',linewidth=1.0,zorder=2)


plt.grid(zorder=0)

plt.legend(loc='lower right')
plt.savefig('填料吸收.pdf',bbox_inches='tight')
plt.show()
numpy.savetxt("new.csv", [V1,p1,t11,V2b(V1,t11),u(V1,t11)], delimiter=',')
numpy.savetxt("new1.csv", [V2,p2,t12,t22,V2b(V2,t12),u(V2,t12)], delimiter=',')
numpy.savetxt("new2.csv", [V3,p3,t13,t23,V2b(V3,t13),u(V3,t13)], delimiter=',')
