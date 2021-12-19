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
    plt.title(r'The Relationship Of $\lg{\lambda} - \lg{Re}$')
    plt.ylabel(r'$\lg{\lambda}$')
    plt.xlabel(r'$\lg{Re}$')
    plt.axis([np.log10(400), np.log10(100000),-1.80, -0.4])
    # plt.axis([])
    return plt
print(matplotlib.matplotlib_fname())


#流量导入(L/h)
Q = [10,14,20,30,40,50,60,80,100,160,200,280,360,400,600,800,1200,1600]
    #转换单位(m^{3}/s)
Q = np.array(Q)
Q = Q / 1000
Q = Q / 3600
print(Q)
#压差(单位是Pa)
p = [10*0.00981,14*0.00981,15*0.00981,21*0.00981,35*0.00981,53*0.00981,63*0.00981,101*0.00981,156*0.00981,282*0.00981,415*0.00981,842*0.00981,11.8-0.2,14.4-0.2,28.9-0.2,48.8-0.2,101.4-0.2,167-0.2]
p = np.array(p)
p = p * 1000

#水温(单位是摄氏度)
t = [20.4,20.5,20.5,20.6,20.6,20.7,20.7,20.7,20.8,20.8,20.8,20.9,20.9,20.9,20.9,20.9,21,21]
t = np.array(t)
#插值数据：
    #密度(单位是kg/m^{3})
rho20 = 998.2071
rho21 = 997.9955
rho = (rho21 - rho20)*(t - 21)
rho = rho + rho21
# print(rho)
    #粘度(单位是Pa s)
mu20 = 0.001002
mu21 = 0.0009779
mu = (mu21 - mu20)*(t - 21)
mu = mu + mu21
# print(mu)


#横截面积(单位是m^{2})
d = 0.0078
A = d*d * math.pi / 4

#流速(单位是m/s)
u = Q / A
print('u')
print(u)
#雷诺数(纯数)
Re = d * u * rho /(mu)

#直管摩擦系数(纯数)
l = 1.575
lemma = 2 * d * p / (l * rho * u *
                     u)
#层流拟合
A = np.polyfit(np.log10(Re)[0:4],np.log10(lemma)[0:4],1)
B = np.poly1d(A)
print(B)

#湍流拟合
D = np.polyfit(np.log10(Re)[8:],np.log10(lemma)[8:],2)
E = np.poly1d(D)
print(E)

print(Re)
print(lemma)
#fig1
plt=runplt()
plt.grid(zorder=0)
plt.scatter(np.log10(Re[0:5]),np.log10(lemma[0:5]),c='purple',marker='o',label='Hiryu District',zorder=3)
plt.scatter(np.log10(Re[5:8]),np.log10(lemma[5:8]),c='green',marker='o',label='Transition Area',zorder=3)
plt.scatter(np.log10(Re[8:]),np.log10(lemma[8:]),c='C0',marker='o',label='Turbulent Area',zorder=3)
# plt.plot(np.log(Re),B(np.log(p)),ls='-',c='orange',label=r'$ \ln{V_{s}} =  0.5196 \ln{\Delta p} - 8.504$',zorder=2)
plt.plot(np.log10(np.arange(440,2000)),B(np.log10(np.arange(440,2000))),ls='-',c='orange',zorder=2)
plt.plot(np.log10(np.arange(4000,90000)),E(np.log10(np.arange(4000,90000))),ls='-',c='orange',zorder=2)
plt.plot(np.log10(np.arange(2000,4000)),B(np.log10(np.arange(2000,4000))),ls='--',c='orange',zorder=2)
plt.plot(np.log10(np.arange(2000,4000)),E(np.log10(np.arange(2000,4000))),ls='--',c='orange',zorder=2)
plt.legend(loc='upper right')
plt.savefig('流体阻力1.pdf',bbox_inches='tight')
plt.show()



