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
    plt.axis([np.log10(400), np.log10(100000),-1.80, 0])
    # plt.axis([])
    return plt
print(matplotlib.matplotlib_fname())


#流量导入(L/h)
Q = [2000,1800,1600,1400,1200]
    #转换单位(m^{3}/s)
Q = np.array(Q)
Q = Q / 1000
Q = Q / 3600

#压差(单位是Pa)
p1 = [78.1,62.7,50.7,38.6,29.0]
p1 = np.array(p1)
p1 = p1 + 0.1
p2 = [76.4,61.5,48.1,37.5,28.2]
p2 = np.array(p2)
p2 = p2 + 0.1
p1 = p1 * 1000
p2 = p2 * 1000

#水温(单位是摄氏度)
t1 = [21.3,21.5,21.9,22.0,22.3]
t2 = [21.4,21.7,21.7,22.1,22.2]
t1 = np.array(t1)
t2 = np.array(t2)
t = t1 + t2
t = t / 2

#插值数据：
    #密度(单位是kg/m^{3})
rho21 = 997.9955
rho23 = 997.5415
rho = (rho23 - rho21)*(t - 23)
rho = rho + rho23
# print(rho)
    #粘度(单位是Pa s)
mu21 = 0.0009779
mu23 = 0.0009325
mu = (mu23 - mu21)*(t - 23)
mu = mu + mu23
# print(mu)

dp = 2 * p2 - p1


#横截面积(单位是m^{2})
d = 0.015
A = d*d * math.pi / 4

#流速(单位是m/s)
u = Q / A

#局部阻力引起的能量损失(单位为J/kg)
l = 1.2
h = dp / rho
#局部阻力系数(纯数)
s = 2 * dp / rho
s = s / u
s = s / u
print(dp)
print(np.average(s))
numpy.savetxt("new.csv", s, delimiter=',')




