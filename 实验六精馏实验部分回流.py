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
    plt.title(r'The Composition diagram of Ethanol(Partial reflux)')
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
#组分测定函数
def xw(nw):
    Ww = 25.555*nw - 34.0378
    xw = (Ww/46)/((Ww)/(46)+(1 - Ww)/(18))
    return xw

t = symbols('t')

#精馏段操作线的绘制
def C(t):
    c = (2 * t) / 3 + xw(1.3611) / 3
    return c
plt.plot(np.arange(0,1,0.001),C(np.arange(0,1,0.001)),ls='-',linewidth=1.0,c='coral',label=r'Distillation section operation line',zorder=2)

#q线的绘制
nF = 1.3448
xF = xw(nF)
tF = 27
    #汽化潜热
ry20 = 43910
ry40 = 42300
ry = ((ry40-ry20)/(20)) * (tF - 20) + ry20
rh20 = 44033.4
rh40 = 43219.8
rh = ((rh40-rh20)/(20)) * (tF - 20) + rh20
    #比热
cy20 = 109.4
cy40 = 117.5
cy = ((cy40-cy20)/(20)) * (tF - 20) + cy20
ch20 = 75.294
ch40 = 75.132
ch = ((ch40-ch20)/(20)) * (tF - 20) + ch20
c = xF * cy + (1-xF) * ch
r = xF * ry + (1-xF) * rh

q = (r + c*(78.4 - 27))/r
print('nnnn')
print(ry)
print(rh)
print(cy)
print(ch)
print(r)
print(c)
print(q)
def Q(t):
    y = (q/(q-1))*t - xF/(q-1)
    return y

A1 = np.polyfit(np.arange(0,1,0.001),C(np.arange(0,1,0.001)),1)
B1 = np.poly1d(A1)
A2 = np.polyfit(np.arange(0,1,0.001),Q(np.arange(0,1,0.001)),1)
B2 = np.poly1d(A2)
Ix = np.roots(B1-B2)
plt.scatter(Ix,B1(Ix),c='purple',marker='o',zorder=3)
plt.text(Ix+0.01,B1(Ix), '$d$',ha='left', va='top', fontsize=13)



plt.plot([0,1],[0,1],ls='-',linewidth=1.0,c='silver',zorder=2)
A3 = np.polyfit([0,1],[0,1],1)
B3 = np.poly1d(A3)
Ix1 = np.roots(B2-B3)
plt.scatter(Ix1,B3(Ix1),c='purple',marker='o',zorder=3)
plt.text(Ix1+0.01,B3(Ix1), '$e$',ha='left', va='top', fontsize=13)
plt.plot(np.arange(Ix1,1,0.001),Q(np.arange(Ix1,1,0.001)),ls='-',linewidth=1.0,c='dimgrey',label=r'$q-$ line',zorder=2)
plt.vlines(Ix1,0,Ix1, colors = 'dimgrey',linewidth=1.0)

#提馏段操作线的绘制
plt.vlines(xw(1.3611),0, C(xw(1.3611)), colors = 'royalblue',linewidth=1.0)
plt.vlines(xw(1.3340),0, xw(1.3340), colors = 'royalblue',linewidth=1.0)
plt.scatter(xw(1.3340),xw(1.3340),c='purple',marker='o',zorder=3)
plt.text(xw(1.3340)+0.01,xw(1.3340)+0.005, '$c$',ha='left', va='top', fontsize=13)

plt.scatter(xw(1.3611),C(xw(1.3611)),c='purple',marker='o',zorder=3)
plt.text(xw(1.3611)+0.01,C(xw(1.3611))+0.005, '$a$',ha='left', va='top', fontsize=13)

x = np.array([xw(1.3340),Ix[0].real])
y = np.array([xw(1.3340),B1(Ix[0].real)])
A4 = np.polyfit(x,y,1)
B4 = np.poly1d(A4)
plt.plot([xw(1.3340),Ix[0].real],[xw(1.3340),B1(Ix[0].real)],ls='-',linewidth=1.0,c='orangered',label=r'Stripping section operation line',zorder=2)
#理论塔板数的计算
Ix2 = np.roots(B-C(xw(1.3611)))

print(Ix2)
plt.text(xw(1.3611),0, r'$x_{\rm{D}}$',ha='left', va='top', fontsize=13)
plt.text(xw(1.3340),0, r'$x_{\rm{W}}$',ha='left', va='top', fontsize=13)
plt.hlines(C(xw(1.3611)),Ix2[5].real,C(xw(1.3611)), colors = 'r',linewidth=1.0,zorder=0)
plt.vlines(Ix2[5].real,C(Ix2[5].real),C(xw(1.3611)), colors = 'r',linewidth=1.0,zorder=0)
Ix3 = np.roots(B-C(Ix2[5].real))
plt.text((Ix2[5].real+C(xw(1.3611)))/2, C(xw(1.3611)), '$1$',ha='center', va='bottom', fontsize=13)

print(Ix3)
plt.hlines(C(Ix2[5].real),Ix3[7].real,Ix2[5].real, colors = 'r',linewidth=1.0,zorder=0)
plt.vlines(Ix3[7].real,B4(Ix3[7].real),C(Ix2[5].real), colors = 'r',linewidth=1.0,zorder=0)
Ix4 = np.roots(B-B4(Ix3[7].real))
plt.text((Ix3[7].real+Ix2[5].real)/2, C(Ix2[5].real), '$2$',ha='center', va='bottom', fontsize=13)

print(Ix4)
plt.hlines(B4(Ix3[7].real),Ix4[7].real,Ix3[7].real, colors = 'r',linewidth=1.0,zorder=0)
plt.vlines(Ix4[7].real,Ix4[7].real,B(Ix4[7].real), colors = 'r',linewidth=1.0,zorder=0)
plt.text((Ix4[7].real+Ix3[7].real)/2, B4(Ix3[7].real), '$3$',ha='center', va='bottom', fontsize=13)

plt.legend(loc='upper left')
plt.savefig('部分回流.pdf',bbox_inches='tight')
plt.show()
print(xw(1.3611))
print(xw(1.3448))
print(xw(1.3340))

d1 = -Ix4[7].real+B(Ix4[7].real)
d2 = B(Ix4[7].real) - xw(1.3340)
print(1-(d1-d2)/d1 + 2)