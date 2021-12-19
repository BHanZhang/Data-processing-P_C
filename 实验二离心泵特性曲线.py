import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
from pathlib import Path
from scipy.interpolate import make_interp_spline
np.set_printoptions(suppress=True)



#压力导入(Pa)
#压力表读数
p1 = [0.220,0.195,0.185,0.175,0.165,0.155,0.145,0.135,0.125,0.115,0.105,0.095,0.085,0.075,0.070]
p1 = np.array(p1)
#真空表读数
p2 = [0,0,0,0,-0.004,-0.008,-0.012,-0.016,-0.020,-0.024,-0.026,-0.030,-0.034,-0.038,-0.040]
p2 = np.array(p2)
p = p1 - p2
p = p * 1000
p = p * 1000
#流量导入(Pa)
Q = [-1.1,0.1,1.3,2.5,3.7,4.9,6.1,7.3,8.5,9.7,10.9,12.1,13.3,14.4,15.1]
Q = np.array(Q)
Q = Q + 1.1
Q = Q * 1000
#单位转换(m^3/h)
Qe = 0.0742 * pow(Q,0.4958)
Qe = Qe / 3600
#功率导入(kW)
P = [0.32,0.44,0.49,0.53,0.57,0.59,0.61,0.62,0.64,0.65,0.66,0.67,0.67,0.67,0.67]
P = np.array(P)
#水温导入(摄氏度)
t = [20.8,20.9,20.9,21,21,21.1,21.2,21.3,21.4,21.5,21.6,21.6,21.7,21.8,21.8]
t = np.array(t)
#内插法确定密度
rho20 = 998.2071
rho22 = 997.7735
rho = ((rho22 - rho20)/2)*(t - 22) + rho22
#压头(m)
H = p/(rho*9.8)
#扬程(m)
He = H + 0.18
#有效功率Ne(kW)
Ne = He * Qe * rho / 102
#轴功率
N = (P*0.6)
#效率
eta = Ne / N
eta = eta * 100
print(N)
print(eta)

fpath = Path(mpl.get_data_path(), "/System/Library/Fonts/Supplemental/Times New Roman.ttf")


host = host_subplot(111, axes_class=axisartist.Axes)
plt.subplots_adjust(right=0.75)
host.axis["top"].set_visible(False)
par1 = host.twinx()
par2 = host.twinx()

par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(45, 0))

par1.axis["right"].toggle(all=True)
par1.axis["right"].label.set_color('r')
par1.axis["right"].line.set_color('r')
par1.axis["right"].major_ticks.set_color('r')
par1.axis["right"].major_ticklabels.set_color('r')
par2.axis["right"].toggle(all=True)
par2.axis["right"].label.set_color('b')
par2.axis["right"].line.set_color('b')
par2.axis["right"].major_ticks.set_color('b')
par2.axis["right"].major_ticklabels.set_color('b')

print(Qe)
c2 = np.linspace(0,9.06737234/3600,40000)
a2 = make_interp_spline(Qe,He)(c2)
b2 = make_interp_spline(Qe,eta)(c2)
d2 = make_interp_spline(Qe,N)(c2)



p1, = host.plot(c2,a2,'g',label=r'$He - Qe\ $relationship')
host.scatter(He,Qe,marker='o',c = 'g')
# p4, = host.plot(He,Qe,'g', marker='o',linewidth=3)
p2, = par1.plot(c2,b2,'r-', label=r'$\eta - Qe\ $relationship')
p3, = par2.plot(c2,d2,'b-', label=r'$N - Qe\ $relationship')

host.set_xlim(min(Qe),max(Qe))
host.set_ylim(min(He), max(He)+1)
par1.set_ylim(min(eta), max(eta)+8)
par2.set_ylim(min(N), max(N)+0.1)

host.set_xlabel(r'$Qe\ /\ \rm{m}^{3}\cdot\rm{s}^{-1}$')
host.set_ylabel(r"$He\ /\ \rm{m}$",color='g',loc='top')
par1.set_ylabel(r"$\eta\ /\ 100\%$",color='r',loc='top')
par2.set_ylabel(r"$N\ /\ \rm{kW}$",color='b')


host.legend(loc = 'lower right')
host.set_title('Centrifugal pump characteristic curve(WB70/055)(2800r/min)', font=fpath,fontsize=13)
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())
plt.savefig('离心泵特性曲线.pdf')
plt.show()
np.savetxt("new.csv", eta, delimiter=',')

