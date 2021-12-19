import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
from pathlib import Path
from scipy.interpolate import make_interp_spline
np.set_printoptions(suppress=True)



#压力导入(Pa)
#压力表读数
p1 = [0.007,0.010,0.012,0.015,0.018,0.020,0.025,0.030,0.035,0.040,0.045,0.050,0.055,0.065,0.070]
p1 = np.array(p1)
#真空表读数
p2 = [0,0,0,0,0,-0.002,-0.006,-0.010,-0.014,-0.018,-0.022,-0.028,-0.032,-0.036,-0.040]
p2 = np.array(p2)
p = p1 - p2
p = p * 1000
p = p * 1000
#流量导入(Pa)
Q = [-0.5,-0.2,0.4,1.0,1.8,2.8,3.8,5.0,6.3,7.7,9.2,10.8,12.3,13.9,15.1]
Q = np.array(Q)
Q = Q + 0.8
Q = Q * 1000
#单位转换(m^3/s)
Qe = 0.0742 * pow(Q,0.4958)
Qe = Qe / 3600
#水温导入(摄氏度)
t = [22.4,22.4,22.3,22.3,22.3,22.3,22.3,22.3,22.3,22.3,22.2,22.2,22.1,22.0,21.9]
t = np.array(t)
#电机频率(Hz)
f = [50,47,44,41,38,35,32,29,26,23,20,17,14,11,8]
#内插法确定密度
rho21 = 997.9955
rho23 = 997.5415
rho = ((rho23 - rho21)/2)*(t - 23) + rho23
#压头(m)
H = p/(rho*9.8)
#扬程(m)
He = H + 0.18

fpath = Path(mpl.get_data_path(), "/System/Library/Fonts/Supplemental/Times New Roman.ttf")


print(Qe)
print(He)
c2 = np.linspace(0.00034854441059712493,0.0024954800297795006,50000)
a2 = make_interp_spline(Qe,He)(c2)

plt.figure(figsize=(8,5))
plt.title('characteristic curve', font=fpath,fontsize=13)
plt.xlabel(r'$Qe\ /\ \rm{m}^{3}\cdot\rm{s}^{-1}$')
plt.ylabel(r"$He\ /\ \rm{m}$",loc='top')
# plt.axis([min(Qe),min(He), max(He)+1)
plt.plot(c2,a2,'g-',label=r'Piping characteristic curve')
plt.scatter(Qe,He,c='g',marker='o')

np.savetxt("new.csv", He, delimiter=',')





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
c2 = np.linspace(0,9.06737234/3600,40000)
a2 = make_interp_spline(Qe,He)(c2)
plt.plot(c2,a2,'orange',label=r'Centrifugal pump characteristic curve')
plt.scatter(Qe,He,c='orange',marker='o')

plt.legend(loc = 'upper right')
plt.savefig('管路特性曲线.pdf')
plt.show()


