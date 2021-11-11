import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
from pathlib import Path
fpath = Path(mpl.get_data_path(), "/System/Library/Fonts/Supplemental/Times New Roman.ttf")

x1 = [0.411,0.413,0.407,0.410,0.415,0.402,0.413,0.416,0.399,0.410,0.409,0.415]
x2 = [0.399,0.414,0.415,0.409,0.407,0.406,0.405,0.419,0.410,0.422,0.407,0.418]
x3 = [0.424,0.408,0.411,0.407,0.411,0.405,0.407,0.405,0.417,0.414,0.417,0.410]
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
x = [0,1,2,3,4,5,6,7,8,9,10,11]
x = np.array(x)
x = x + 1
ave = []
jicha = []
for i in [0,1,2,3,4,5,6,7,8,9,10,11]:
    p = np.average([x1[i],x2[i],x3[i]])
    ave.append(p)
    q = np.max([x1[i],x2[i],x3[i]]) - np.min([x1[i],x2[i],x3[i]])
    jicha.append(q)

print(ave)
print(jicha)

sum1 = sum(ave)
ave1 = sum1 / 12
sum2 = sum(jicha)
ave2 = sum2 / 12
UCLx = ave1 + 0.577 * ave2
CLx = ave1
LCLx = ave1 - 0.577 * ave2

plt.figure(figsize=(8,5))
plt.title('Control Figure', font=fpath,fontsize=13)
plt.hlines(UCLx,0,13,'r',label=r'UCLx',linestyles='--')
plt.hlines(CLx,0,13,'b',label=r'CLx',linestyles='--')
plt.hlines(LCLx,0,13,'g',label=r'LCLx',linestyles='--')
plt.scatter(x,ave,c='orange',marker='o')
plt.legend(loc = 'lower right')
plt.xlabel(r'number')
plt.ylabel(r'analysis number')
plt.axis([0,13, 0.400,0.420])
plt.savefig('质量控制.pdf' )
plt.show()