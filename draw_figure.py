import numpy as np
import matplotlib.pyplot as plt

START=0.
END=10
STEP=0.01

def spike(intime,weight,start=START,end=END,step=STEP):
    line = np.arange(start,end,step)
    exp = weight*np.exp((intime-line)/3)
    switch = np.greater_equal(line,intime)
    exp = exp*switch
    return line,exp

x1,y1 = spike(1.1,2)
x2,y2 = spike(0.1,1.3)
x3,y3 = spike(2.3,0.9)
x4,y4 = spike(5,4)

x = x1
y = y1+y2+y3+y4
y2 = np.cumsum(y/100)


fig1 = plt.subplot(211)
fig2 = plt.subplot(212)

fig1.plot(x,y)
fig2.plot(x,y2)
plt.show()
