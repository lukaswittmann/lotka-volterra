import numpy as np
import matplotlib.pyplot as plt

t_tot = 1000
dt = 0.01
t = np.arange(0, t_tot+0.01, dt)
n = len(t)

x = np.empty(n)
y = np.empty(n)

x[:]=0
y[:]=0

x[0]=1
y[0]=1

alpha = 0.5
beta = 0.5
delta = 0.4
gamma = 0.1


'''Iterationsschleife'''
for i in range(1, n):
    x[i]=x[i-1]+(alpha*x[i-1]-beta*x[i-1]*y[i-1])*dt
    y[i]=y[i-1]+(delta*x[i-1]*y[i-1]-gamma*y[i-1])*dt

# show c(t)
plt.clf()
plt.plot(t, x, label='prey',linewidth=1.5)
plt.plot(t, y, label='predator',linewidth=1.5)
plt.title("c(t)")
plt.legend()
plt.grid(linestyle='dotted', linewidth=0.25)
plt.ylabel('c')
plt.xlabel('t')
plt.tight_layout()
plt.show()

# show limit cycle
plt.clf()
plt.title("limit cycle")
plt.grid(linestyle='dotted', linewidth=0.25)
plt.plot(x,y, label='limit cycle',linewidth=1.5)
plt.show()





