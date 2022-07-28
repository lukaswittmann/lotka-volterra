import numpy as np
import matplotlib.pyplot as plt
import time
import random

'''Definition der Laufvariablen'''
t_tot = 500
dt = 0.01
l_tot = 250
t = np.arange(0, t_tot+0.01, dt)
n = len(t)

alpha = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)
alpha[:,:] = 0.5

beta = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)
beta[:,:] = 0.5

delta = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)
delta[:,:] = 0.4

gamma = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)
gamma[:,:] = 0.1

'''Arrays fuer die beiden Konzentrationen'''
x = np.zeros(shape=(l_tot+1, l_tot+1), dtype=float)
y = np.zeros(shape=(l_tot+1, l_tot+1), dtype=float)

xn = np.zeros(shape=(l_tot+1, l_tot+1), dtype=float)
yn = np.zeros(shape=(l_tot+1, l_tot+1), dtype=float)

'''Verschobene Konzentrationsarrays fuer die Diffusion'''
x_oben = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)
x_unten = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)
x_links = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)
x_rechts = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)

x_diff_gesamt = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)

y_oben = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)
y_unten = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)
y_links = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)
y_rechts = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)

y_diff_gesamt = np.empty(shape=(l_tot+1, l_tot+1), dtype=float)

Dx = 0.5
Dy = 0.05

betrachtungsintervall = 50
h = 0

plt.ion()

def update_figures(x,y, dt, i):
    figure1.set_data(x[:,:])
    figure2.set_data(y[:, :])
    fig.suptitle("t=" + str(np.round((dt * i), decimals=1)) + ", Iteration " + str(i), fontsize=14)
    plt.draw()
    #plt.savefig("export/image_" + str(h) + ".png", dpi=250)
    plt.pause(0.001)
    return

# create figure
fig = plt.figure(figsize=(7, 4))

# setting values to rows and column variables
rows = 1
columns = 2

# adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
figure1 = plt.imshow(x[:,:], cmap='plasma',interpolation="bilinear") #Greys bicubic bilinear
plt.axis('off')
plt.clim(0, 5)
fig.colorbar(figure1)
plt.title("prey", fontsize=10)

# adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
figure2 = plt.imshow(y[:,:], cmap='plasma',interpolation="bilinear")
plt.axis('off')
plt.clim(0, 8)
fig.colorbar(figure2)
plt.title("predator", fontsize=10)
fig.suptitle("t=0, Iteration 0", fontsize=12)
plt.tight_layout()

start = time.time()

print("Start of iteration...")
for i in range(1, n):


    x_diff_gesamt[:, :] = Dx * (- 4 * x[:, :] + np.roll(x[:, :], -1, axis=0) + np.roll(x[:, :], 1, axis=0) + np.roll(x[:, :], -1, axis=1) + np.roll(x[:, :], 1, axis=1))
    y_diff_gesamt[:, :] = Dy * (- 4 * y[:, :] + np.roll(y[:, :], -1, axis=0) + np.roll(y[:, :], 1, axis=0) + np.roll(y[:, :], -1, axis=1) + np.roll(y[:, :], 1, axis=1))


    xn[:, :] = x[:, :] + (alpha[:,:]*x[:,:]-beta[:,:]*x[:,:]*y[:,:]+ x_diff_gesamt[:, :]) * dt
    yn[:, :] = y[:, :] + (delta[:,:]*x[:,:]*y[:,:]-gamma[:,:]*y[:,:]+ y_diff_gesamt[:, :]) * dt

    x[:, :] = xn[:, :]
    y[:, :] = yn[:, :]


    end = time.time()

    # Creation of random fluctionations at the start
    if i % int(random.random() * 100 + 1) == 0 and i < 500:
        f1 = 1
        f2 = 1
        x1 = int(random.random() * l_tot)
        y1 = int(random.random() * l_tot)
        x2 = int(random.random() * l_tot)
        y2 = int(random.random() * l_tot)
        x[x1,y1] += f1
        y[x2,y2] += f2

    if i % ((t_tot/dt)/1000) == 0:
        end = time.time()
        print("Berechnung: " + str(np.round((i/n)*100, decimals=1)) + "% erledigt, Iteration " + str(i) + " mit " + str(np.round(i / (end - start), decimals=1)) + " Iters/s Restdauer: " + str(np.round((((t_tot/dt)-i)/(i / (end - start)))/60, decimals=1)) + " min")

    if i == 1:
        update_figures(x,y, dt, i)
        h += 1

    if i % betrachtungsintervall == 0:
        update_figures(x,y, dt, i)
        h += 1
