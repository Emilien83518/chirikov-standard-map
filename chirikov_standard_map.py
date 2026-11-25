

import numpy as np
from matplotlib import pyplot as plt

plt.close('all')

eps = 0.971

np.random.seed(2)

plt.figure(1)

for eloop in range(0,100):

    rlast = 2*np.pi*(0.5-np.random.random())
    thlast = 4*np.pi*np.random.random()


    rplot = np.zeros(shape=(200,))
    thetaplot = np.zeros(shape=(200,))
    for loop in range(0,200):
        rnew = rlast + eps*np.sin(thlast)
        thnew = np.mod(thlast+rnew,4*np.pi)

        thetaplot[loop] = np.mod(thnew-np.pi,4*np.pi)

        rtemp = np.mod(rnew + np.pi,2*np.pi)

        rplot[loop] = rtemp - np.pi


        rlast = rnew
        thlast = thnew

    plt.plot(np.real(thetaplot),np.real(rplot),'o',ms=.2)
    plt.xlim(xmin=np.pi,xmax=4*np.pi)
    plt.ylim(ymin=-2.5,ymax=2.5)

plt.xticks(
    [np.pi, 2*np.pi, 3*np.pi, 4*np.pi],
    [r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$']
)
plt.yticks(
    [-np.pi/2, 0, np.pi/2],
    [r'$-\pi/2$', r'$0$', r'$\pi/2$']
)
plt.xlabel(r'$\theta$', fontsize=12)
plt.ylabel(r'$p$', fontsize=12)

K = eps
plt.title(f'Chirikov Standard Map   $K = {K}$', fontsize=13)
plt.savefig('StandMap.png',dpi=900)
