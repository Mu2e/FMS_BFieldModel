import numpy as np
import matplotlib.pyplot as plt
from emtracks.plotting import config_plots

config_plots()
plt.rcParams['figure.figsize'] = [8, 8] # larger figures
# plt.rcParams['axes.grid'] = True         # turn grid lines on
# plt.rcParams['axes.axisbelow'] = True    # put grid below points
# plt.rcParams['grid.linestyle'] = '--'    # dashed grid
plt.rcParams.update({'font.size': 18.0})   # increase plot font size

plotdir = '/home/ckampa/data/plots/html/random/'

xs = np.linspace(-4, 4, 100)
ys_chi2 = xs**2
ys_huber = (2*np.abs(xs)**(1/2) - 1)**2
ys_huber[np.abs(xs)<=1] = xs[np.abs(xs)<=1]**2

fig, ax = plt.subplots()
ax.plot(xs, ys_chi2, linewidth=2, label=r'$\chi^2$ loss')
ax.plot(xs, ys_huber, linewidth=2, label='Huber loss')
# ax.set_xlabel('Distance from True Parameter Value')
# ax.set_ylabel('Cost')
ax.set_xlabel('(Data - Fit)')
ax.set_ylabel('Contribution to Objective Function')
ax.legend()

fig.savefig(plotdir+'huber.pdf')
fig.savefig(plotdir+'huber.png')

# plt.show()
