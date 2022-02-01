import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
fig, axes =plt.subplots(1, 1)
axes.set_title('Cost Function Vs Learning Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Cost function')
fig.show()
fig.canvas.draw()
fig.canvas.flush_events()
plt.show()