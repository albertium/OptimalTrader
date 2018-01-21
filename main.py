

from _LIB_Core import plot_lines
import numpy as np

y1 = np.arange(10)
y2 = y1 * 1.1
data = {"y1": y1, "y2": y2}
plot_lines(data)