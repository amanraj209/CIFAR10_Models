# Setup code for this notebook
import matplotlib.pyplot as plt
from IPython import get_ipython

# This is a bit of magic to make matplotlib figures appear inline
# in the notebook rather than in a new window
get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# %load_ext autoreload
# %autoreload 2

