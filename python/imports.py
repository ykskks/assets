# built-in
import copy
import random
import sys

# 3rd-party
import dask.dataframe as dd
import japanize_matplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from matplotlib_venn import venn2, venn3
from tqdm import tqdm_notebook as tqdm

# options
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=["#43C700"]) # 7FB469
%matplotlib inline
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None) # show non-truncated text in column
