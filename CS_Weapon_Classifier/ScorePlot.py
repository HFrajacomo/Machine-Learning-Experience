import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Plots a 2d graph', add_help=True)
parser.add_argument('-t','--tabular', dest='tabular', type=str, help='File containing all info', required=True)

args = parser.parse_args()

listA = pd.read_table(args.tabular)


plt.plot(listA.iloc[:,0])
plt.show()