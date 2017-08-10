import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Plots a dataframe relationship graph', add_help=True)
parser.add_argument('-t','--tabular', dest='tabular', type=str, help='File containing all info', required=True)
parser.add_argument('-td','--3dplot', dest='td' ,action='store_true', default=False, help='plots a 3d graph if active')

args = parser.parse_args()

objects = pd.read_table(args.tabular)

X = objects[[list(objects)[4], list(objects)[5], list(objects)[6]]]
y = objects[list(objects)[0]]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)


if(not args.td):
	# Two Dimensional Dataframe plot

	cmap = cm.get_cmap('gnuplot')
	scatter = pd.plotting.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(12,12), cmap=cmap)
	plt.show()

	#End here

else:

	# Three Dimensional Dataframe plot
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')

	ax.scatter(X_train.iloc[:,0], X_train.iloc[:,1], X_train.iloc[:,2], c=y_train, marker = 'o', s=100) 
	ax.set_xlabel(list(X_train)[0])
	ax.set_ylabel(list(X_train)[1])
	ax.set_zlabel(list(X_train)[2])
	plt.show()
	#End Here
	