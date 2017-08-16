import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from MiraPlotting import setList
from MiraPlotting import miraplot_knn

parser = argparse.ArgumentParser(description='Plots a dataframe relationship graph', add_help=True)
parser.add_argument('-t','--tabular', dest='tabular', type=str, help='File containing all info', required=True)
parser.add_argument('-p','--plot', dest='PLOT' ,action='store_true', default=False, help='plots a 2d graph if active')
parser.add_argument('-td','--3dplot', dest='td' ,action='store_true', default=False, help='plots a 3d graph if active')
parser.add_argument('-hm','--heatmap', dest='hm' ,action='store_true', default=False, help='plots a knn graph if active')
parser.add_argument('-nn','--n-neighbors', dest='nn',type=int, default=20, help='Sets n-neighbors for heatmap representation')


args = parser.parse_args()

objects = pd.read_table(args.tabular)

X = objects[[list(objects)[4], list(objects)[5], list(objects)[6]]]
y = objects[list(objects)[0]]

label = setList(list((objects['weapon_type'].values)))

if(args.PLOT):
	# Two Dimensional Dataframe plot

	cmap = cm.get_cmap('gnuplot')
	scatter = pd.plotting.scatter_matrix(X, c= y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(12,12), cmap=cmap)
	plt.show()

	#End here

if(args.td):

	# Three Dimensional Dataframe plot
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')

	ax.scatter(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2], c=y, marker = 'o', s=100) 
	ax.set_xlabel(list(X)[0])
	ax.set_ylabel(list(X)[1])
	ax.set_zlabel(list(X)[2])
	plt.show()
	#End Here

if(args.hm):
	miraplot_knn(X, y, args.nn, 'uniform', label, 0, 1)
	