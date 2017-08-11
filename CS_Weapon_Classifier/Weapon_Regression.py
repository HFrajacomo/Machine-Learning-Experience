import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split
from MiraPlotting import setList
from MiraPlotting import miraplot_knn
import os
import random as rd
import argparse

parser = argparse.ArgumentParser(description='Trains a ML algorithm to better predict weapon type based on statiscs', add_help=True)
parser.add_argument('-t','--tabular', dest='tabular', type=str, help='File containing all info', required=True)
parser.add_argument('-x','--x_plot_column', dest='X_label', type=int, help='int representing the column number', required=True)
parser.add_argument('-y','--y_plot_column', dest='Y_label', type=int, help='int representing the column number', required=True)
parser.add_argument('-p','--plot', dest='plotgraph' ,action='store_true', default=False, help='plots a graph')
parser.add_argument('-l','--learn', dest='learning' ,action='store_true', default=False, help='adds results to train data')
parser.add_argument('-sp','--saveprediction', dest='savep' ,action='store_true', default=False, help='saves prediction data in a new file')
parser.add_argument('-spb','--savepredictionboth', dest='saveboth' ,action='store_true', default=False, help='saves prediction data in a new and data files')


args = parser.parse_args()
argTable = args.tabular


weapon = pd.read_table(argTable)

X = weapon[['rpm', 'damage', 'accuracy']]
y = weapon['weapon_id']
label = setList(list((weapon['weapon_type'].values)))
LearningCoeficient = 0

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=rd.randrange(0,1000))

lookup_weapon_name = dict(zip(weapon.weapon_id.unique(), weapon.weapon_type.unique()))

#Create classifier object
from sklearn.neighbors import KNeighborsClassifier
	
# Finding the best coeficient
C_range = range(1,20)
scores = []

for LearningCoeficient in C_range:
	knn = KNeighborsClassifier(n_neighbors = LearningCoeficient)
	knn.fit(X_train, y_train)
	scores.append(knn.score(X_test, y_test))

knn = KNeighborsClassifier(n_neighbors = scores.index(max(scores)) + 1)

if(not os.path.isfile("Learning_History.txt")): 
	file2 = open("Learning_History.txt", "w")
	file2.write("score_value\n")
else:
	file2 = open("Learning_History.txt", "a")

file2.write(str(max(scores)) + "\n")
file2.close()


#Train the classifier using training data
knn.fit(X_train, y_train)

#Estimate the accuracy of the classifier on future data
print("Test Score: " + str(knn.score(X_test, y_test)))
print("K neighbors: " + str(scores.index(max(scores))+ 1))

# Plot the decision boundaries of the k-NN classifier

if(args.plotgraph):
	miraplot_knn(X_train, y_train, scores.index(max(scores)) + 1, 'uniform', label, args.X_label, args.Y_label)

# Adding Prediction to dataset
if(args.learning):
	if(args.saveboth):
		file1 = open("tmp.txt", "w")
		if(not os.path.isfile("Predictions.txt")): 
			file3 = open("Predictions.txt", "w")
			file3.write("weapon_id\tweapon_name\tweapon_type\tshot_style\trpm\tdamage\taccuracy")
		else:
			file1 = open("Predictions.txt", "a")
	elif(not args.savep):
		file1 = open("tmp.txt", "w")
	else:
		if(not os.path.isfile("Predictions.txt")): 
			file1 = open("Predictions.txt", "w")
			file1.write("weapon_id\tweapon_name\tweapon_type\tshot_style\trpm\tdamage\taccuracy")
		else:
			file1 = open("Predictions.txt", "a")



	i = 0
	for i in range(0,10):
		mRPM = rd.random()*weapon.rpm.max()
		mDAMAGE = rd.random()*weapon.damage.max()
		mACCURACY = rd.random()*weapon.accuracy.max()
		weapon_prediction = knn.predict([[mRPM,mDAMAGE,mACCURACY]])	
		mTYPE = lookup_weapon_name[weapon_prediction[0]]
		file1.write("\n" + str(int(weapon_prediction)) + "\t0\t" + str(mTYPE) + "\t0\t" + str(mRPM) + "\t" + str(mDAMAGE) + "\t" + str(mACCURACY))
		if(args.saveboth):
			file3.write("\n" + str(int(weapon_prediction)) + "\t0\t" + str(mTYPE) + "\t0\t" + str(mRPM) + "\t" + str(mDAMAGE) + "\t" + str(mACCURACY))
	if(args.saveboth):
		file3.close()
	file1.close()

	if(args.savep and not args.saveboth):
		pass
	else:
		os.system("copy /A "+ str(argTable) +" + tmp.txt /B "+ str(argTable))
		os.system("del tmp.txt /Q")


