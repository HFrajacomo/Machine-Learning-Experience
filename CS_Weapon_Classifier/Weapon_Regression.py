import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from MiraPlotting import setList
import os
import random as rd
import argparse
from datetime import datetime

def to_txt(self, filename):
	f = open(filename, "w")
	for line in self:
		f.write(line)
	return

parser = argparse.ArgumentParser(description='Trains a ML algorithm to better predict weapon type based on statiscs', add_help=True)
parser.add_argument('-t','--tabular', dest='tabular', type=str, help='File containing all info', required=True)
parser.add_argument('-tf','--testfile', dest='testfile', type=str, help='File containing testing info', required=True)
parser.add_argument('-sp','--saveprediction', dest='savep' ,action='store_true', default=False, help='saves prediction data in a new file')

start_time = datetime.now()
args = parser.parse_args()
argTable = args.tabular

lineInfile = open(args.testfile, "r")

kvalue = -1

for line in lineInfile:
	kvalue = kvalue + 1

lineInfile.close()

weapon = pd.read_table(argTable)
TestingTable = pd.read_table(args.testfile)

X = weapon[['rpm', 'damage', 'accuracy']]
y = weapon['weapon_id']

Test_X = TestingTable[['rpm', 'damage', 'accuracy']]
Test_y = TestingTable['weapon_id'] 

LearningCoeficient = 0

lookup_weapon_name = dict(zip(weapon.weapon_id.unique(), weapon.weapon_type.unique()))

#Create classifier object
from sklearn.neighbors import KNeighborsClassifier
	
# Finding the best coeficient

knn = KNeighborsClassifier(n_neighbors = int(kvalue/2))

#Train the classifier using training data
knn.fit(X, y)
score = knn.score(X,y)
print("Score: " + str(score))

if(args.savep):
	file3 = open(os.path.splitext(args.testfile)[0] + "_Results.txt", "w")
else:
	file3 = open(args.testfile, "w")

file3.write("weapon_id\tweapon_name\tweapon_type\tshot_style\trpm\tdamage\taccuracy\n")
TestFile = pd.read_table(args.testfile)

i = 0
for i in range(0,kvalue):

	weapon_prediction = knn.predict([[TestFile.loc[i]['rpm'], TestFile.loc[i]['damage'], TestFile.loc[i]['accuracy']]])	
	TestFile.loc[i]['weapon_id'] = str(int(weapon_prediction))
	TestFile.loc[i]['weapon_type'] = lookup_weapon_name[weapon_prediction[0]]
	file3.write(str(int(weapon_prediction)) + "\t0\t" + lookup_weapon_name[weapon_prediction[0]] + "\t0\t" + str(TestFile.loc[i]['rpm']) + "\t" + str(TestFile.loc[i]['damage']) + "\t" + str(TestFile.loc[i]['accuracy']) + "\n")
	'''
	Outfile = open(os.path.splitext(args.testfile)[0] + "_Results.txt", "w")
	for rows in TestFile:
		Outfile.write(str(rows))
	
	TestFile.to_csv(os.path.splitext(args.testfile)[0] + "_Results.txt", sep='\t')

	
	Outfile = open(args.testfile, "w")
	for rows in TestFile:
		Outfile.write(str(rows))
	
	TestFile.to_csv(args.testfile, sep='\t')
	'''
file3.close()
#Outfile.close()
print("Elapsed Time: " + str(datetime.now() - start_time))

