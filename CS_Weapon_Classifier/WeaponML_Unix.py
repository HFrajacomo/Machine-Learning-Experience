import os
import argparse
import time as tm
from datetime import datetime

parser = argparse.ArgumentParser(description='Runs Weapon_Regression.py in a loop', add_help=True)
parser.add_argument('-t','--tabular', dest='tabular', type=str, help='File containing all info', required=True)
parser.add_argument('-i','--iterations', dest='iter', type=int, help='amount of iterations', required=True)
parser.add_argument('-x','--x_plot_column', dest='X_label', type=int, help='int representing the column number', required=True)
parser.add_argument('-y','--y_plot_column', dest='Y_label', type=int, help='int representing the column number', required=True)

args = parser.parse_args()

i=0

start_time = datetime.now()

for i in range(0,args.iter):
	os.system("python Weapon_Regression_Unix.py -t " + args.tabular + " -x " + str(args.X_label) + " -y " + str(args.Y_label) + " -l")
	print("Prediction: " + str(i+1) + " of " + str(args.iter))

print("Learning process finished!")
print("Elapsed Time: " + str(datetime.now() - start_time))
os.system("python 3-featurePlot.py -t " + args.tabular)