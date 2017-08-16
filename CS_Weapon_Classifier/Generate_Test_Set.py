import argparse
import random as rd
import sys

parser = argparse.ArgumentParser(description='Generate random dataframe for weapon testing', add_help=True)
parser.add_argument('-fn','--filename', dest='name', type=str, help='string containing the output file name', required=True)
parser.add_argument('-r','--rows', dest='iterations', type=int, help='int representing the amount of rows in output file', required=True)

args = parser.parse_args()
header = False
i = 0

if(args.iterations <= 0):
	print("-r number has to be bigger than 0")
	sys.exit()

outputfile = open(args.name, "w")

for i in range (0, args.iterations):
	if(not header):
		outputfile.write("weapon_id\tweapon_name\tweapon_type\tshot_style\trpm\tdamage\taccuracy\n")
		header = True
	mRPM = rd.random()*1000
	mDAMAGE = rd.random()*180
	mACCURACY = rd.random()*20
	outputfile.write("\t\t\t\t" + str(mRPM) + "\t" + str(mDAMAGE) + "\t" + str(mACCURACY) + "\n")

outputfile.close()
print("Generated " + str(args.iterations) + " entries successfully")