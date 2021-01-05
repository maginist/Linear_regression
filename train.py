#!/usr/bin/python3

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plot

class	Process(object):
	
	def __init__(self, datafile, output, theta=[0, 0], rate=0.1, range=1000):
		self.output=output,
		self.theta=theta,
		self.data=datafile
		self.range=range,
		self.rate=rate

	# def 

def open_thetafile(thetafile):
	try:
		f = open(thetafile, "w")
		f.write("theta0,theta1\n0,0")
		f.close()
	except:
		sys.exit("Can't open thetafile")
	return thetafile

def open_datafile(datafile):
	try:
		data = pd.read_csv(datafile)
	except pd.errors.EmptyDataError:
		sys.exit("Empty data file.")
	except pd.errors.ParserError:
		raise argparse.ArgumentTypeError("Error parsing file, needs to be a well formated csv.")
	except:
		sys.exit("File {datafile} corrupted or does not exist.")
	return data


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Linear regression training program")
	parser.add_argument("datafile_train", type=open_datafile, help="input a csv file well formated")
	parser.add_argument("-o", "--output", type=open_thetafile, default="theta.csv", help="output data file")
	parser.add_argument("-r", "--range", type=int, default=1000, help="training range")
	parser.add_argument("-rt", "--rate", type=float, default=0.1, help="training rate")
	parser.add_argument("-v", "--visual", action="store_true", default=False, help="show regression")
	args = parser.parse_args()
	training = Process(args.datafile_train, args.output, rate=args.rate, range=args.range)