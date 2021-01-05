#!/usr/bin/python3

import argparse
import sys
import pandas as pd
from train import open_datafile

def predict(theta0, theta1, km):
	return theta0 + (theta1 * km)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Linear regression predict program")
	parser.add_argument("datafile", type=open_datafile, help="input a csv file with 2 floats well formated [theta0, theta1]")
	parser.add_argument("km", type=int, default="10000", help="input kilometers of the car you want to know the price for")
	args = parser.parse_args()
	price = predict(args.datafile.at[0, "theta0"], args.datafile.at[0, "theta1"], args.km)
	if args.datafile.at[0, "theta0"] == 0 and args.datafile.at[0, "theta1"] == 0:
		print("Your model is not usable")
		sys.exit()
	if price > 0:
		print("Your car is predicted to be at %d euros." % price)
	else:
		print("The price of your car cannot be estimated.")