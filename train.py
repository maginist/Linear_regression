#!/usr/bin/python3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Process(object):

    def __init__(self, datafile, output, rate, range):
        self.output = output
        self.theta = [0, 0]
        self.range = range
        self.rate = rate
        self.data = np.array(datafile)
        self.km_ref = self.data[:, 0]
        self.price_ref = self.data[:, 1]
        self.km = self.standardize(self.data[:, 0])
        self.price = self.standardize(self.data[:, 1])
        self.sum_km = np.sum(self.data[:,0])
        self.sum_price = np.sum(self.data[:,1])

    def standardize(self, x):
        return ((x - np.mean(x)) / np.std(x))

    def destandardize(self, x, x_ref):
        return x * np.std(x_ref) + np.mean(x_ref)

    def mean_square_error(self):
        return

    def gradient_descent(self):
        return

    def write_theta(self):
        try:
            f = open(training.output, "w")
            print("coucou")
            f.write(f"theta0,theta1\n%d,%d"% training.theta[:,0], training.theta[:,1] )
            f.close()
            exit("The thetafile is written, you need to use predict.py now.")
        except Exception:
            exit("Can't open thetafile.")
        return

def train(training):
    return


def open_thetafile(thetafile):
    try:
        f = open(thetafile, "w")
        f.write("theta0,theta1\n0,0")
        f.close()
    except Exception:
        exit("Can't open thetafile.")
    return thetafile


def open_datafile(datafile):
    try:
        data = pd.read_csv(datafile)
        data = data.sort_values(by=["km"], ignore_index=True)
    except pd.errors.EmptyDataError:
        exit("Empty data file.")
    except pd.errors.ParserError:
        raise argparse.ArgumentTypeError("Error parsing file, needs to be a well formated csv.")
    except Exception:
        exit("File {datafile} corrupted or does not exist.")
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
    train(training)
    print(args.datafile_train)
