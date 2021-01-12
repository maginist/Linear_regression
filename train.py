#!/usr/bin/python3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Process(object):

    def __init__(self, datafile, output, rate, range):
        self.output = output
        self.theta = [0, 0]
        self.th_history = []
        self.range = range
        self.rate = rate
        self.data = np.array(datafile)
        self.km_ref = self.data[:, 0]
        self.price_ref = self.data[:, 1]
        self.km = self.standardize(self.data[:, 0])
        self.price = self.standardize(self.data[:, 1])

    def predict(self, theta0, theta1, km):
        return theta0 + (theta1 * km)

    def predict_sum(self, t0, t1, km, price, b):
        sum = []
        for i in range(len(km)):
            if b == 0:
                sum.append(self.predict(t0, t1, km[i]) - price[i])
            else:
                sum.append(self.predict(t0, t1, km[i]- price[i]) * km[i])
        return sum

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
            f = open(t.output, "w")
            f.write(f"theta0,theta1\n%d,%d"% t.theta[0], t.theta[1])
            f.close()
            exit("The thetafile is written, you need to use predict.py now.")
        except Exception:
            exit("Can't open thetafile.")
        return

def train(t):
    for i in range(t.range):
        tmp0 = t.rate * (1 / len(t.km)) * np.asarray(t.predict_sum(t.theta[0], t.theta[1], t.km, t.price, 0))
        tmp1 = t.rate * (1 / len(t.km)) * np.asarray(t.predict_sum(t.theta[0], t.theta[1], t.km, t.price, 1))

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
    parser = argparse.ArgumentParser(description="Linear regression t program")
    parser.add_argument("datafile_train", type=open_datafile, help="input a csv file well formated")
    parser.add_argument("-o", "--output", type=open_thetafile, default="theta.csv", help="output data file")
    parser.add_argument("-r", "--range", type=int, default=1000, help="t range")
    parser.add_argument("-rt", "--rate", type=float, default=0.1, help="t rate")
    parser.add_argument("-v", "--visual", action="store_true", default=False, help="show regression")
    args = parser.parse_args()
    t = Process(args.datafile_train, args.output, rate=args.rate, range=args.range)
    train(t)
