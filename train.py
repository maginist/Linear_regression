#!/usr/bin/python3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Process(object):

    def __init__(self, datafile, output, rate, range):
        self.output = output
        self.theta = [0, 0]
        self.th_history = [[], []]
        self.tmp_history = [[], []]
        self.range = range
        self.rate = rate
        self.data = np.array(datafile)
        self.km_ref = self.data[:, 0]
        self.price_ref = self.data[:, 1]
        self.km = self.standardize(self.data[:, 0])
        self.price = self.standardize(self.data[:, 1])

    def predict(self, theta0, theta1, km):
        return theta0 + (theta1 * km)

    def standardize(self, x):
        return ((x - np.mean(x)) / np.std(x))

    def destandardize(self, x, x_ref):
        return x * np.std(x_ref) + np.mean(x_ref)

    def write_theta(self, t0, t1):
        try:
            f = open(t.output, "w")
            print(t0,t1)
            f.write(f"theta0,theta1\n{t0},{t1}")
            f.close()
            print("The thetafile is written, you need to use predict.py now.")
        except Exception as error:
            exit(f"{error}: Can't open thetafile.")
        return

def train(t):
    m = len(t.km)
    for i in range(t.range):
        tmp0 = t.rate * (1 / m) * sum([t.predict(t.theta[0], t.theta[1], t.km[i]) - t.price[i] for i in range(m)])
        tmp1 = t.rate * (1 / m) * sum([(t.predict(t.theta[0], t.theta[1], t.km[i]) - t.price[i]) * t.km[i] for i in range(m)])
        t.theta[0] -= tmp0
        t.theta[1] -= tmp1
        t.tmp_history[0].append(tmp0)
        t.tmp_history[1].append(tmp1)
        t.th_history[0].append(t.theta[0])
        t.th_history[1].append(t.theta[1])
    t.write_theta(t.theta[0], t.theta[1])
    return


def open_thetafile(thetafile):
    try:
        f = open(thetafile, "w")
        f.write("theta0,theta1\n0,0")
        f.close()
    except Exception as error:
        exit(f"{error}: Can't open thetafile.")
    return thetafile


def open_datafile(datafile):
    try:
        data = pd.read_csv(datafile)
        data = data.sort_values(by=["km"], ignore_index=True)
    except pd.errors.EmptyDataError:
        exit("Empty data file.")
    except pd.errors.ParserError:
        raise argparse.ArgumentTypeError("Error parsing file, needs to be a well formated csv.")
    except Exception as error:
        exit(f"{error}: File {datafile} corrupted or does not exist.")
    return data

def print_history(hist, tmp):
    hst, ((hist0, mean0), (hist1, mean1)) = plt.subplots(2, 2)
    hist0.plot(hist[0], "b-")
    mean0.plot(tmp[0], "b-")
    hist1.plot(hist[1], "r-")
    mean1.plot(tmp[1], "r-")
    mean0.set_title("MSE of Th0")
    hist0.set_title("Theta0")
    mean1.set_title("MSE of Th1")
    hist1.set_title("Theta1")
    plt.show()

def print_regression(km, price, th0, th1):
    plt.plot(km, price, "bo", label="Data")
    plt.plot(km, th0 + th1 * km, "-r", label="regression")
    plt.xlabel("km")
    plt.ylabel("price")
    plt.xlim(min(km), max(km))
    plt.ylim(min(price), max(price))
    plt.legend()
    plt.title("ft_linear_regression")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear regression t program")
    parser.add_argument("datafile_train", type=open_datafile, help="input a csv file well formated")
    parser.add_argument("-o", "--output", type=open_thetafile, default="theta.csv", help="output data file")
    parser.add_argument("-r", "--range", type=int, default=1000, help="training range (epochs)")
    parser.add_argument("-rt", "--rate", type=float, default=0.1, help="training rate")
    parser.add_argument("-s", "--show", action="store_true", default=False, help="show regression")
    parser.add_argument("-H", "--history", action="store_true", default=False, help="show history (mean square error")
    args = parser.parse_args()
    t = Process(args.datafile_train, args.output, rate=args.rate, range=args.range)
    train(t)
    if args.show:
        print_regression(t.km, t.price, t.theta[0], t.theta[1])
    if args.history:
        print_history(t.th_history, t.tmp_history)
