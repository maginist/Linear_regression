#!/usr/bin/python3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progress.bar import ChargingBar


class Process(object):

    def __init__(self, datafile, output, rate, range):
        self.output = output
        self.theta = [0, 0]
        self.theta_dest = [0, 0]
        self.th_history = [[], []]
        self.tmp_history = [[], []]
        self.range = range
        self.rate = rate
        self.data = np.array(datafile)
        self.km_ref = self.data[:, 0]
        self.price_ref = self.data[:, 1]
        self.km = self.standardize(self.data[:, 0])
        self.price = self.standardize(self.data[:, 1])
        self.r2 = 0

    def predict(self, theta0, theta1, km):
        return theta0 + (theta1 * km)

    def standardize(self, x):
        return ((x - np.mean(x)) / np.std(x))

    def destandardize(self, x, x_ref):
        return x * np.std(x_ref) + np.mean(x_ref)

    def destandardize_theta(self, km_ref, price_ref, theta):
        x0 = 1000
        y0 = 42000
        x0_stdz = (x0 - np.mean(km_ref)) / np.std(km_ref)
        y0_stdz = (y0 - np.mean(km_ref)) / np.std(km_ref)
        x1_stdz = theta[0] + (theta[1] * x0_stdz)
        y1_stdz = theta[0] + (theta[1] * y0_stdz)
        x1 = x1_stdz * np.std(price_ref) + np.mean(price_ref)
        y1 = y1_stdz * np.std(price_ref) + np.mean(price_ref)
        th1 = (y1 - x1) / (y0 - x0)
        th0 = x1 - (th1 * x0)
        theta[0] = th0
        theta[1] = th1

    def write_theta(self, t0, t1):
        try:
            f = open(t.output, "w")
            print(f"The obtained thetas are : \nth0 : {t0}\nth1 : {t1}\n")
            f.write(f"theta0,theta1\n{t0},{t1}")
            f.close()
            print("The thetafile is written, you need to use predict.py now.")
        except Exception as error:
            exit(f"{error}: Can't open thetafile.")
        return


def train(t, autorate):
    m = len(t.km)
    bar = ChargingBar('Training', max=t.range, suffix='%(percent)d%%')
    for i in range(t.range):
        tmp0 = t.rate * (1 / m) * sum([t.predict(t.theta[0], t.theta[1], t.km[i]) - t.price[i] for i in range(m)])
        tmp1 = t.rate * (1 / m) * sum([(t.predict(t.theta[0], t.theta[1], t.km[i]) - t.price[i]) * t.km[i] for i in range(m)])
        t.theta[0] -= tmp0
        t.theta[1] -= tmp1
        t.tmp_history[0].append(tmp0)
        t.tmp_history[1].append(tmp1)
        t.th_history[0].append(t.theta[0])
        t.th_history[1].append(t.theta[1])
        bar.next()
    bar.finish()
    t.theta_dest[0] = t.theta[0]
    t.theta_dest[1] = t.theta[1]
    t.destandardize_theta(t.km_ref, t.price_ref, t.theta_dest)
    t.write_theta(t.theta_dest[0], t.theta_dest[1])


def compute_r2(t):
    m = len(t.km)
    sum_mse = sum([(t.predict(t.theta[0], t.theta[1], t.km[i]) - t.price[i]) ** 2 for i in range(m)])
    sum2 = sum([(t.km[i] - np.mean(t.km)) ** 2 for i in range(m)])
    t.r2 = 1 - sum_mse / sum2
    print(f"The R2 score for this data set is : {t.r2}")


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
    hst.suptitle("History of Thetas and MSEs during training")
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
    plt.legend()
    plt.title("ft_linear_regression")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear regression training program")
    parser.add_argument("datafile_train", type=open_datafile, help="input a csv file well formated")
    parser.add_argument("-o", "--output", type=open_thetafile, default="theta.csv", help="output data file")
    parser.add_argument("-r", "--range", type=int, default=1000, help="training range (epochs)")
    parser.add_argument("-rt", "--rate", type=float, default=0.1, help="training rate")
    parser.add_argument("-s", "--show", action="store_true", default=False, help="show regression")
    parser.add_argument("-sd", "--show_dest", action="store_true", default=False, help="show regression destandarized")
    parser.add_argument("-H", "--history", action="store_true", default=False, help="show history (mean square error")
    parser.add_argument("-ar", "--autorate", action="store_true", default=False, help="Use autorating for stopping the program when it's not usefull to continue.")
    parser.add_argument("-r2", "--r2score", action="store_true", default=False, help="Compute R2 score (coefficient of determination)")
    args = parser.parse_args()
    t = Process(args.datafile_train, args.output, rate=args.rate, range=args.range)
    train(t, args.autorate)
    if args.show:
        print_regression(t.km, t.price, t.theta[0], t.theta[1])
    if args.show_dest:
        print_regression(t.km_ref, t.price_ref, t.theta_dest[0], t.theta_dest[1])
    if args.history:
        print_history(t.th_history, t.tmp_history)
    if args.r2score:
        compute_r2(t)
