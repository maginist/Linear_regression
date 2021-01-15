#!/usr/bin/python3

import argparse
import pandas as pd


def predict(theta0, theta1, km):
    return theta0 + (theta1 * km)


def open_datafile(datafile):
    try:
        data = pd.read_csv(datafile)
    except pd.errors.EmptyDataError:
        exit("Empty data file.")
    except pd.errors.ParserError:
        raise argparse.ArgumentTypeError("Error parsing file, needs to be a well formated csv.")
    except Exception as error:
        exit(f"{error}: File {datafile} corrupted or does not exist.")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear regression predict program")
    parser.add_argument("datafile", type=open_datafile, help="input a csv file with 2 floats well formated [theta0, theta1]")
    parser.add_argument("km", type=int, default="10000", help="input kilometers of the car you want to know the price for")
    args = parser.parse_args()
    if args.datafile.at[0, "theta0"] == 0 and args.datafile.at[0, "theta1"] == 0:
        exit("Thetas are not set.")
    price = predict(args.datafile.at[0, "theta0"], args.datafile.at[0, "theta1"], args.km)
    if price > 0:
        print("Your car is predicted to be at %d euros." % price)
    else:
        print("The price of your car cannot be estimated.")
