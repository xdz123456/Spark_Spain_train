import argparse

import Bayes
import DecisionTree
import LogisticRegression
import OneVsRest
import RandomForest


def start_ml(dirName):
    if dirName == None:
        dirName = "./Result"
    Bayes.execute(dirName)
    DecisionTree.execute(dirName)
    LogisticRegression.execute(dirName)
    OneVsRest.execute(dirName)
    RandomForest.execute(dirName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirName", help="output directory name")
    args = parser.parse_args()

    start_ml(args.dirName)
