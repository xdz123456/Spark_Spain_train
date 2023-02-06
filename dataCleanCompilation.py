import argparse

import dataClean1
import dataClean2
import dataClean3
import dataClean4


def startDataClean(dirName):
    if dirName == None:
        dirName = "./Result"
    # execute data cleaning
    dataClean1.execute(dirName)  # delete unwanted attributes, invalid rows; shrink dataset
    dataClean2.execute(dirName)  # save number of attributes of columns for further data analysis
    dataClean3.execute(dirName)  # change dataset for machine learning algorithms
    dataClean4.execute(dirName)  # calculate pearson correlation matrix for features

    print("Data clean finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirName", help="output directory name")
    args = parser.parse_args()

    startDataClean(args.dirName)
