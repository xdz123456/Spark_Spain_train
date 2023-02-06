# calculate pearson correlation matrix
import argparse

from pyspark.mllib.stat import Statistics
from pyspark.sql import SparkSession
import numpy as np
from matplotlib import pyplot as plt
import itertools

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

def execute(dirName):
    spark = SparkSession\
        .builder\
        .master("local[*]")\
        .appName("coursework_dataClean4")\
        .getOrCreate()

    # read csv file
    df = spark.read.csv("./"+dirName+"/ml sample", inferSchema=True, header=True)
    columns = ['origin_id', 'destination_id', 'vehicle_type_id', 'vehicle_class_id', 'label', 'duration_buckets']
    columnNums = len(columns)
    tempDF = df.select(columns)

    # convert to vector column first
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=tempDF.columns, outputCol=vector_col)
    df_vector = assembler.transform(tempDF).select(vector_col)

    # get correlation matrix
    matrix = Correlation.corr(df_vector, vector_col)
    displayableMatrix = matrix.collect()[0]["pearson({})".format(vector_col)].values.reshape((columnNums, columnNums))
    print(displayableMatrix)

    # plot correlation matrix
    # reference: https://blog.csdn.net/weixin_43550531/article/details/106676119
    def plot_confusion_matrix(cm, classes, normalize=False, title='Correlation Matrix', cmap=plt.cm.Blues):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        plt.axis("equal")

        ax = plt.gca()
        left, right = plt.xlim()
        ax.spines['left'].set_position(('data', left))
        ax.spines['right'].set_position(('data', right))
        for edge_i in ['top', 'bottom', 'right', 'left']:
            ax.spines[edge_i].set_edgecolor("white")

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            num = '{:.2f}'.format(cm[i, j]) if normalize else round(float(cm[i, j]), 3)
            plt.text(j, i, num,
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="white" if num > thresh else "black")

        plt.ylabel('Self patt')
        plt.xlabel('Transition patt')

        plt.tight_layout()
        plt.savefig("./" + dirName + "/correlation matrix.png", transparent=True, dpi=800)

        plt.show()


    plot_confusion_matrix(displayableMatrix, columns)

    print("finished")
    spark.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirName", required=True, help="output directory name")
    args = parser.parse_args()

    execute(args.dirName)