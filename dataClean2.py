# save number of attributes of columns
import argparse

from pyspark.sql import SparkSession


def execute(dirName):
    spark = SparkSession\
        .builder\
        .master("local[*]")\
        .appName("coursework_dataClean2")\
        .getOrCreate()

    # read original csv file
    rawDF = spark.read.csv("./"+dirName+"/sample", inferSchema=True, header=True)
    columnNames = rawDF.columns

    # count attribute number for each column
    for column in columnNames:
        # get count df
        tempDF1 = rawDF.groupBy(column).count()

        # add new column 'index'
        tempRDD1 = tempDF1.rdd
        tempRDD2 = tempRDD1.zipWithIndex()
        tempDF2 = tempRDD2.toDF()

        # configure column form
        tempDF2 = tempDF2.withColumn(column, tempDF2['_1'].getItem(column))
        tempDF2 = tempDF2.withColumn('count', tempDF2['_1'].getItem('count'))
        tempDF3 = tempDF2.select('_2', column, 'count').withColumnRenamed('_2', 'id')
        tempDF3.show(10)

        tempDF3\
            .coalesce(1)\
            .write\
            .format("com.databricks.spark.csv") \
            .option("header", "true") \
            .save("./"+dirName+"/sample column counts/" + column)

    print("finish")
    spark.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirName", required=True, help="output directory name")
    args = parser.parse_args()

    execute(args.dirName)