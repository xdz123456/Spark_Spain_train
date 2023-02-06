# change attribute names for machine learning
import argparse

from pyspark.ml.feature import StringIndexer, Bucketizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, to_timestamp
from pyspark.sql.types import IntegerType


def execute(dirName):
    spark = SparkSession\
        .builder\
        .master("local[*]")\
        .appName("coursework_dataClean3")\
        .getOrCreate()

    # read sample csv file
    sampleDF = spark.read.csv("./"+dirName+"/sample", inferSchema=True, header=True)

    # convert row attributes of ['origin', 'destination', 'vehicle_type', 'vehicle_class'] to ids
    stringColumnNames = ['origin', 'destination', 'vehicle_type', 'vehicle_class']
    tempDF = sampleDF
    for columnName in stringColumnNames:
        si = StringIndexer(inputCol=columnName, outputCol=columnName + '_id')
        model = si.fit(tempDF)
        tempDF = model.transform(tempDF)

        # tempDF = tempDF.withColumn(columnName, tempDF[columnName].cast(IntegerType()))

        print("column " + columnName + " finished")

    # calculate price approximate quantiles
    priceNumQ = 15
    priceValueQ = 1 / priceNumQ
    priceAQs = tempDF.stat.approxQuantile('price', [priceValueQ * indexQ for indexQ in range(1, priceNumQ)], 0)
    # divide price to buckets
    splits1 = [-float('inf')]
    for AQ in priceAQs:
        # avoid duplication
        for e in splits1:
            if e == AQ:
                AQ = AQ + 0.0001
        splits1.append(AQ)
    splits1.append(float('inf'))
    br1 = Bucketizer(splits=splits1, inputCol='price', outputCol='label')
    tempDF = br1.transform(tempDF)

    # calculate duration approximate quantiles
    durationNumQ = 10
    durationValueQ = 1 / durationNumQ
    durationAQs = tempDF.stat.approxQuantile('duration', [durationValueQ * indexQ for indexQ in range(1, durationNumQ)], 0)

    splits2 = [-float('inf')]
    for AQ in durationAQs:
        splits2.append(AQ)
    splits2.append(float('inf'))
    br2 = Bucketizer(splits=splits2, inputCol='duration', outputCol='duration_buckets')
    tempDF = br2.transform(tempDF)

    # change column types
    timeColumnNames = ['departure', 'arrival']
    for columnName in timeColumnNames:
        tempDF = tempDF.withColumn(columnName, to_timestamp(tempDF[columnName]))
    intColumnNames = ['origin_id', 'destination_id', 'vehicle_type_id', 'vehicle_class_id', 'label', 'duration_buckets']
    for columnName in intColumnNames:
        tempDF = tempDF.withColumn(columnName, tempDF[columnName].cast(IntegerType()))
    tempDF.printSchema()
    
    # draw the distribution diagram of 'label'
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.array(tempDF.select("price").collect()).reshape(-1)
    max = np.max(x)
    min = np.min(x)
    tempPriceAQs = priceAQs
    tempPriceAQs.append(max)
    tempPriceAQs = [min] + tempPriceAQs
#    plt.hist(x, bins=tempPriceAQs, histtype='bar', ec='black')
    plt.hist(x, bins=50, color='b', alpha=0.5)
    for pos in priceAQs:
        plt.vlines(x=pos, ymin=0, ymax=70000, lw=1.0, color='r')
    plt.savefig("./"+dirName+"/histogram.png")
#    plt.show()
 
    # save sample dataframe
    tempDF.repartition(1).write.csv("./"+dirName+"/ml sample", encoding="utf-8", header=True)

    print("finished")
    spark.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirName", required=True, help="output directory name")
    args = parser.parse_args()

    execute(args.dirName)

