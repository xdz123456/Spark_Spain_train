# delete unwanted columns, invalid rows; shrink dataset
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id


def execute(dirName):
    spark = SparkSession \
        .builder \
        .master("local[*]") \
        .appName("coursework_dataClean1") \
        .getOrCreate()

    # read original csv file
    rawDF = spark.read.csv("BD15 raw data.csv", inferSchema=True, header=True)

    # drop unwanted columns (company, seats, insert_date, meta)
    tempDF1 = rawDF.drop("company").drop("seats").drop("insert_date").drop("meta").drop("id")
    tempDF1.show(10)

    # select rows that:
    # 1. any of missing columns have missing value (vehicle_class, price, fare)
    # 2. the attribute of 'fare' is 'Flexible'
    tempDF1.createOrReplaceTempView("tempTV1")
    tempDF2 = spark.sql("""
    SELECT origin, destination, departure, arrival, duration, vehicle_type, vehicle_class, price
    FROM tempTV1
    WHERE 
    vehicle_class IS NOT NULL
    AND price IS NOT NULL
    AND price <> 0
    AND fare IS NOT NULL
    
    AND fare = 'Flexible'
    ORDER BY price
    """)

    # shrink dataframe
    sampleDF = tempDF2.sample(fraction=0.05, seed=1)

    # save sample dataframe
    sampleDF.repartition(1).write.csv("./"+dirName+"/sample", encoding="utf-8", header=True)

    print("finish")
    spark.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirName", required=True, help="output directory name")
    args = parser.parse_args()

    execute(args.dirName)
