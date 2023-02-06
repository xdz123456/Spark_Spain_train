from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.sql.types import IntegerType
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import argparse

def execute(dirName):
    spark = SparkSession\
        .builder\
        .master("local[*]")\
        .appName("OneVsRest")\
        .getOrCreate()
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)

    # read sample csv file
    myDF = spark.read.csv("./"+dirName+"/ml sample", inferSchema=True, header=True)
    myDF.printSchema()

    # select useful
    myDF.select('origin_id', 'destination_id', 'vehicle_type_id', 'vehicle_class_id','duration_buckets', 'label')

    # One-Hot Encoding
    origin_onehoter = OneHotEncoder(inputCol='origin_id', outputCol='origin_vector').fit(myDF)
    destination_onehoter = OneHotEncoder(inputCol='destination_id', outputCol='destination_vector').fit(myDF)
    vehicle_type_onehoter = OneHotEncoder(inputCol='vehicle_type_id', outputCol='vehicle_type_vector').fit(myDF)
    vehicle_class_onehoter = OneHotEncoder(inputCol='vehicle_class_id', outputCol='vehicle_class_vector').fit(myDF)
    duration_buckets_onehoter = OneHotEncoder(inputCol='duration_buckets', outputCol='duration_buckets_vector').fit(myDF)

    myDF = origin_onehoter.transform(myDF)
    myDF = destination_onehoter.transform(myDF)
    myDF = vehicle_type_onehoter.transform(myDF)
    myDF = vehicle_class_onehoter.transform(myDF)
    myDF = duration_buckets_onehoter.transform(myDF)


    assembler = VectorAssembler(inputCols=['origin_vector', 'destination_vector', 'vehicle_type_vector', 'vehicle_class_vector','duration_buckets_vector'], outputCol='features')

    dataSet = assembler.transform(myDF).select(['label', 'features'])
    trainSet, testSet = dataSet.randomSplit([0.75, 0.25])
    print(' train_df shape : (%d , %d)'%(trainSet.count(), len(trainSet.columns)))
    print(' test_df  shape: :(%d , %d)'%(testSet.count(), len(testSet.columns)))

    logic_rec = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

    ovr = OneVsRest(classifier=logic_rec)

    # train the multiclass model.
    ovrModel = ovr.fit(trainSet)



    test_result = ovrModel.transform(testSet)

    tp = test_result.filter(test_result['label'] == test_result['prediction']).count();
    tf = test_result.filter(test_result['label'] != test_result['prediction']).count();


    # test_result.select(['label', 'prediction']).toPandas().to_csv('resultOneVsRest.csv')
    # model_path = "./OVRmodel"
    # ovr.save(model_path)
    data=open("./"+dirName+"/AccerancyOneVsRest.txt",'w+')
    print("Accerancy %f\n" %(tp/(tp+tf)), file = data)
    data.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirName", required=True, help="output directory name")
    args = parser.parse_args()

    execute(args.dirName)
