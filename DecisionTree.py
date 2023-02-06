from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.sql.types import IntegerType
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import argparse

def execute(dirName):
    spark = SparkSession\
        .builder\
        .master("local[*]")\
        .appName("LogisticRegression")\
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

    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(dataSet)
    # Automatically identify categorical features, and index them.
        # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dataSet)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = dataSet.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxDepth=30)

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)
    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    # predictions.select("prediction", "indexedLabel", "features").toPandas().to_csv('resultDT.csv')

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    #
    # model_path = "./DecsionTreeModel"
    # model.save(model_path)

    accuracy = evaluator.evaluate(predictions)
    data=open("./"+dirName+"/AccerancyDT.txt",'w+')
    print("Accerancy %f\n" %accuracy, file = data)
    data.close()

    spark.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirName", required=True, help="output directory name")
    args = parser.parse_args()

    execute(args.dirName)
