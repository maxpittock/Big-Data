import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, count, isnan, when
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from functools import reduce
#intialise findspark
findspark.init()
# SparkSession class
spark = SparkSession.builder.getOrCreate()

# function Checks dataframe for nan and null values 
def print_null_nan(data):
    
    print("<<< Finding NAN and NULL values >>>")
    #uses data (df) parameter to count amount of nan or nulls that appear in the dataset
    nannull = data.select([count(when(isnan(i) | col(i).isNull(), i)).alias(i) for i in data.columns])
    print(nannull.show())

# function to store the statistics for task 2
def statistics(data):
    #variable for group
    Group = data.groupBy('Status')

    #find Max column values for each group
    #print("Max")
    max = Group.agg(*[F.max(i).alias(i) for i in data.columns if i not in {"Status"}])

    #find Min column values for each group
    #print("Min")
    min = Group.agg(*[F.min(i).alias(i) for i in data.columns if i not in {"Status"}])

    #Find median value for each group
    #print("Median")
    median = Group.agg(*[F.percentile_approx(i, 0.5).alias(i) for i in data.columns if i not in {"Status"}])

    #Mean column values for each group
    #print("Mean")
    mean = Group.agg(*[F.mean(i).alias(i) for i in data.columns if i not in {"Status"}])

    #Variance column values for each group
    #print("Variance")
    variance = Group.agg(*[F.variance(i).alias(i) for i in data.columns if i not in {"Status"}])

    #print("1st quartile")
    fq = Group.agg(*[F.percentile_approx(i, 0.25).alias(i) for i in data.columns if i not in {"Status"}])
    #display(fq)

    #print("3rd quartile")
    tq = Group.agg(*[F.percentile_approx(i, 0.75).alias(i) for i in data.columns if i not in {"Status"}])
    #display(tq)


    #Mode column values for each group
    #print("Mode")
    #Select all columns
    #columns = data.select([i for i in data.columns])
    #Create seperate df for Normal and Abnoraml groups
    Normal = data.where(data.Status == "Normal")
    Abnormal = data.where(data.Status == "Abnormal")
    #Create list to store normal mode values
    N_list = []
    #loop through columns
    for i in data.columns:
        #Counts highest occuring number in each column, orders by descending and selects the first number from dataset
        N_counts = Normal.groupBy(i).count().orderBy('count', ascending=False).first()[0]
        ##Append items to list as they are found
        N_list.append(N_counts)
    #Create RDD from list
    rdd = spark.sparkContext.parallelize(N_list)
    #convert rdd to spark df
    N_Base_DF = rdd.map(lambda x: (x, )).toDF()
    #Convert to pandas df to tranpose columns to rows
    N_Traspose = N_Base_DF.toPandas().T
    #Change dataframe back to spark
    Normal_Mode = spark.createDataFrame(N_Traspose)

    #Create list to store abnormal mode values
    Ab_list = []
    #loops through columns
    for c in data.columns:
        #Counts highest occuring number in each column, orders by descending and selects the first number from dataset
        Ab_counts = Abnormal.groupBy(c).count().orderBy('count', ascending=False).first()[0]
        ##Append items to list as they are found
        Ab_list.append(Ab_counts)
    #Create RDD from list
    rdd = spark.sparkContext.parallelize(Ab_list)
    #convert rdd to spark df
    AB_Base_DF = rdd.map(lambda x: (x, )).toDF()
    #Convert to pandas df to tranpose columns to rows
    AB_Traspose = AB_Base_DF.toPandas().T
    #Change dataframe back to spark
    Abnormal_Mode = spark.createDataFrame(AB_Traspose)


    #print("testing DF")
    #convert datafram to pandas for boxplot
    boxplot_df = data.toPandas()
    #plot pandas dataframe by status
    boxplot_df.boxplot(by="Status")
    #plt the boxplot
    plt.show()
    
    # put relevent dfs in a list for combining 
    statsdfs2 = [max, min, median, mean, variance, Abnormal_Mode , Normal_Mode, fq, tq]
    #creates new df joining all the statistics together
    Fullstatdf = reduce(DataFrame.unionAll, statsdfs2).toPandas()
    #Rename index so that a user can tell which statistic is which
    FullStatsTable = Fullstatdf.rename(index={0:'Max -', 1:'...', 2:'Min -', 3:'...', 4:'Median -', 5:'...', 6:'Mean -', 7:'...', 8:'Variance -', 9:'...', 10:'Mode', 11: "...", 12:'First quartile', 13: "...", 14:"Third quartile", 15:"..."} )
    #display the new dataframe with all statistics
    print("<<< All statistics >>>")
    display(FullStatsTable)
# Function for creatin a correlation matrix
def correlatonMatrix(data):
    # drop status so i can call matrix without status
    drop_status = data.drop("Status")
    #created list of all columns 
    features = (drop_status.columns)
    #print(features, type(features))

    print("<<< Spark Correlation matrix >>>")
    vector_col = "features"
    #Assembles all data into vector
    V_assembler = VectorAssembler(inputCols=features, outputCol=vector_col)
    #constructs vector to be i 1 column of dataframe
    V_df = V_assembler.transform(data).select(vector_col)
    #Constructs the correlation matrix from the vector dataframe using the pearsons method
    Cor_matrix = Correlation.corr(V_df, vector_col, method='pearson')
    #Collects results as a numpy array
    result_array = Cor_matrix.collect()[0]["pearson({})".format(vector_col)].values
    #Converts array to pandas dataframe
    Cor_matrix_Pandas = pd.DataFrame(result_array.reshape(-1, len(features)), columns=features, index=features)
    #Displays saprk correlation matrix as panads dataframe
    display(Cor_matrix_Pandas)

    #User input for if statement
    Uinput = input("If you would like to find a specific correlation manually please type 'Search' if not please just press enter: ")
    #user input for while loop exit
    exitinput = " "
    #If user types "Search"
    if Uinput == "Search":
        #While "Exit" has no been specified
        while exitinput != 'Exit':
            #Getting 2 user inputs for correlation comparison
            print("<<< Find correlation of 2 features >>>") 
            # Use user input so a customer can search whichever correlation they would like
            UsI1 = input("Please enter 1st column name: ")
            UsI2 = input("Please enter 2nd column name: ")
            #Use user input and find the correlation between the two entered columns 
            UserInput_Feature_Correlation = data.corr(UsI1, UsI2, method='pearson')
            #print the output
            print(UserInput_Feature_Correlation)

            exitinput = input("Press enter to find the correlation between other features or type 'Exit'")
# Function to prepare data (Status column changed to numeric values so they can be used as "Label")
def data_Prep(data):

    #String indexer changes status column to binary integers (Normal = 1 abnormal =0)
    Status_Change = StringIndexer(inputCol="Status", outputCol="Label")
    #Fit model to change the dataframe
    ML_df_fit = Status_Change.fit(data)
    #transform and create new df with "label" column and binary data
    ML_df = ML_df_fit.transform(data)
    #Show new dataframe to check output
    #print("ML dataframe with bianry outputs")
    #ML_df.show()
    #Returns df for use in other funcs
    return ML_df
#Vector seembler function for ML processes
def VecAssembler(data):
    #List of all data columns for use in ML as "features"
    drop_status = data.drop("Status")
    drop_Label = drop_status.drop("Label")
    features = (drop_Label.columns)
    #Assembles all feature information into vector in one column 
    va = VectorAssembler(inputCols = features, outputCol="features")
    #Creates new dataframe with a column holding all feature data as vector
    va_df = va.transform(data)

    #Selects features column and label columns and returns the new df for later use
    va_df = va_df.select(['features', 'Label'])   
    #va_df.show()
    return va_df
#Scaler function for ML processes
def Scaler(Vec_data):
    #Creates standard scalar from data to be used when implementing ML models
    StdS = StandardScaler(inputCol ="features", outputCol="Scaled_features")
    #fit the scaled data 
    Scaledfit = StdS.fit(Vec_data)
    #transform and create new df with scaled features based on the test set using the fitted model
    scaledDataDF = Scaledfit.transform(Vec_data)
    #scaledDataDF.show()
    #return scaleddf for use in other funcs
    return scaledDataDF
#evaluation function for ML processes
def evaluation():
        
    #Create evaluator to find accuracy of ML model
    evaluator = MulticlassClassificationEvaluator( labelCol="Label", predictionCol="prediction", metricName="accuracy")

    #print(type(evaluator), "eval")
    #Returns evaluator for use in other funcs
    return evaluator
#Spliting the data into test and train sets
def Data_split(data):
    #print("Test for data split")
    #randomly split data into 2 dataframes for the train and test set
    train, test = data.randomSplit([0.7, 0.3])
    #checking type
    #print("Training", type(train))
    #print("Testing", type(test))   
    # Print how many rows are in each set 
    print("<<< There are ", train.count(), " rows in the training set, and ", test.count(), " in the test set >>>")

    # return variables to be used by other funcs
    return train, test
#Decsion tree model
def Decision_Tree(train, test):
    
    #Create decision tree by using the columns in the va_df dataset
    dtc = DecisionTreeClassifier(featuresCol="features", labelCol="Label", impurity="gini")
    #fit the model using the decision tree classifier and the training set
    dtc = dtc.fit(train)
    print("<<< Decision Tree prediction dataframe >>>")
    #Use the fitted decision tree model on the test set to get predictions 
    DT_Predict = dtc.transform(test) 
    #Show the decision tree and its predictions in a dataframe
    DT_Predict.show()

    #Returns dataframe so it can be used by other funcs
    return DT_Predict
#Support vector machine model
def Support_Vector_Machine_Model(train, test):
    #Create SVM model and define parameters
    Svm = LinearSVC(maxIter=10, regParam=0.1 , featuresCol="features", labelCol="Label")
   # print(type(Svm), "svm")

    #Fit the training data to the model
    fit_SVM = Svm.fit(train)
    #Transform and create predictions using the fitted training data on the test set
    Predict_DF_SVM = fit_SVM.transform(test)
    #Show final predictions in dataframe
    print("<<< Support vector machine model Predicted outcome >>>")
    Predict_DF_SVM.show()

    #Return predict dataset for use in other funcs  
    return Predict_DF_SVM 
# Function for multilayer perceptron
def Multi_Layer_Perceptron(train, test):
    # specify layer values for the neural network: 12 cause 12 input features and 2 cause we will hav e 2 outputs - 0 or 1
    layers = [12, 5, 4, 2]
    # create the trainer and set its parameters
    Mlp = MultilayerPerceptronClassifier(labelCol="Label", featuresCol="Scaled_features", maxIter=100, layers=layers, blockSize=128)
    #Fit the test data to the model
    Mlp_fit = Mlp.fit(train)
    #Create a new dataframe and predict the results of tes set based of the fitted model
    Mlp_Predict = Mlp_fit.transform(test)
    #Show predicted dataset
    print("<<< Multi layer perceptron predicted outcome >>>")
    Mlp_Predict.show()
    
    #Return MLP_predict for use in other funcs
    return Mlp_Predict
#function for analysis of error rate, sensitivity and specficity for the training models
def ML_analysis(Prediction, evaluator):
    
    #Finds the count for predicted values
    N_predicted = Prediction.where((col("prediction") == 1)).count()
    AB_predicted = Prediction.where((col("prediction") == 0)).count()
    #print("N predicted", N_predicted, "Ab predicted", AB_predicted)

    #Counts true abnormals (abnormals that are predicted to be abnormals)
    TrueAbnormals = Prediction.where((col("prediction")=="0") & (col("Label")==0)).count()
    #print(" PositiveAbnormals: ", TrueAbnormals)
    #Counts true normals (normals that are predicted to be normals)
    TrueNormals = Prediction.where((col("prediction")=="1") & (col("Label")==1)).count()
    #print("PositiveNormals: ",  TrueNormals)
    #Counts false abnormals (normals that are predicted to be abnormals)
    FalseAbnormals = Prediction.where((col("prediction")=="0") & (col("Label")==1)).count()
    #print("NegativeAbnormals: ", FalseAbnormals)
    #Counts false normals (abnormals that are predicted to be normals)
    FalseNormals = Prediction.where((col("prediction")=="1") & (col("Label")==0)).count()
    #print(" NegativeNormals: ", FalseNormals)


    # adding correctly classified samples together
    clasified_samples = N_predicted + AB_predicted
    #Adding falsely classified samples together
    false_clasified_samples = FalseNormals + FalseAbnormals
    #Compute the error rate by dividing the amount of false classified values by the actually calssified values
    Error_Rate = false_clasified_samples / clasified_samples
    #print error rate
    print("<<< Actual error rate >>>")
    print(Error_Rate)

    #First part of equation adding true_Normal and false_Normals together
    Sensitivity_part_1 = TrueNormals + FalseNormals 
    #First part of equation adding true_abnormals and false_abnormals together
    Specifcity_part_1 = TrueAbnormals + FalseAbnormals
    # Construct a sensitivity equation using the previously found information (true Normal predictions / true Normal predictions + false Normal predictions)
    # divide first part of equation by true normal
    Sensitivity = TrueNormals / Sensitivity_part_1
    # Construct a specificity equation using the previously found information (true Abnormal predictions / true Annormal predictions + false Abnormal predictions)
    # divide first part of equation by true normal
    Specificity = TrueAbnormals / Specifcity_part_1 
    #print sensitivity & Specificity
    print("<<< Sensitivity >>>")
    print(Sensitivity)
    print("<<< Specificity >>>")
    print(Specificity)

    #Use evaluator to calculate the accuray of the ML algorithm
    Accuracy = evaluator.evaluate(Prediction)
    print("Accuracy of MLP Classifier is = %g"%(Accuracy))
#Function for finding min, max and mean with mapreduce
def Map_Reduce(data):
    #Map the column data into rdd for min and max calculation
    min_max_rdd = data.rdd.map(lambda x: [x[0],x[1], x[2], x[3], x[4],x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12]])
    #print(rdd.collect())
    # Use reduce on rdd to find the max value of the normal dataset as list
    max_val = min_max_rdd.reduce(lambda a, b: a if a > b else b)
    #print(max_val)
    # Use reduce on rdd to find the min value of the normal dataset as list
    min_val = min_max_rdd.reduce(lambda a, b: a if a < b else b)
    #print(min_val, type(min_val))

    #converts rdd to a spark dataframe 
    convert_max = spark.sparkContext.parallelize(max_val)
    #y = convert_max.join(convert_min)
    max_df = convert_max.map(lambda x: (x, )).toDF()
    # converts spark df to a pandas df and tranposes - switches rows to columns
    Transpose_max_df = max_df.toPandas().T
    #renames index for later concat of data
    Max_df_renamed = Transpose_max_df.rename(index={"_1": "Min"})
    
    #Converts list value "min_val" to an rdd 
    convert_min = spark.sparkContext.parallelize(min_val)
    #converts rdd to a spark dataframe 
    min_df = convert_min.map(lambda x: (x, )).toDF()
    # converts spark df to a pandas df and tranposes - switches rows to columns
    Transpose_min_df = min_df.toPandas().T  
    #renames index for later concat of data
    Min_df_renamed = Transpose_min_df.rename(index={"_1": "Min"})

    #Create rdd storing all columns data and a counter
    mean_rdd = data.rdd.map(lambda x: [x[0],x[1], x[2], x[3], x[4],x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], 1])
    #Map all columns into a list so we can reduceByKey - key = "Normal"
    result = mean_rdd.map(lambda x: (x[0], list(x[1:])))
    #Reduce by key with key being normal - add all rows to one another by column and find the count of all rows
    x2 = result.reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3], x[4]+y[4], x[5]+y[5], x[6]+y[6], x[7]+y[7], x[8]+y[8], x[9]+y[9], x[10]+y[10], x[11]+y[11], x[12]+y[12] ))
    #Divide all added columns by added count - so x/1000000 to get the mean
    mean = x2.mapValues(lambda x: (x[0]/x[12], x[1]/x[12], x[2]/x[12] ,x[3]/x[12], x[4]/x[12], x[5]/x[12], x[6]/x[12], x[7]/x[12], x[8]/x[12], x[9]/x[12], x[10]/x[12], x[11]/x[12]))
           
    #print("please work")
    #Maps all values from rdd into a dataframe
    Mean_spark_df=mean.flatMap(lambda l: [(l[0], value) for value in l[1]]).toDF()
    #Mean_spark_df.show()
    #converts mean df to pandas and transposes it so that rows become columns
    Transpose_mean_df = Mean_spark_df.toPandas().T
    #Insert "Status" column with Normal into pandas datafram at column 0
    Transpose_mean_df.insert(0, "Status", ["0" ,"Normal"], True)
    #rename datafram columns so it is structured and displayed nicely
    mean_renamed = Transpose_mean_df.rename(columns= {"Status": 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11:12}, index={"_2" : "Mean"})
    #After transposing drop first row (this row cotains "normal" over and over so is not needed.)
    Mean_DF = mean_renamed.drop(labels="_1")
    #Prints the dataframe


    #Concats (joins) the min, max and mean dataframe together so you can viewe them in the same dataframe
    Final_df = pd.concat([Max_df_renamed, Min_df_renamed, Mean_DF])    
    # Renames columns to print like the dataset
    Final_df_renamed = Final_df.rename(columns = {0 : "Status" ,1: "Power_range_sensor_1", 2: "Power_range_sensor_2", 3: "Power_range_sensor_3", 4: "Power_range_sensor_4", 5: "Pressure _sensor_1", 6: "Pressure _sensor_2", 7: "Pressure _sensor_3", 8: "Pressure _sensor_4", 9: "Vibration_sensor_1", 10: "Vibration_sensor_2", 11: "Vibration_sensor_3", 12: "Vibration_sensor_4"})
    #print min max df
    print("<<< Max, Min and Mean - With MapReduce >>>")
    print(Final_df_renamed)

#Main function
def main():
    # Create spark dataframe using CSV provided by uni
    df = spark.read.csv('nuclear_plants_small_dataset.csv', inferSchema=True, header=True)
    big_df = spark.read.csv('nuclear_plants_big_dataset.csv', inferSchema=True, header=True)
    #df.orderBy(col=df.columns,  ascending=True)
    # show table
    #df.show()
    # counts rows and shows schema
    #df.printSchema()
    #x = df.count()
    #print(x)

    
    print_null_nan(df)
    statistics(df)
    correlatonMatrix(df)
    ML_df = data_Prep(df)
    va_df = VecAssembler(ML_df)
    scaledDataDF = Scaler(va_df)
    evaluator = evaluation()
    train, test = Data_split(va_df)
    strain, stest = Data_split(scaledDataDF)
    DT_Predict = Decision_Tree(train, test)
    print("<<< Analytics for the Decision tree >>>")
    ML_analysis(DT_Predict, evaluator) # analysis for decision tree
    Predict_DF_SVM = Support_Vector_Machine_Model(strain, stest)
    print("<<< Analytics for SVM >>>")
    ML_analysis(Predict_DF_SVM, evaluator)
    Mlp_Predict = Multi_Layer_Perceptron(strain, stest)
    print("<<< Analytics for MLP >>>")
    ML_analysis(Mlp_Predict, evaluator)
    Map_Reduce(big_df)


if __name__ == '__main__':
    main()
