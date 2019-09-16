import time
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from skdist.distribute.search import DistGridSearchCV
from pyspark.sql import SparkSession 
# instantiate spark session
spark = (   
    SparkSession    
    .builder    
    .getOrCreate()    
    )
sc = spark.sparkContext 
# the digits dataset
digits = datasets.load_digits()
X = digits["data"]
y = digits["target"] 
# create a classifier: a support vector classifier
classifier = svm.SVC()
param_grid = {
    "C": [0.01, 0.01, 0.1, 1.0, 10.0, 20.0, 50.0], 
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1], 
    "kernel": ["rbf", "poly", "sigmoid"]
    }
scoring = "f1_weighted"
cv = 10
# hyperparameter optimization
start = time.time()

model_DIST = DistGridSearchCV(    
    classifier, param_grid,     
    sc=sc, cv=cv, scoring=scoring,
    verbose=True    
    )
model = GridSearchCV(
    classifier, param_grid,
    cv=cv, scoring=scoring,
    verbose=True
    )

model.fit(X,y)
print("Train time for scikit-learn: {0}".format(time.time() - start))
print("Best score: {0}".format(model.best_score_))
print("Train time for sk-dist: {0}".format(time.time() - start))
print("Best score: {0}".format(model_DIST.best_score_))
