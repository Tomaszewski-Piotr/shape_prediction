# first neural network with keras tutorial
from numpy import loadtxt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor
from neptunecontrib.monitoring.sklearn import log_confusion_matrix_chart


# load the dataset
#read in data using pandas
all_df = pd.read_csv('data.csv')
#check data has been read in properly
print(all_df.head())

#create a dataframe with all training data except the target columns
all_X = all_df.drop(columns=['ellipsoid', 'cylinder', 'sphere'])

#create a dataframe with only the target column
all_y = all_df[['ellipsoid', 'cylinder', 'sphere']]

X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2) # 80% training and 20% test

# Random forest
parameters = {'n_estimators': 120,
              'random_state': 0}

clf = RandomForestClassifier(**parameters)
clf.fit(X_train, y_train)

# Connect your script to Neptune

y_pred = clf.predict(X_test)
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred))
feature_imp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
print(feature_imp)

if os.getenv('CI') == "true":
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), project_qualified_name=os.getenv('NEPTUNE_PROJECT_NAME'))
else:
    neptune.init('shared/sklearn-integration', api_token='ANONYMOUS')

neptune.create_experiment(name='shape_prediction')
log_confusion_matrix_chart(clf, X_train, X_test, y_train, y_test)  # log confusion matrix chart
neptune.stop()

