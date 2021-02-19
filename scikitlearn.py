# first neural network with keras tutorial
from numpy import loadtxt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import neptune
if os.getenv('CI') == "true":
    from neptunecontrib.monitoring.keras import NeptuneMonitor
    from neptunecontrib.monitoring.sklearn import log_confusion_matrix_chart
    from neptunecontrib.monitoring.sklearn import log_precision_recall_chart
    from neptunecontrib.monitoring.sklearn import log_scores
import zipfile
import shutil

from sklearn.metrics import multilabel_confusion_matrix

# load the dataset
#read in data using pandas
all_df = pd.read_csv('data.csv')
#check data has been read in properly

#create a dataframe with all training data except the target columns
all_X = all_df.drop(columns=['ellipsoid', 'cylinder', 'sphere', 'shape'])
#create a dataframe with only the target column
all_y = all_df[['shape']]

X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2) # 80% training and 20% test

# Random forest
parameters = {'n_estimators': 120,
              'random_state': 0}

clf = RandomForestClassifier(**parameters)
clf.fit(X_train, y_train.values.ravel())
y_pred_RFC = clf.predict(X_test)
accuracy_RFC = metrics.accuracy_score(y_test, y_pred_RFC)


#res = pd.DataFrame(y_pred)
#res.index = X_test.index # its important for comparison
#res.columns = ["prediction"]
#res.to_csv("prediction.csv")
y_pred_all_RFC = clf.predict(all_X)

clf2 = KNeighborsClassifier(5)
clf2.fit(X_train, y_train.values.ravel())
y_pred_KNC = clf2.predict(X_test)
accuracy_KNC = metrics.accuracy_score(y_test, y_pred_KNC)
y_pred_all_KNC = clf2.predict(all_X)


all_df['RFG'] = y_pred_all_RFC
all_df['KNC'] = y_pred_all_KNC
for i in X_test.index:
    all_df.loc[i, 'test_set'] = "yes"
all_df.to_csv("predictions.csv")

inpath  = "predictions.csv"
outpath = "predictions.zip"
with zipfile.ZipFile(outpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(inpath, os.path.basename(inpath))


print("Random Forest Accuracy:", accuracy_RFC, "\n")
print("KNC Accuracy:", accuracy_KNC, "\n")
#feature_imp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
#print(feature_imp)


if os.getenv('CI') == "true":
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), project_qualified_name=os.getenv('NEPTUNE_PROJECT_NAME'))
else:
#    neptune.init('shared/sklearn-integration', api_token='ANONYMOUS')
     exit(0)

neptune.create_experiment(name='shape_prediction')
neptune.log_metric('Random Forest Accuracy', accuracy_RFC)
neptune.log_metric('KNC Accuracy', accuracy_KNC)

log_confusion_matrix_chart(clf, X_train, X_test, y_train, y_test)  # log confusion matrix chart
log_confusion_matrix_chart(clf2, X_train, X_test, y_train, y_test)  # log confusion matrix chart
log_precision_recall_chart(clf, X_test, y_test)
log_precision_recall_chart(clf2, X_test, y_test)
#log_scores(clf, X_test, y_test, name='testRF')
#log_scores(clf2, X_test, y_test, name='testKN')

neptune.log_artifact('predictions.zip')
neptune.stop()

