# first neural network with keras tutorial
from numpy import loadtxt
import os
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import neptune
if os.getenv('CI') == "true":
    from neptunecontrib.monitoring.sklearn import log_confusion_matrix_chart
    from neptunecontrib.monitoring.sklearn import log_precision_recall_chart
import zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
import pathlib
import os


# load the datasets
data_path = 'data'
data_suffix = '_Iq.csv'

# concat contents off all data files
content = []
for txt_file in pathlib.Path(data_path).glob('*' + data_suffix):
    filename = os.path.basename(txt_file)
    print('Retrieving data from: ' + filename)
    # read in data using pandas
    shape = pd.read_csv(txt_file)
    #add 'shape' column with shape name
    shape['shape'] = filename[:-len(data_suffix)]
    content.append(shape)

all_df = pd.concat(content, axis=0, ignore_index=True)
#call the first column 'id'
all_df.rename(columns = {all_df.columns[0]: 'id'}, inplace = True)

#create a dataframe with all training data except the target columns
all_X = all_df.drop(columns=['id', 'shape'])

#create a dataframe with only the target column
all_y = all_df[['shape']]

X_train, X_test_base, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2) # 80% training and 20% test

scaler = StandardScaler()

# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test_base)
all_X = scaler.transform(all_X)


# Make an instance of the Model
pca = PCA(25)
pca.fit(X_train)

print("PCA reduction to:", pca.n_components_, "components")
#print(pca.n_samples_)
#print(pca.components_)
#print(pca.explained_variance_)
#print(pca.explained_variance_ratio_)
#print(len(pca.explained_variance_ratio_))

#apply the PCA transform
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)
#all_X = pca.transform(all_X)


#Prepare classifiers
names = ["Nearest Neighbors", 
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
# Random forest
parameters = {'n_estimators': 220,
              'random_state': 0}

classifiers = [
    ('KNeighborsClassifier', KNeighborsClassifier(3)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=7)),
    ('RandomForestClassifier', RandomForestClassifier(**parameters)),
    ('MLPClassifier', MLPClassifier(alpha=1, max_iter=5000)),
    ('AdaBoostClassifier', AdaBoostClassifier()),
    ('GaussianNB', GaussianNB()),
    ('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()),
    ('SVC', SVC(gamma='auto')),
    ('LogisticRegression', LogisticRegression(max_iter = 5000))
]

classifiers.append(('Stacked', StackingClassifier(estimators=classifiers[:-1], final_estimator=LogisticRegression(max_iter = 5000))))

y_pred = {}
accuracy = {}

for name, clf in classifiers:
    print("Evaluating: ", name)
    clf.fit(X_train, y_train.values.ravel())
    y_pred[name] = clf.predict(X_test)
    accuracy[name] = metrics.accuracy_score(y_test, y_pred[name])
    all_df[name] = clf.predict(all_X)
    print(name, ":", "{0:.0%}\n".format(accuracy[name]))

for i in X_test_base.index:
    all_df.loc[i, 'test_set'] = "yes"

all_df.to_csv("predictions.csv")
np.savetxt("pca.csv", all_X, delimiter=",")
inpath  = "predictions.csv"
outpath = "predictions.zip"
with zipfile.ZipFile(outpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(inpath, os.path.basename(inpath))


if os.getenv('CI') == "true":
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), project_qualified_name=os.getenv('NEPTUNE_PROJECT_NAME'))
else:
    neptune.init(project_qualified_name='piotrt/shape-prediction', api_token=os.getenv('NEPTUNE_API_TOKEN'))


neptune.create_experiment(name='shape_prediction')
for name, clf in classifiers:
    neptune.log_metric(name, accuracy[name])
    if os.getenv('CI') == "true":
        log_confusion_matrix_chart(clf, X_train, X_test, y_train, y_test)  # log confusion matrix chart
        log_precision_recall_chart(clf, X_test, y_test)

neptune.log_artifact('predictions.zip')
neptune.stop()

