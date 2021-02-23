# first neural network with keras tutorial
from numpy import loadtxt
import os
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

# load the dataset
#read in data using pandas
all_df = pd.read_csv('data.csv')
#check data has been read in properly

#create a dataframe with all training data except the target columns
all_X = all_df.drop(columns=['ellipsoid', 'cylinder', 'sphere', 'shape'])
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
pca = PCA(0.95)
pca.fit(X_train)

print("PCA reduction to:", pca.n_components_, "components")
#print(pca.n_samples_)
#print(pca.components_)
#print(pca.explained_variance_)
#print(pca.explained_variance_ratio_)
#print(len(pca.explained_variance_ratio_))

#apply the PCA transform
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
all_X = pca.transform(all_X)


#Prepare classifiers
names = ["Nearest Neighbors", 
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
# Random forest
parameters = {'n_estimators': 120,
              'random_state': 0}

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(**parameters),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

y_pred = {}
accuracy = {}

for name, clf in zip(names, classifiers):
   print("Evaluating: ", name)
   clf.fit(X_train, y_train.values.ravel())
   y_pred[name] = clf.predict(X_test)
   accuracy[name] = metrics.accuracy_score(y_test, y_pred[name])
   all_df[name] = clf.predict(all_X)


for i in X_test_base.index:
    all_df.loc[i, 'test_set'] = "yes"

all_df.to_csv("predictions.csv")

inpath  = "predictions.csv"
outpath = "predictions.zip"
with zipfile.ZipFile(outpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(inpath, os.path.basename(inpath))


for name in names:
    print(name,":", "{0:.0%}\n".format(accuracy[name]))


if os.getenv('CI') == "true":
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), project_qualified_name=os.getenv('NEPTUNE_PROJECT_NAME'))
else:
#    neptune.init('shared/sklearn-integration', api_token='ANONYMOUS')
     exit(0)

neptune.create_experiment(name='shape_prediction')
for name, clf in zip(names, classifiers):
    neptune.log_metric(name, accuracy[name])
    log_confusion_matrix_chart(clf, X_train, X_test, y_train, y_test)  # log confusion matrix chart
    log_precision_recall_chart(clf, X_test, y_test)


neptune.log_artifact('predictions.zip')
neptune.stop()

