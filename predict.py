# Train shape recognition
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pathlib
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
import argparse
from datetime import datetime
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier
from joblib import dump, load
import common
from common import log_verbose
from common import model_file


parser = argparse.ArgumentParser(description='Shape prediction')
parser.add_argument('--input',  action='store', dest='input', required=True,
                    help='Input file with shapes')
parser.add_argument('--output',  action='store', dest='output', required=True,
                    help='Annotated output file with shape classifications')
parser.add_argument('--verbose', '-v', action='store_true', default=False,
                    dest='verbose',
                    help='Set verbose mode')

results = parser.parse_args()

#set common lib to verbose mode
common.verbose = results.verbose


if os.path.isdir(common.model_dir):
    log_verbose('\nModels will be retrieved from :', common.model_dir )
else:
    print("Model directory (", common.model_dir,") missing, existing")
    exit(1)


input_file = results.input

if not os.path.isfile(input_file):
    print("Input file missing")
    exit(1)

# read and pre-process input file
input_df = common.preprocess_input_file(input_file)

shapes_df = input_df.drop(columns=['id'])
log_verbose('Found file with following dimensions: ' + str(shapes_df.shape))

#normalize input
X_predict = common.apply_scaler(shapes_df)

# apply PCA
X_predict = common.check_apply_pca(X_predict)

# perform all defined predictions
for model_full_file in pathlib.Path(common.model_dir).glob('*' + common.model_suffix):
    model_file = os.path.basename(model_full_file)
    model_name = model_file[:-len(common.model_suffix)]
    log_verbose(' Retrieving prediction model based on ' + model_name + ' from file: ' + str(model_full_file))
    if model_name in common.one_hot_encoded:
        print(model_name + ' currently not supported' )
    else:
        clf = load(model_full_file)
        common.append_predictions(clf, X_predict, model_name, input_df)


input_df.to_csv(results.output)

log_verbose('Saved results to ' + results.output)

