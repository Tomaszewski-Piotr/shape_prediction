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
import neptune

import zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
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
import matplotlib.pyplot as plt
from PIL import Image
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import confusion_matrix
import seaborn as sn
import shutil
from joblib import dump, load
from common import log_verbose
import common


#start time measurement
start_time = datetime.now()

#parse input
predefined_switches = [
    ('--default', '-d', 'default_classifier', 'Run RandomForest and XGBClassifier'),
    ('--limited', '-l', 'limited_classifier', 'Run RandomForest, XGBClassifier, KNeighbors, NeuralNet '),
    ('--full', '-f', 'all_classifier', 'Run all classifiers (painfully slow and rather pointless)')]

algorithm_switches = [
    ('--rfc',  'rfc_classifier', 'Run RandomForestClassifier'),
    ('--xgb',  'xgb_classifier', 'Run XGBClassifier'),
    ('--xrfc', 'xrfc_classifier', 'Run XGBRFClassifier'),
    ('--mlp',  'mlp_classifier', 'Run MLPClassifier'),
    ('--dct',  'dct_classifier', 'Run DecisionTreeClassifier'),
    ('--ada',  'ada_classifier', 'Run AdaBoostClassifier'),
    ('--gnb',  'gnb_classifier', 'Run GaussianNB'),
    ('--qda',  'qda_classifier', 'Run QuadraticDiscriminantAnalysis'),
    ('--svc',  'svc_classifier', 'Run SVC'),
    ('--lr',   'lr_classifier', 'Run LogisticRegression'),
    ('--knc',  'knc_classifier', 'Run KNeighborsClassifier'),
    ('--nn',   'nn_classifier', 'Run NeuralNet (do not run stacked)')
]

parser = argparse.ArgumentParser(description='Demo for various training shape recognition methods')
parser.add_argument('--pca', '-p', action='store', dest='pca',
                    help='Perform PCA on input data. Values below 1 for explained variability, from 1 number of PCA components')
parser.add_argument('--extended', action='store_true', default=False, dest='extended',
                    help='Produces predictions for the complete data set')
parser.add_argument('--verbose', '-v', action='store_true', default=False,
                    dest='verbose',
                    help='Set verbose mode')
parser.add_argument('--upload', '-u', action='store_true', default=False,
                    dest='upload',
                    help='Upload results to Neptune, NEPTUNE_API_TOKEN must be set in the shell')
parser.add_argument('--from', action='store', dest='start', default=0,
                    help='specify subset of features to be used for training, start point <0,499>')
parser.add_argument('--to', action='store', dest='end', default=499,
                    help='specify subset of features to be used for training, end point <0,499>')
parser.add_argument('--tag', action='store', dest='tag', help='Identifier for trained model. If not specified the default will be replaced')
group = parser.add_mutually_exclusive_group()
for item in predefined_switches:
    group.add_argument(item[0], item[1], action='store_true', dest=item[2], help=item[3])

for item in algorithm_switches:
        parser.add_argument(item[0], action='store_true', dest=item[1], help=item[2])

#add option to run stacked
parser.add_argument('--stacked', '-s', action='store_true', default=False,
                    dest='stc_classifier',
                    help='Additionally run selected classifiers stacked')
results = parser.parse_args()

# pass relevant settings to the common lib
common.verbose = results.verbose
if results.tag:
    common.tag = results.tag

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#prepare output directory
output_dir = common.model_directory()
log_verbose('\nResults will be placed in: ', output_dir)

if os.path.isdir(output_dir):
    log_verbose('\nCleaning output directory')
    shutil.rmtree(output_dir)

if not os.path.isdir(common.model_dir):
    os.mkdir(common.model_dir)

os.mkdir(output_dir)

# concat contents off all data files
log_verbose('\nPreparing data')

content = []
for txt_file in common.get_data_files():
    filename = os.path.basename(txt_file)
    log_verbose(' Retrieving data from: ' + filename)
    # read in data
    shape = common.preprocess_input_file(txt_file)
    #add 'shape' column with shape name
    shape['shape'] = filename[:-len(common.data_suffix)]
    content.append(shape)

all_df = pd.concat(content, axis=0, ignore_index=True)


#create a dataframe with all training data except the target columns
all_X = all_df.drop(columns=['id', 'shape'])

#remove subset if required
log_verbose(' Using features from position: ' + str(results.start) + ' to position ' + str(results.end))
all_X.drop(all_X.iloc[:, int(results.end)+1:], inplace = True, axis = 1)
all_X.drop(all_X.iloc[:, 0:int(results.start)], inplace = True, axis = 1)

#create a dataframe with only the target column
all_y = all_df[['shape']]

#create target in one-hot encoding
# define the keras model
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(all_y.values.ravel())
# get ordinal encoding
ordinal_y = encoder.transform(all_y.values.ravel())
# get one hot encoding
one_hot_y = np_utils.to_categorical(ordinal_y)

#save classes to the file
common.save_encoder(encoder)

#split into test and train
X_train, X_test_base, y_train, y_test, one_hot_y_train, one_hot_y_test, ordinal_y_train, ordinal_y_test = train_test_split(all_X, all_y, one_hot_y, ordinal_y, test_size=0.2) # 80% training and 20% test

#normalize input
scaler = StandardScaler()
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test_base)
all_X = scaler.transform(all_X)
#save the scaler
common.save_scaler(scaler)

# Check if PCA should be done
if results.pca:
    pca_parameter = float(results.pca);
    if pca_parameter >= 1:
        pca_parameter = int(pca_parameter);
    pca = PCA(pca_parameter)
    pca.fit(X_train)
    common.save_pca(pca)
    log_verbose("\nPCA reduction:")
    log_verbose(" Number of selected components: ", pca.n_components_)
    log_verbose(" Explained variance: {0:.0%}".format(pca.explained_variance_ratio_.sum()))
    # apply the PCA transform
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    if results.extended:
        all_X = pca.transform(all_X)


# Random forest
parameters = {'n_estimators': 220,
              'random_state': 0}


# define the keras model for Neural Network
#get number of columns and categories in training data
train_cols = X_train.shape[1]
no_categories = one_hot_y.shape[1]

def baseline_model(train_cols, no_categories):
    def bm():
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=train_cols, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(no_categories, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        return model
    return bm

def baseline_model2(train_cols, no_categories):
    def bm():
        # create model
        model = Sequential([
            Dense(units=256, input_dim=train_cols, activation='relu'),
            Dense(units=192, activation='relu'),
            Dense(units=128, activation='relu'),
            Dense(units=no_categories, activation='softmax')
        ])
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        return model
    return bm


log_verbose('\nPreparing classifiers')
classifiers = []

##default classifiers
if results.all_classifier or results.limited_classifier or results.default_classifier or results.rfc_classifier:
    classifiers.append(('RandomForestClassifier', RandomForestClassifier(**parameters)))

if results.all_classifier or results.limited_classifier or results.default_classifier or results.xgb_classifier:
    classifiers.append(('XGBClassifier', XGBClassifier(use_label_encoder=False, eval_metric = "merror")))

#limited classifiers

if results.all_classifier or results.limited_classifier or results.knc_classifier:
    classifiers.append(('KNeighborsClassifier', KNeighborsClassifier(3)))

if results.all_classifier or results.limited_classifier or results.nn_classifier:
    # compile the keras classifier. Arbitrarily select smaller classifier if data reduction (PCA) applied
    if results.pca:
        keras_estimator = KerasClassifier(build_fn=baseline_model(train_cols, no_categories), epochs=500, batch_size=5,
                                          verbose=0)
    else:
        keras_estimator = KerasClassifier(build_fn=baseline_model2(train_cols, no_categories), epochs=500, batch_size=5,
                                          verbose=0)
    classifiers.append(('NeuralNet', keras_estimator))


#full classifiers
if results.all_classifier or results.xrfc_classifier:
    classifiers.append(('XGBRFClassifier', XGBRFClassifier(use_label_encoder=False, eval_metric = "merror", n_estimators=220)))

if results.all_classifier or results.mlp_classifier:
    classifiers.append(('MLPClassifier', MLPClassifier(alpha=1, max_iter=5000)))

if results.all_classifier or results.dct_classifier:
    classifiers.append(('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=7)))

if results.all_classifier or results.ada_classifier:
    classifiers.append(('AdaBoostClassifier', AdaBoostClassifier()))

if results.all_classifier or results.gnb_classifier:
    classifiers.append(('GaussianNB', GaussianNB()))

if results.all_classifier or results.qda_classifier:
    classifiers.append(('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()))

if results.all_classifier or results.svc_classifier:
    classifiers.append(('SVC', SVC(gamma='auto')))

if results.all_classifier or results.lr_classifier:
    classifiers.append(('LogisticRegression', LogisticRegression(max_iter = 5000)))

if results.stc_classifier:
    classifiers.append(('Stacked', StackingClassifier(estimators=classifiers[:-1], final_estimator=LogisticRegression(max_iter=5000))))

for name, clf in classifiers:
    log_verbose(' Added classifier: ', name)

#list of classifiers where one hot encoding is required
one_hot_encoded = ['NeuralNet']

accuracy = {}
class_probability_names = []
for class_name in encoder.classes_:
    class_probability_names.append(class_name + '_prob')

for name, clf in classifiers:
    log_verbose("\nEvaluating: ", name)
    train_start_time = datetime.now()
    fig, ax = plt.subplots()
    if name in one_hot_encoded:
        clf.fit(X_train, one_hot_y_train)
        y_pred = encoder.inverse_transform(clf.predict(X_test))
        accuracy[name] = metrics.accuracy_score(y_test, y_pred)
        c_m = confusion_matrix(y_test, y_pred, labels = encoder.classes_ )
        df_cfm = pd.DataFrame(c_m, index=encoder.classes_ , columns=encoder.classes_ )
        plt.figure(figsize=(10, 7))
        cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='d')
        plt.savefig(common.model_file(name + ".png"))
        plt.close(fig)
        #clf.save(output_file(name + ".joblib")) does not work not NN
    else:
        cm = ConfusionMatrix(clf, encoder=encoder, is_fitted=False, ax=ax)
        cm.fit(X_train, ordinal_y_train)
        accuracy[name] = cm.score(X_test, ordinal_y_test)
        cm.finalize()
        plt.savefig(common.model_file(name + ".png"))
        plt.close(fig)
        dump(clf, common.model_file(name + ".joblib"))

    train_end_time = datetime.now()

    if results.extended:
        all_df = common.append_predictions(clf, all_X, name, all_df)


    print('', name, "accuracy:", "{0:.0%}".format(accuracy[name]))
    eval_end_time = datetime.now()
    log_verbose(' Training time: {}'.format(train_end_time - train_start_time))
    log_verbose(' Evaluation time: {}\n'.format(eval_end_time - train_end_time))


# save the results file if required
if results.extended:
    for i in X_test_base.index:
        all_df.loc[i, 'test_set'] = "yes"
    #save the results file
    log_verbose('Saving predictions')
    all_df.to_csv(common.model_file("predictions.csv"))

end_time = datetime.now()
log_verbose('Total execution time: {}'.format(end_time - start_time))

if results.upload:
    if os.getenv('CI') == "true":
        log_verbose('Uploading results from CI')
        neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), project_qualified_name=os.getenv('NEPTUNE_PROJECT_NAME'))
    else:
        token = os.getenv('NEPTUNE_API_TOKEN')
        if token:
            log_verbose('Uploading results')
            neptune.init(project_qualified_name='piotrt/shape-prediction', api_token=os.getenv('NEPTUNE_API_TOKEN'))
        else:
            print('NEPTUNE_API_TOKEN must be specified in the shell')
            exit(1)


    def zip_it(in_file, out_file):
        inpath = common.model_file(in_file)
        outpath = common.model_file(out_file)
        with zipfile.ZipFile(outpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(inpath, os.path.basename(inpath))
        #unix style required by naptune
        return outpath.as_posix()

    neptune.create_experiment(name='shape_prediction')
    for name, clf in classifiers:
        log_verbose('Uploading data for: ', name)
        neptune.log_metric(name, accuracy[name])
        # Load image
        image = Image.open(common.model_file(name+".png"))
        neptune.log_image('Confusion matrices', image, image_name=name, description='Confusion matrix for '+name)

        #can't save NN at the moment
        if name not in one_hot_encoded:
            neptune.log_artifact(zip_it(name + ".joblib", name + ".zip"))

        neptune.log_artifact(common.scaler_file.as_posix())
        neptune.log_artifact(common.class_name_file.as_posix())
        if results.pca:
            neptune.log_artifact(common.pca_file.as_posix())

    # if requested zip and add extended results
    if results.extended:
        log_verbose("Zipping and uploading complete prediction result file")
        neptune.log_artifact(zip_it("predictions.csv", "predictions.zip"))

    neptune.stop()

