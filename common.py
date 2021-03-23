import pathlib
import os
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np


#list of classifiers where one hot encoding is required
one_hot_encoded = ['NeuralNet']

#set paths
model_dir = 'output'
model_suffix = '.joblib'
data_path = 'data'
data_suffix = '_Iq.csv'

verbose = False

def log_verbose(*args):
    if verbose:
        print(*args)

def model_file(filename):
    return pathlib.Path(model_dir, filename)

def preprocess_input_file(filename):
    input_df = pd.read_csv(filename)
    input_df.rename(columns={input_df.columns[0]: 'id'}, inplace=True)
    return input_df


def apply_scaler(data):
    scaler = load(model_file('std_scaler.bin'))
    return scaler.transform(data)

def check_apply_pca(data):
    if os.path.isfile(model_file('std_scaler.bin')):
        log_verbose('Applying pca')
        pca = load(model_file('pca.bin'))
        return pca.transform(data)


def retrieve_class_names():
    encoder = LabelEncoder()
    encoder.classes_ = np.load(model_file('classes.npy'), allow_pickle=True)
    class_probability_names = []
    for class_name in encoder.classes_:
        class_probability_names.append(class_name + '_prob')
    return encoder, class_probability_names

def append_predictions(clf, data, model_name, output_df):
    encoder, class_probability_names = retrieve_class_names()
    prediction = clf.predict(data)
    output_df[model_name] = encoder.inverse_transform(prediction)
    prob = clf.predict_proba(data)
    df_prob = pd.DataFrame(prob, columns=class_probability_names)
    for current_prob in class_probability_names:
        output_df[current_prob + '_' + model_name] = df_prob[current_prob].values
    return output_df


def get_data_files()
    return pathlib.Path(data_path).glob('*' + data_suffix):