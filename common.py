import pathlib
import os
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np

#default to no verbose, change if needed
verbose = False

#list of classifiers where one hot encoding is required
one_hot_encoded = ['NeuralNet']

#set paths
model_dir = 'models'
model_suffix = '.joblib'
data_path = 'data'
data_suffix = '_Iq.csv'

def model_file(filename):
    return pathlib.Path(model_dir, filename)

scaler_file = model_file('std_scaler.bin')
pca_file = model_file('pca.bin')
class_name_file = model_file('classes.npy')



def log_verbose(*args):
    if verbose:
        print(*args)

def preprocess_input_file(filename):
    input_df = pd.read_csv(filename)
    input_df.rename(columns={input_df.columns[0]: 'id'}, inplace=True)
    return input_df


def apply_scaler(data):
    scaler = load(scaler_file)
    return scaler.transform(data)

def save_scaler(scaler):
    dump(scaler, scaler_file, compress=True)


def check_apply_pca(data):
    if os.path.isfile(pca_file):
        log_verbose('Applying pca')
        pca = load(pca_file)
        return pca.transform(data)

def save_pca(pca):
    dump(pca, pca_file, compress=True)



def retrieve_class_names():
    encoder = LabelEncoder()
    encoder.classes_ = np.load(class_name_file, allow_pickle=True)
    class_probability_names = []
    for class_name in encoder.classes_:
        class_probability_names.append(class_name + '_prob')
    return encoder, class_probability_names

def save_encoder(encoder):
    np.save(class_name_file, encoder.classes_)


def append_predictions(clf, data, model_name, output_df):
    encoder, class_probability_names = retrieve_class_names()
    prediction = clf.predict(data)
    output_df[model_name] = encoder.inverse_transform(prediction)
    prob = clf.predict_proba(data)
    df_prob = pd.DataFrame(prob, columns=class_probability_names)
    for current_prob in class_probability_names:
        output_df[current_prob + '_' + model_name] = df_prob[current_prob].values
    return output_df


def get_data_files():
    return pathlib.Path(data_path).glob('*' + data_suffix)