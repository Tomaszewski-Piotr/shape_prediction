import streamlit as st
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import pathlib
import os
from joblib import load
import common
from common import log_verbose

file = st.file_uploader(label="Upload file for prediction", accept_multiple_files=False, type=["csv"], help="Select file to perform prediction on")

#if selection made
if file is not None:
    #prepare for possible multiple files

    input_df = common.preprocess_input_file(file)

    shapes_df = input_df.drop(columns=['id'])
    print('Found file with following dimensions: ' + str(shapes_df.shape))

    # normalize input
    X_predict = common.apply_scaler(shapes_df)

    # apply PCA
    X_predict = common.check_apply_pca(X_predict)
    # perform all defined predictions
    for model_full_file in pathlib.Path(common.model_directory()).glob('*' + common.model_suffix):
        model_file = os.path.basename(model_full_file)
        model_name = model_file[:-len(common.model_suffix)]
        log_verbose(' Retrieving prediction model based on ' + model_name + ' from file: ' + str(model_full_file))
        if model_name in common.one_hot_encoded:
            print(model_name + ' currently not supported')
        else:
            clf = load(model_full_file)
            input_df = common.append_predictions(clf, X_predict, model_name, input_df)

    st.dataframe(input_df)


    # provide excel download
    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=True, sheet_name='Results', float_format="%.2f")
        writer.save()
        processed_data = output.getvalue()
        return processed_data


    def get_table_download_link(df):
        val = to_excel(df)
        b64 = base64.b64encode(val)
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="result.xlsx">Download result as Excel file</a>'


    st.markdown(get_table_download_link(input_df), unsafe_allow_html=True)

