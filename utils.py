import streamlit as st
import numpy as np
import pandas as pd
import json, pickle
from definitions import *

def load_sidebar_features():
    feature_form = st.sidebar.form('feature-form')
    location = feature_form.selectbox('Select Location', get_cols(location_only=True))
    total_sqft = feature_form.slider('Total Square Ft.', min_value=0, max_value=15000, value=1000)
    bhk_size = feature_form.slider('Bedroom (BHK)', 0, 20, 4)
    bath = feature_form.slider('No. of Bathroom', 0, 10, 2)
    feature_form.form_submit_button('Predict', on_click=predict, args=[location, total_sqft, bhk_size, bath])

    features = pd.DataFrame({
        'Location': location,
        'Total Square Ft.': total_sqft,
        'Bedroom (BHK)': bhk_size,
        'No. of Bathroom': bath
    }, index=["Features"])

    return features

def predict(location, total_sqft, bhk_size, bath):
    X = np.array(get_cols())
    x = np.zeros(X.shape[0])
    x[0] = np.sqrt(total_sqft)
    x[1] = np.sqrt(bath)
    x[2] = np.sqrt(bhk_size)

    try:
        loc_index = np.where(np.char.lower(X) == location.lower())[0][0]
    except IndexError:
        loc_index = np.where(np.char.lower(X) == 'other')[0][0]
    finally:
        x[loc_index] = 1

    scaler = get_scaler()
    x = scaler.transform(x.reshape(1, -1))

    model = get_model()
    prediction = model.predict(x)**2
    st.session_state.features = {
        'location': location,
        'total_sqft': total_sqft,
        'bath': bath,
        'bhk_size': bhk_size
    }
    st.session_state.predicted_val = prediction[0]

def render_prediction(container):
    container.caption('Closest Actual Price :')
    container.write(get_closest_data())
    container.info(f'Predicted Price: {st.session_state.predicted_val}')

def get_closest_data():
    df = load_data(load_clean=True)
    filtered_df = df.loc[df['location'] == st.session_state.features['location']]

    dist = (filtered_df['total_sqft'] - st.session_state.features['total_sqft']).abs() + \
        (filtered_df['bath'] - st.session_state.features['bath']).abs() + \
        (filtered_df['bhk_size'] - st.session_state.features['bhk_size']).abs()

    closest_row = filtered_df.loc[dist.idxmin()].to_frame().T

    return closest_row.iloc[:, 1:]

@st.experimental_memo
def get_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    return model

@st.experimental_memo
def get_scaler():
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    return scaler

@st.experimental_memo
def get_cols(location_only=False):
    with open(FEATURE_COL_PATH, 'r') as f:
        feature_cols = json.load(f)

    not_a_loc_list = ['total_sqft', 'bath', 'bhk_size']
    if location_only:
        return [x for x in feature_cols['data_columns'] if x not in not_a_loc_list]

    return feature_cols['data_columns']

@st.experimental_memo
def load_data(load_clean=False):
    if load_clean:
        return pd.read_csv(CLEAN_DATA_PATH)

    return pd.read_csv(DATA_PATH)