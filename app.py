from venv import create
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "https://github.com/awesomedeba10",
        'Report a bug': "https://github.com/awesomedeba10/Bengaluru-House-Price-Prediction/issues",
    }
)

st.title('Bengaluru House Price Prediction')

st.sidebar.header('User Input Parameters')

df = load_sidebar_features()

st.subheader('Specified Input Features')
st.write(df)

if 'predicted_val' in st.session_state:
    render_prediction(st.container())
