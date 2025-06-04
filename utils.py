import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Cache data loading for performance
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('Anti-Reproductive Data.csv')
    return df

def clean_and_engineer_features(df):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=['float', 'int']).columns:
        df_clean[col] = df_clean[col].fillna(0)
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].fillna('Unknown')
    df_clean['DATE'] = pd.to_datetime(df_clean[['YEAR', 'MONTH']].assign(day=1))
    df_clean['QUARTER'] = df_clean['DATE'].dt.quarter
    df_clean['SEASON'] = df_clean['MONTH'].map({12:'Winter', 1:'Winter', 2:'Winter',
                                                3:'Spring', 4:'Spring', 5:'Spring',
                                                6:'Summer', 7:'Summer', 8:'Summer',
                                                9:'Fall', 10:'Fall', 11:'Fall'})
    categorical_cols = ['LOCATION TYPE', 'TYPE OF LOSS', 'PROPERTY CATEGORY', 
                       'S.RACE', 'S.GENDER', 'V.RACE', 'V.GENDER', 'VICTIM TYPE', 
                       'WEAPON', 'DESCRIPTION', 'CHARGE TYPE']
    le = LabelEncoder()
    for col in categorical_cols:
        df_clean[col + '_ENC'] = le.fit_transform(df_clean[col])
    return df_clean

# Example generic countplot utility (for EDA)
def plot_countplot(df, x, hue=None, title='', palette='husl', orient='v', ax=None, figsize=(8,4)):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    # Fix for Seaborn FutureWarning: palette without hue
    if orient == 'v':
        if palette is not None and hue is None:
            sns.countplot(x=x, data=df, ax=ax, palette=palette, hue=x, legend=False)
        else:
            sns.countplot(x=x, hue=hue, data=df, ax=ax, palette=palette)
        ax.set_xlabel(x)
    else:
        if palette is not None and hue is None:
            sns.countplot(y=x, data=df, ax=ax, palette=palette, hue=x, legend=False)
        else:
            sns.countplot(y=x, hue=hue, data=df, ax=ax, palette=palette)
        ax.set_ylabel(x)
    ax.set_title(title)
    return fig