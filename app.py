import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
st.title("Breast Cancer Prediction")
st.header("Data Exploration")

st.write("Dataset Shape:")
st.write(df.shape)
st.write("Dataset:")
st.dataframe(df)
st.write("Missing Values:")
st.write(df.isnull().sum())

st.write("Target Names:")
st.write(data.target_names)

st.write("Target Distribution:")
st.bar_chart(df["target"].value_counts())

st.write("Feature Distribution:")
feature = st.selectbox("Choose a feature", data.feature_names)
fig, ax = plt.subplots()
ax.hist(df[feature], bins=20)
ax.set_xlabel(feature)
st.pyplot(fig)