import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
st.title("Breast Cancer Prediction")

def train_model():
    x = df[data.feature_names]
    y = df["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    return model, acc


model, accuracy = train_model()

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
st.header("Prediction")
st.write(f"Model: Random Forest — Test Accuracy: **{accuracy:.2%}**")
st.write("Adjust the feature values below to predict whether the tumor is malignant or benign.")

st.sidebar.header("Input Features")
input_data = {}
for col in data.feature_names:
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    col_mean = float(df[col].mean())
    input_data[col] = st.sidebar.slider(
        col, min_value=col_min, max_value=col_max, value=col_mean
    )

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    label = data.target_names[prediction]

    if label == "need treatmeant":
        st.error(f"Prediction: **{label.upper()}**")
    else:
        st.success(f"Prediction: **{label.upper()}**")

    st.write("Prediction Probabilities:")
    prob_df = pd.DataFrame(
        {"Class": data.target_names, "Probability": probabilities}
    )
    st.dataframe(prob_df)
