import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# Streamlit Title
# -----------------------------
st.title("🐶🐱 Dog vs Cat Classifier")
st.write("This app predicts whether an animal is a **Dog or Cat** based on height, weight, ear shape, and tail.")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload dog_cat_dataset.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv('dog_cat_dataset.csv')

    st.subheader("📌 Dataset Preview")
    st.dataframe(df)

    # Label Encoding
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    # Splitting Data
    x = df.drop("Label", axis=1)
    y = df["Label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Train Model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    st.subheader("🔮 Enter Values for Prediction")

    # -----------------------------
    # User Inputs
    # -----------------------------
    height = st.number_input("Enter Height:", min_value=0, value=10)
    weight = st.number_input("Enter Weight:", min_value=0, value=5)
    ear_shape = st.number_input("Enter Ear Shape:", min_value=0, value=1)
    tail = st.number_input("Enter Tail:", min_value=0, value=1)

    if st.button("Predict"):
        y_pred = model.predict([[height, weight, ear_shape, tail]])

        if y_pred[0] == 0:
            st.success("Prediction: 🐱 **Cat**")
        else:
            st.success("Prediction: 🐶 **Dog**")
