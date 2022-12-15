import streamlit as st
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("Health_insurance.csv")

col1, col2, col3 = st.columns(3)


def predict_val(arr):
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    val = model.predict(arr)
    return val


col1.dataframe(df.head())
col2.dataframe(df.tail())
col3.dataframe(df.head())

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["a", "b", "c"])

with st.form("Predict"):
    st.header("input the values")
    age = st.text_input("enter the age")
    bmi = st.text_input("enter the bmi")
    children = st.text_input("enter the children ")
    submit = st.form_submit_button("Upload")

if submit:
    st.markdown(f"## prediction = {predict_val([[int(age), int(bmi), int(children)]])}")
    st.markdown(age)
    st.markdown(age)
