import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("Health_insurance.csv")

df = df[['age', 'bmi', 'children', 'charges']]

df = df.fillna(0)

X = df[['age', 'bmi', 'children']].values
y = df['charges'].values

lr = LinearRegression()
lr.fit(X, y)

with open('model.pkl', 'wb') as file:
    pickle.dump(lr, file)

