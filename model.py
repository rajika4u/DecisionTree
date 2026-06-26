import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title('Decision Tree Model Classifier')
col1, col2 = st.columns(2)

with col1:
    st.image('negotiating.jpg', width = 200)
with col2:
    st.image('negotiating.jpg', width = 200)

df = pd.read_csv('music.csv')

with st.expander('This is our Machine Learning Model for Decision Tree'):
    st.dataframe(df)

x = df[['age','gender']]
y = df[['genre']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(x,y)

predictions = model.predict(x_test)

score = accuracy_score(y_test, predictions)
score

age = st.slider('Select Age Range', 10,70,18)
#the numbers means, starting the range from number 10 and stop at 70, the slider should be at 18 when not in use
gender = st.selectbox('Select Gender', options = ['Male','Female'])

gender_code = 1 if gender == 'male' else 0

if st.button('Click To Get Classification'):
    prediction = model.predict([[age, gender_code]])
    st.success(f'Correct: **{prediction[0]}**')
